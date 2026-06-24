"""
跨数据集泛化实验评测脚本。

用法示例：
    python crackfm/eval_cross_dataset.py \
        --weight /path/to/duflow.pth \
        --data_root /path/to/CSF-50K/testing \
        --dataset csf50k \
        --batch_size 8 \
        --use_cal_only

支持的 dataset 类型：
    csf50k, cracktree260, crack500, crackls315, cfd, generic
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让 crackfm 包可被导入：脚本位于 crackfm/ 下，父目录是 CrackSeg 根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crackfm.data import CrackDataset, CSF50KDataset, prepare_mask
from crackfm.models import DUFlowInferencer


def parse_args():
    parser = argparse.ArgumentParser(description="DUFlow cross-dataset evaluation")
    parser.add_argument("--weight", type=str, required=True, help="DUFlow 预训练权重路径")
    parser.add_argument("--data_root", type=str, required=True, help="目标数据集根目录")
    parser.add_argument("--dataset", type=str, default="generic",
                        choices=["csf50k", "cracktree260", "crack500", "crackls315", "cfd", "deepcrack", "generic"],
                        help="数据集类型，用于选择默认目录结构")
    parser.add_argument("--split", type=str, default="test",
                        help="数据划分，generic 数据集使用；CSF-50K 默认 testing")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10,
                        help="Flow Matching 采样步数，use_cal_only 时无效")
    parser.add_argument("--use_cal_only", action="store_true",
                        help="只使用 cal 分支快速推理")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bfscore_boundary_px", type=float, default=2.0,
                        help="BFscore 边界匹配距离阈值（像素）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_pred", type=str, default=None,
                        help="预测结果保存目录，不指定则不保存")
    parser.add_argument("--fusion_type", type=str, default="concat")
    parser.add_argument("--base_channels", type=int, default=64)
    return parser.parse_args()


def build_dataset(args):
    if args.dataset == "csf50k":
        return CSF50KDataset(
            root_dir=args.data_root,
            split="testing",
            image_size=(args.image_size, args.image_size),
        )
    # CRACK500 和 DeepCrack 的 mask 目录名为 cracks/
    if args.dataset in ("crack500", "deepcrack"):
        return CrackDataset(
            root_dir=args.data_root,
            split="test",
            image_size=(args.image_size, args.image_size),
        )
    return CrackDataset(
        root_dir=args.data_root,
        split=args.split if args.dataset == "generic" else "test",
        image_size=(args.image_size, args.image_size),
    )


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, bf_boundary_px: float = 2.0):
    """计算 IoU、Dice、F1、Precision、Recall、clDice、BFscore。"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target_bin)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target_bin).sum(dim=(1, 2, 3))

    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    cldice = compute_cldice(pred_bin, target_bin)
    bfscore = compute_bfscore(pred_bin, target_bin, boundary_threshold=bf_boundary_px)

    return {
        "iou": iou,
        "dice": dice,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "cldice": cldice,
        "bfscore": bfscore,
    }


def compute_cldice(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """
    clDice (centerline Dice) for a batch.
    clDice = 2 * T_prec * T_sens / (T_prec + T_sens)
    T_prec  = |Pred_skeleton ∩ GT| / |Pred_skeleton|
    T_sens  = |GT_skeleton ∩ Pred| / |GT_skeleton|
    """
    B = pred_bin.shape[0]
    device = pred_bin.device
    scores = torch.zeros(B, device=device)

    pred_np = pred_bin.cpu().numpy().astype(bool)
    target_np = target_bin.cpu().numpy().astype(bool)

    for i in range(B):
        p = pred_np[i, 0]
        t = target_np[i, 0]

        if p.sum() == 0 and t.sum() == 0:
            scores[i] = 1.0
            continue
        if p.sum() == 0 or t.sum() == 0:
            scores[i] = 0.0
            continue

        p_skel = skeletonize(p)
        t_skel = skeletonize(t)

        t_prec = (p_skel & t).sum() / (p_skel.sum() + 1e-6)
        t_sens = (t_skel & p).sum() / (t_skel.sum() + 1e-6)
        scores[i] = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-6)

    return scores


def compute_bfscore(pred_bin: torch.Tensor, target_bin: torch.Tensor, boundary_threshold: float = 2.0) -> torch.Tensor:
    """
    BFscore (Boundary F-score) for a batch.
    提取 pred 和 target 的边界像素，计算边界 precision/recall，再求 F1。
    """
    B = pred_bin.shape[0]
    device = pred_bin.device
    scores = torch.zeros(B, device=device)

    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    target_np = target_bin.cpu().numpy().astype(np.uint8)

    for i in range(B):
        p = pred_np[i, 0]
        t = target_np[i, 0]

        if p.sum() == 0 and t.sum() == 0:
            scores[i] = 1.0
            continue
        if p.sum() == 0 or t.sum() == 0:
            scores[i] = 0.0
            continue

        p_boundary = p - ndimage.binary_erosion(p, structure=np.ones((3, 3))).astype(np.uint8)
        t_boundary = t - ndimage.binary_erosion(t, structure=np.ones((3, 3))).astype(np.uint8)

        if p_boundary.sum() == 0 or t_boundary.sum() == 0:
            scores[i] = 0.0
            continue

        dt_target = ndimage.distance_transform_edt(1 - t_boundary)
        dt_pred = ndimage.distance_transform_edt(1 - p_boundary)

        p_boundary_pts = p_boundary.astype(bool)
        t_boundary_pts = t_boundary.astype(bool)

        precision = (dt_target[p_boundary_pts] <= boundary_threshold).mean()
        recall = (dt_pred[t_boundary_pts] <= boundary_threshold).mean()

        scores[i] = 2.0 * precision * recall / (precision + recall + 1e-6)

    return scores


def save_prediction(pred: torch.Tensor, name: str, save_dir: str, image_size: int):
    """将预测结果保存为 0/255 灰度图。"""
    os.makedirs(save_dir, exist_ok=True)
    arr = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    import cv2
    cv2.imwrite(os.path.join(save_dir, name), arr)


def main():
    args = parse_args()

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    inferencer = DUFlowInferencer(
        weight_path=args.weight,
        image_size=args.image_size,
        device=args.device,
        sample_steps=args.sample_steps,
        use_cal_only=args.use_cal_only,
        fusion_type=args.fusion_type,
        base_channels=args.base_channels,
    )

    all_metrics = {"iou": [], "dice": [], "f1": [], "precision": [], "recall": [], "cldice": [], "bfscore": []}

    pbar = tqdm(loader, desc=f"Eval {args.dataset}")
    for imgs, masks, names in pbar:
        masks = prepare_mask(masks, args.image_size).to(inferencer.device)
        preds = inferencer.predict(imgs)

        metrics = compute_metrics(preds, masks, args.threshold, args.bfscore_boundary_px)
        for k in all_metrics:
            all_metrics[k].extend(metrics[k].cpu().numpy().tolist())

        if args.save_pred:
            for pred, name in zip(preds, names):
                save_prediction(pred.unsqueeze(0), name, args.save_pred, args.image_size)

        mean_iou = np.mean(all_metrics["iou"])
        pbar.set_postfix({"IoU": f"{mean_iou:.4f}"})

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    summary["dataset"] = args.dataset
    summary["data_root"] = args.data_root
    summary["n_samples"] = len(dataset)
    summary["use_cal_only"] = args.use_cal_only
    summary["sample_steps"] = args.sample_steps if not args.use_cal_only else None

    print("\n" + "=" * 50)
    print(f"Cross-dataset evaluation on {args.dataset}")
    print("=" * 50)
    for k in ["iou", "dice", "f1", "precision", "recall", "cldice", "bfscore"]:
        print(f"  {k.upper():12s}: {summary[k]:.4f}")
    print("=" * 50)

    result_path = os.path.join(
        os.path.dirname(args.weight),
        f"eval_{args.dataset}_{'cal' if args.use_cal_only else 'flow'}.json",
    )
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
