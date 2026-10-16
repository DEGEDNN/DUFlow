"""
DualBranchParallelFlowModel 评测脚本。

评测两个分割输出：
1. cal  — 特征分支直接输出，已经过 sigmoid，不需要 ODE 采样，速度快
2. flow — 流匹配 Euler ODE 采样结果（t: 0→1，从噪声积分到预测掩码）

使用示例：
    python -m test.eval_dual_branch_parallel --data-root dataset --test-split Test --checkpoint checkpoints/dual_branch_parallel/best.pt --output-dir eval_results/dual_branch_parallel --sample-steps 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_matching.dual_branch_model_parallel import DualBranchParallelFlowModel  # noqa: E402
from test.metrics import BinarySegmentationStats, update_stats_from_logits  # noqa: E402


def _solve_global_oracle(records: list[dict], eps: float = 1e-12) -> tuple[list[bool], dict]:
    """Exactly maximize dataset-level micro-IoU over one branch choice per image.

    Dinkelbach's fractional-programming update reduces each iteration to an
    independent per-image decision: choose the branch with larger TP-lambda*U,
    where U = TP + FP + FN.  The returned booleans are True for FLOW and False
    for CAL.  Since micro-Dice is a strictly increasing transform of micro-IoU,
    the same choices maximize both metrics.
    """
    if not records:
        return [], {"iterations": 0, "cal_selected": 0, "flow_selected": 0}

    lam = 0.0
    previous: list[bool] | None = None
    selected: list[bool] = [False] * len(records)

    for iteration in range(1, 101):
        selected = []
        total_tp = 0.0
        total_union = 0.0
        for record in records:
            score_cal = record["tp_cal"] - lam * record["union_cal"]
            score_flow = record["tp_flow"] - lam * record["union_flow"]
            use_flow = score_flow > score_cal  # ties deliberately fall back to CAL
            selected.append(use_flow)
            suffix = "flow" if use_flow else "cal"
            total_tp += record[f"tp_{suffix}"]
            total_union += record[f"union_{suffix}"]

        new_lam = (total_tp + eps) / (total_union + eps)
        if selected == previous or abs(new_lam - lam) <= eps:
            lam = new_lam
            break
        previous = selected
        lam = new_lam

    total_pred = 0.0
    total_target = 0.0
    total_tp = 0.0
    for record, use_flow in zip(records, selected):
        suffix = "flow" if use_flow else "cal"
        total_tp += record[f"tp_{suffix}"]
        total_pred += record[f"pred_{suffix}"]
        total_target += record["target"]

    stats = BinarySegmentationStats(
        intersection=total_tp,
        pred_sum=total_pred,
        target_sum=total_target,
    )
    details = {
        "iterations": iteration,
        "cal_selected": selected.count(False),
        "flow_selected": selected.count(True),
        "objective_iou": lam,
    }
    return selected, {**stats.compute(), **details}


def parse_args():
    parser = argparse.ArgumentParser(description="评测 DualBranchParallelFlowModel 的 Dice / IoU")
    parser.add_argument("--data-root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--test-split", type=str, default="Test", help="测试子目录名")
    parser.add_argument("--checkpoint", type=str, required=True, help="best.pt 路径")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="结果输出目录")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 建议保持 0")
    parser.add_argument("--image-size", type=int, default=256, help="必须与训练时一致")
    parser.add_argument("--base-channels", type=int, default=64, help="必须与训练时一致")
    parser.add_argument(
        "--fusion-type",
        choices=["concat", "add", "attention"],
        default="concat",
        help="必须与训练时一致",
    )
    parser.add_argument("--sample-steps", type=int, default=1, help="Euler ODE 积分步数；论文默认 1")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--seed", type=int, default=20260817, help="ODE sampling seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def build_loader(args: argparse.Namespace) -> DataLoader:
    from flow_matching.dataset_loader import CustomDataset

    dataset = CustomDataset(
        args=None,
        data_path=str(Path(args.data_root) / args.test_split),
        transform=transforms.ToTensor(),
        mode="Test",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# 预处理（与 train_dual_branch_parallel.py 保持逐字一致）
# ---------------------------------------------------------------------------

def prepare_condition(img: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert dataset image tensors to [B, 6, H, W]."""
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    img = img.float()

    if img.ndim == 3:
        img = img.unsqueeze(1)
    if img.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got shape {tuple(img.shape)}")

    # Common path for this dataset after ToTensor(): [B, 1, H, 6].
    if img.shape[1] == 1 and img.shape[-1] == 6:
        img = img[:, 0].permute(0, 2, 1).unsqueeze(-1)
        return F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)

    if img.shape[1] == 6:
        if img.shape[-2:] != (image_size, image_size):
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img

    if img.shape[-1] == 6:
        img = img.permute(0, 3, 1, 2)
        if img.shape[-2:] != (image_size, image_size):
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img

    if img.shape[1] == 1:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img.repeat(1, 6, 1, 1)

    raise ValueError(f"Cannot convert image batch shape {tuple(img.shape)} to 6 channels")


def prepare_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert masks to [B, 1, H, W] in {0, 1}."""
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.float()

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask batch with 4 dims, got shape {tuple(mask.shape)}")

    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != (image_size, image_size):
        mask = F.interpolate(mask, size=(image_size, image_size), mode="nearest")
    return (mask > 0.5).float()


# ---------------------------------------------------------------------------
# 推理：特征分支 (cal)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cal_forward(
    model: DualBranchParallelFlowModel,
    condition: torch.Tensor,
) -> torch.Tensor:
    """
    仅运行特征分支，返回 cal [B, 1, H, W]（sigmoid 后的概率图）。

    cal 只依赖 condition（x[:, 3:9]），x_t 和 noise_channel 填零不影响结果。
    """
    B, _, H, W = condition.shape
    dummy_xt = torch.zeros(B, 3, H, W, device=condition.device, dtype=condition.dtype)
    dummy_noise = torch.zeros(B, 1, H, W, device=condition.device, dtype=condition.dtype)
    t_zero = torch.zeros(B, device=condition.device)
    model_input = torch.cat([dummy_xt, condition, dummy_noise], dim=1)  # [B, 10, H, W]
    _, cal = model(model_input, t_zero)
    return cal  # sigmoid 已在模型内部完成，值域 [0, 1]


# ---------------------------------------------------------------------------
# 推理：流匹配 ODE 采样
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_ode_sample(
    model: DualBranchParallelFlowModel,
    condition: torch.Tensor,
    num_steps: int,
    image_size: int,
) -> torch.Tensor:
    """
    Euler 积分从 t=0（纯噪声）到 t=1（预测掩码）。

    输入结构与训练时一致：
        [x_t(3ch), condition(6ch), noise_channel(1ch)] → 10 通道

    积分结束后对 3 通道取均值，得到 [B, 1, H, W] 的概率图，clamp 到 [0, 1]。
    """
    B = condition.shape[0]
    device = condition.device
    dtype = condition.dtype

    # noise_channel 在整个 ODE 过程中保持不变（推理时固定随机噪声）
    noise_channel = torch.randn(B, 1, image_size, image_size, device=device, dtype=dtype)

    # 初始状态：t=0 → 纯噪声
    x = torch.randn(B, 3, image_size, image_size, device=device, dtype=dtype)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
        model_input = torch.cat([x, condition, noise_channel], dim=1)  # [B, 10, H, W]
        v_pred, _ = model(model_input, t_tensor)                        # [B, 3, H, W]
        x = x + v_pred * dt  # Euler step

    # 3 通道取均值 → [B, 1, H, W]
    pred = x.mean(dim=1, keepdim=True)
    return torch.clamp(pred, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 整体评测
# ---------------------------------------------------------------------------

def _save_binary_pred(pred: torch.Tensor, path: Path, threshold: float) -> None:
    """将 [1, H, W] 概率图二值化后保存为灰度 PNG（0 / 255）。"""
    binary = (pred[0] >= threshold).cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(path)


@torch.no_grad()
def evaluate(
    model: DualBranchParallelFlowModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict, dict, dict, dict, dict]:
    model.eval()
    stats_cal = BinarySegmentationStats()
    stats_flow = BinarySegmentationStats()
    stats_best = BinarySegmentationStats()

    cal_dir = output_dir / "cal_preds"
    flow_dir = output_dir / "flow_preds"
    best_dir = output_dir / "best_preds"
    oracle_dir = output_dir / "oracle_global_preds"
    cal_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-7
    oracle_records: list[dict] = []
    total_time = 0.0
    total_samples = 0

    for img, mask, names in tqdm(loader, desc="evaluating", unit="batch"):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        condition = prepare_condition(img, args.image_size)
        target_mask = prepare_mask(mask, args.image_size)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        # --- 特征分支 (cal) ---
        cal = cal_forward(model, condition)
        # --- 流匹配 ODE ---
        flow_pred = euler_ode_sample(model, condition, args.sample_steps, args.image_size)

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - t0
        batch_size = target_mask.shape[0]
        total_time += batch_time
        total_samples += batch_size

        update_stats_from_logits(
            stats_cal, cal, target_mask, threshold=args.threshold, from_logits=False
        )

        # --- 流匹配 ODE ---
        update_stats_from_logits(
            stats_flow, flow_pred, target_mask, threshold=args.threshold, from_logits=False
        )

        # --- 逐样本 best-one 选择 ---
        cal_binary = (cal >= args.threshold).float()
        flow_binary = (flow_pred >= args.threshold).float()
        target_binary = target_mask  # 已是 {0, 1}

        for i in range(target_mask.shape[0]):
            cal_i = cal_binary[i:i + 1]
            flow_i = flow_binary[i:i + 1]
            target_i = target_binary[i:i + 1]

            inter_cal = float((cal_i * target_i).sum().item())
            inter_flow = float((flow_i * target_i).sum().item())
            pred_cal = float(cal_i.sum().item())
            pred_flow = float(flow_i.sum().item())
            tgt_sum = float(target_i.sum().item())

            iou_cal = (inter_cal + eps) / (pred_cal + tgt_sum - inter_cal + eps)
            iou_flow = (inter_flow + eps) / (pred_flow + tgt_sum - inter_flow + eps)

            oracle_records.append(
                {
                    "name": Path(names[i]).stem,
                    "tp_cal": inter_cal,
                    "tp_flow": inter_flow,
                    "pred_cal": pred_cal,
                    "pred_flow": pred_flow,
                    "target": tgt_sum,
                    "union_cal": pred_cal + tgt_sum - inter_cal,
                    "union_flow": pred_flow + tgt_sum - inter_flow,
                }
            )

            if iou_cal >= iou_flow:
                stats_best.update(cal_i, target_i)
            else:
                stats_best.update(flow_i, target_i)

        # --- 逐图保存预测结果 ---
        for i, img_path in enumerate(names):
            stem = Path(img_path).stem
            _save_binary_pred(cal[i], cal_dir / f"{stem}.png", args.threshold)
            _save_binary_pred(flow_pred[i], flow_dir / f"{stem}.png", args.threshold)
            # best_one 预测图
            cal_b = cal_binary[i]
            flow_b = flow_binary[i]
            target_b = target_binary[i:i + 1]
            inter_cal = float((cal_b * target_b).sum().item())
            inter_flow = float((flow_b * target_b).sum().item())
            pred_cal = float(cal_b.sum().item())
            pred_flow = float(flow_b.sum().item())
            tgt_sum = float(target_b.sum().item())
            iou_cal = (inter_cal + eps) / (pred_cal + tgt_sum - inter_cal + eps)
            iou_flow = (inter_flow + eps) / (pred_flow + tgt_sum - inter_flow + eps)
            best_img = cal[i] if iou_cal >= iou_flow else flow_pred[i]
            _save_binary_pred(best_img, best_dir / f"{stem}.png", args.threshold)

    oracle_selected, oracle_metrics = _solve_global_oracle(oracle_records)
    selection_manifest = []
    for record, use_flow in zip(oracle_records, oracle_selected):
        branch = "flow" if use_flow else "cal"
        source_dir = flow_dir if use_flow else cal_dir
        filename = f"{record['name']}.png"
        shutil.copyfile(source_dir / filename, oracle_dir / filename)
        selection_manifest.append({"image": record["name"], "branch": branch})

    with (output_dir / "oracle_global_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2, ensure_ascii=False)

    avg_ms = (total_time / total_samples * 1000) if total_samples > 0 else 0.0
    timing = {"total_time_s": round(total_time, 3), "num_samples": total_samples, "avg_ms_per_image": round(avg_ms, 3)}
    return stats_cal.compute(), stats_flow.compute(), stats_best.compute(), oracle_metrics, timing


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)

    model = DualBranchParallelFlowModel(
        in_channels=10,
        base_channels=args.base_channels,
        fusion_type=args.fusion_type,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    saved_epoch = int(ckpt.get("epoch", -1))

    print(f"已加载 checkpoint（epoch {saved_epoch}）: {args.checkpoint}")
    print(f"测试集: {Path(args.data_root) / args.test_split}")
    print(f"ODE 步数: {args.sample_steps}  |  阈值: {args.threshold}")
    print("-" * 60)

    metrics_cal, metrics_flow, metrics_best, metrics_oracle, timing = evaluate(
        model, loader, args, device, output_dir
    )

    result = {
        "checkpoint_epoch": saved_epoch,
        "checkpoint": str(args.checkpoint),
        "test_split": args.test_split,
        "sample_steps": args.sample_steps,
        "threshold": args.threshold,
        "seed": args.seed,
        "cal_branch": metrics_cal,
        "flow_branch": metrics_flow,
        "best_one": metrics_best,
        "oracle_global": metrics_oracle,
        "timing": timing,
    }

    out_path = output_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("特征分支 (cal):")
    print(f"  IoU  : {metrics_cal['IoU']:.6f}")
    print(f"  Dice : {metrics_cal['Dice']:.6f}")
    print("-" * 60)
    print(f"流匹配 ODE ({args.sample_steps} steps):")
    print(f"  IoU  : {metrics_flow['IoU']:.6f}")
    print(f"  Dice : {metrics_flow['Dice']:.6f}")
    print("-" * 60)
    print("best_one (逐样本选更高 IoU 的分支):")
    print(f"  IoU  : {metrics_best['IoU']:.6f}")
    print(f"  Dice : {metrics_best['Dice']:.6f}")
    print("-" * 60)
    print("oracle_global (GT-visible, globally optimal per-image branch selection):")
    print(f"  IoU  : {metrics_oracle['IoU']:.6f}")
    print(f"  Dice : {metrics_oracle['Dice']:.6f}")
    print(
        f"  CAL/FLOW selected: {metrics_oracle['cal_selected']}/"
        f"{metrics_oracle['flow_selected']}"
    )
    print("-" * 60)
    print(f"平均推理时间: {timing['avg_ms_per_image']:.3f} ms / 图  ({timing['total_time_s']:.3f}s / {timing['num_samples']} 张)")
    print("=" * 60)
    print(f"指标保存至  : {out_path}")
    print(f"cal 预测图  : {output_dir / 'cal_preds'}/")
    print(f"flow 预测图 : {output_dir / 'flow_preds'}/")
    print(f"best 预测图 : {output_dir / 'best_preds'}/")


if __name__ == "__main__":
    main()
