"""
CrackFlow 训练与评测统一指标模块

职责：
1. 计算二值分割任务的 IoU / Dice
2. 支持单个 batch 统计
3. 支持整个数据集累计统计
4. 对非法指标关系进行一致性校验
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BinarySegmentationStats:
    """保存二值分割累计统计量。"""

    intersection: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """使用一个 batch 的二值预测和标签更新累计统计量。"""
        pred = pred.float()
        target = target.float()

        self.intersection += float((pred * target).sum().item())
        self.pred_sum += float(pred.sum().item())
        self.target_sum += float(target.sum().item())

    def compute(self, eps: float = 1e-7) -> Dict[str, float]:
        """根据累计统计量计算 IoU 和 Dice。"""
        union = self.pred_sum + self.target_sum - self.intersection
        iou = (self.intersection + eps) / (union + eps)
        dice = (2.0 * self.intersection + eps) / (self.pred_sum + self.target_sum + eps)

        validate_iou_dice_relation(iou, dice)

        return {
            "IoU": iou,
            "Dice": dice,
        }


def threshold_predictions(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """将概率图按阈值转为二值预测。"""
    return (prob >= threshold).float()



def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """将标签掩码标准化为 {0,1} 二值张量。"""
    return (mask >= 0.5).float()



def validate_binary_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    """检查预测与标签形状是否一致。"""
    if pred.shape != target.shape:
        raise ValueError(f"预测与标签形状不一致: pred={pred.shape}, target={target.shape}")

    if pred.ndim < 2:
        raise ValueError(f"输入维度异常，至少需要 batch 维: pred.ndim={pred.ndim}")



def validate_iou_dice_relation(iou: float, dice: float, tol: float = 1e-6) -> None:
    """校验二值分割中 Dice 不应小于 IoU。"""
    if dice + tol < iou:
        raise ValueError(f"非法指标关系: Dice={dice:.6f} < IoU={iou:.6f}")



def compute_batch_iou_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
    from_logits: bool = True,
) -> Dict[str, float]:
    """
    计算单个 batch 的平均 IoU 和平均 Dice。

    说明：
    1. 当 from_logits=True 时，输入会先经过 sigmoid。
    2. 返回的是 batch 内逐样本计算后再取平均的结果。
    3. 更适合训练过程中的快速监控；论文表格建议使用累计统计版本。
    """
    if from_logits:
        prob = torch.sigmoid(logits)
    else:
        prob = logits

    pred = threshold_predictions(prob, threshold=threshold)
    target = normalize_mask(target)

    validate_binary_shapes(pred, target)

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    if torch.any(dice + 1e-6 < iou):
        min_gap = float((dice - iou).min().item())
        raise ValueError(f"检测到非法指标关系: Dice < IoU, min(Dice-IoU)={min_gap:.6f}")

    return {
        "IoU": float(iou.mean().item()),
        "Dice": float(dice.mean().item()),
    }


@torch.no_grad()
def update_stats_from_logits(
    stats: BinarySegmentationStats,
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> None:
    """使用 logits 或概率图更新累计统计量。"""
    if from_logits:
        prob = torch.sigmoid(logits)
    else:
        prob = logits

    pred = threshold_predictions(prob, threshold=threshold)
    target = normalize_mask(target)

    validate_binary_shapes(pred, target)
    stats.update(pred, target)


@torch.no_grad()
def evaluate_dataset_from_dataloader(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    image_key: str = "image",
    mask_key: str = "mask",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    在整个数据集上累计 TP 类统计量，再统一计算 IoU / Dice。

    注意：
    1. 默认 dataloader 的 batch 是 dict，并包含 image/mask 两个键。
    2. 默认模型输出为 [B,1,H,W] 的 logits。
    3. 若你的工程 batch 结构不同，应在训练/评测脚本中适配后再调用。
    """
    model.eval()
    stats = BinarySegmentationStats()

    for batch in dataloader:
        image = batch[image_key].to(device)
        target = batch[mask_key].to(device).float()

        logits = model(image)
        update_stats_from_logits(stats, logits, target, threshold=threshold, from_logits=True)

    return stats.compute()
