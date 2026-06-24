"""
把任意裂缝图像预处理成 DUFlow 需要的 6 通道 condition。
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def rgb_to_sixch(img: torch.Tensor) -> torch.Tensor:
    """
    将 RGB 图像 [B, 3, H, W] 转换为 6 通道 [B, 6, H, W]。
    策略：加权灰度 + 重复 6 次。
    """
    if img.shape[1] != 3:
        raise ValueError(f"Expected RGB input with 3 channels, got {img.shape[1]}")
    # ITU-R BT.601 标准灰度转换
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return gray.repeat(1, 6, 1, 1)


def prepare_condition(img: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """
    将 dataset 返回的图像张量转换为 [B, 6, H, W] 的 condition。

    支持的输入：
        - [B, 1, H, W] 单通道：重复 6 次
        - [B, 3, H, W] RGB：先转灰度再重复 6 次
        - [B, 6, H, W] 已经 6 通道：直接 resize
        - [B, H, W, 6]：permute 后 resize
    """
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    img = img.float()

    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got shape {tuple(img.shape)}")

    # 处理 HWC 格式
    if img.shape[-1] == 6 and img.shape[1] != 6:
        img = img.permute(0, 3, 1, 2)

    if img.shape[1] == 1:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img.repeat(1, 6, 1, 1)

    if img.shape[1] == 3:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return rgb_to_sixch(img)

    if img.shape[1] == 6:
        if img.shape[-2:] != (image_size, image_size):
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img

    raise ValueError(f"Cannot convert image batch shape {tuple(img.shape)} to 6 channels")


def prepare_mask(mask: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """将 mask 转换为 [B, 1, H, W] 的 {0, 1} 张量。"""
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.float()

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask batch with 4 dims, got shape {tuple(img.shape)}")

    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != (image_size, image_size):
        mask = F.interpolate(mask, size=(image_size, image_size), mode="nearest")
    return (mask > 0.5).float()


def image_net_normalize(img: torch.Tensor) -> torch.Tensor:
    """对 3 通道图像做 ImageNet 归一化；对 6 通道图像按通道重复 mean/std 应用。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    if img.shape[1] == 3:
        return (img - mean) / std
    if img.shape[1] == 6:
        mean6 = mean.repeat(1, 2, 1, 1)
        std6 = std.repeat(1, 2, 1, 1)
        return (img - mean6) / std6
    return img
