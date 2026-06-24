"""
通用裂缝数据集加载器，支持 CSF-50K、CrackTree260、CRACK500、CrackLS315、CFD 等。
输入图像会被统一处理为 6 通道 condition，mask 会被二值化为 {0, 1}。
"""
import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class CrackDataset(Dataset):
    """
    通用裂缝分割数据集。

    目录结构支持两种形式：
        1) images/ + masks/ 平铺：
           root/
             images/00001.png
             masks/00001.png

        2) 子目录形式（CrackTree260 / CRACK500 等）：
           root/
             train/
               images/
               masks/
             test/
               images/
               masks/

    Args:
        root_dir: 数据集根目录
        split: 'train' / 'test' / 'val' / None（None 表示直接读取 root_dir 下的 images/masks）
        image_size: 统一 resize 尺寸
        img_ext: 图像后缀，默认自动识别 .png/.jpg/.jpeg/.bmp/.tif
        mask_ext: mask 后缀，默认与图像同名
        transform: 可选的额外变换
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = "test",
        image_size: Tuple[int, int] = (256, 256),
        img_ext: Optional[str] = None,
        mask_ext: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # 自动定位 images / masks 目录
        self.img_dir, self.mask_dir = self._locate_dirs(root_dir, split)

        self.img_paths = sorted(self._glob_images(self.img_dir, img_ext))
        self.pairs = []
        for img_path in self.img_paths:
            mask_path = self._find_mask(img_path, self.mask_dir, mask_ext)
            if mask_path and os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))

        print(f"[CrackDataset] {root_dir} ({split}) -> {len(self.pairs)} valid pairs")

    def _locate_dirs(self, root_dir: str, split: Optional[str]) -> Tuple[str, str]:
        candidates = []
        if split is not None:
            candidates.append(os.path.join(root_dir, split))
        candidates.append(root_dir)

        for base in candidates:
            for img_name in ["images", "image", "imgs", "img"]:
                img_dir = os.path.join(base, img_name)
                if os.path.isdir(img_dir):
                    for mask_name in ["masks", "mask", "cracks", "crack", "lbs", "labels"]:
                        mask_dir = os.path.join(base, mask_name)
                        if os.path.isdir(mask_dir):
                            return img_dir, mask_dir
        # 兜底：root_dir/images 和 root_dir/masks
        return os.path.join(root_dir, "images"), os.path.join(root_dir, "masks")

    def _glob_images(self, img_dir: str, ext: Optional[str]) -> list:
        if ext is not None:
            return glob.glob(os.path.join(img_dir, f"*{ext}"))
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(img_dir, p)))
        return list(set(files))

    def _find_mask(self, img_path: str, mask_dir: str, mask_ext: Optional[str]) -> Optional[str]:
        base = os.path.splitext(os.path.basename(img_path))[0]
        if mask_ext:
            return os.path.join(mask_dir, base + mask_ext)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            path = os.path.join(mask_dir, base + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.pairs[idx]

        # 读取图像，保持原始通道
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        if img.ndim == 2:
            img = img[..., None]
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32)

        # 转成 tensor
        if img.shape[-1] == 1:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [1, H, W]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor, os.path.basename(img_path)


class CSF50KDataset(CrackDataset):
    """CSF-50K 专用加载器，结构与 CrackDataset 一致。"""

    def __init__(self, root_dir: str, split: str = "testing", image_size: Tuple[int, int] = (256, 256)):
        super().__init__(root_dir, split=split, image_size=image_size)
