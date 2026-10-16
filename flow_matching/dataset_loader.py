import os
from glob import glob
from pathlib import Path

import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode="Training"):
        print("loading data from the directory:", data_path)
        path = data_path
        images = sorted(glob(os.path.join(path, "5d/*.png")))
        masks = sorted(glob(os.path.join(path, "mask/*.bmp")))

        if not images:
            raise RuntimeError(f"No FIND images found under {Path(path) / '5d'}")
        if len(images) != len(masks):
            raise RuntimeError(
                f"Image/mask count mismatch under {path}: {len(images)} images, {len(masks)} masks"
            )

        image_stems = [Path(item).stem for item in images]
        mask_stems = [Path(item).stem for item in masks]
        mismatches = [
            (image_stem, mask_stem)
            for image_stem, mask_stem in zip(image_stems, mask_stems)
            if image_stem != mask_stem
        ]
        if mismatches:
            preview = ", ".join(f"{image}!={mask}" for image, mask in mismatches[:5])
            raise RuntimeError(f"Image/mask filename pairing mismatch: {preview}")

        self.name_list = images
        self.label_list = masks
        self.data_path = path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)
        img = tiff.imread(img_path)
        # img = Image.open(img_path)
        # img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'Training':
            return (img, mask, name)
        else:
            return (img, mask, name)