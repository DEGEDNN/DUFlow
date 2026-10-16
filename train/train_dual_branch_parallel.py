import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_matching.dual_branch_model_parallel import (  # noqa: E402
    DualBranchLoss,
    DualBranchParallelFlowModel,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the independent-encoder dual-branch flow matching model."
    )
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--train-split", type=str, default="Train")
    parser.add_argument("--test-split", type=str, default="Test")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=str, default="checkpoints/dual_branch_parallel")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--fusion-type", choices=["concat", "add", "attention"], default="concat")
    parser.add_argument("--weight-cal", type=float, default=10.0)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_loaders(args):
    try:
        from flow_matching.dataset_loader import CustomDataset
    except ModuleNotFoundError as exc:
        if exc.name == "tifffile":
            raise RuntimeError(
                "dataset_loader.py requires tifffile. Install it with: pip install tifffile"
            ) from exc
        raise

    transform = transforms.ToTensor()
    train_dataset = CustomDataset(
        args=None,
        data_path=os.path.join(args.data_root, args.train_split),
        transform=transform,
        mode="Training",
    )
    validation_dataset = CustomDataset(
        args=None,
        data_path=os.path.join(args.data_root, args.train_split),
        transform=transform,
        mode="Test",
    )
    test_dataset = CustomDataset(
        args=None,
        data_path=os.path.join(args.data_root, args.test_split),
        transform=transform,
        mode="Test",
    )

    if len(train_dataset) != len(validation_dataset):
        raise RuntimeError("Training and validation views must contain the same samples")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    validation_size = int(round(len(train_dataset) * args.val_ratio))
    training_size = len(train_dataset) - validation_size
    if training_size <= 0 or validation_size <= 0:
        raise RuntimeError("The requested train/validation split is empty")

    split_generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_dataset), generator=split_generator).tolist()
    training_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    common = {
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(
        Subset(train_dataset, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **common,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    split_manifest = {
        "seed": args.seed,
        "train_source": str(Path(args.data_root) / args.train_split),
        "test_source": str(Path(args.data_root) / args.test_split),
        "official_training_size": len(train_dataset),
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": len(test_dataset),
        "validation_ratio": args.val_ratio,
        "training_indices_in_official_train": training_indices,
        "validation_indices_in_official_train": validation_indices,
        "test_used_for_model_selection": False,
    }
    return train_loader, validation_loader, test_loader, split_manifest


def prepare_condition(img, image_size):
    """Convert dataset image tensors to [B, 6, H, W]."""
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    img = img.float()

    if img.ndim == 3:
        img = img.unsqueeze(1)

    if img.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got shape {tuple(img.shape)}")

    # Common path for this dataset after ToTensor(): [B, 1, H, 6].
    # Interpret the width dimension as six feature channels and broadcast spatially.
    if img.shape[1] == 1 and img.shape[-1] == 6:
        img = img[:, 0].permute(0, 2, 1).unsqueeze(-1)  # [B, 6, H, 1]
        return F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)

    # Already channel-first.
    if img.shape[1] == 6:
        if img.shape[-2:] != (image_size, image_size):
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img

    # Channel-last fallback: [B, H, W, 6].
    if img.shape[-1] == 6:
        img = img.permute(0, 3, 1, 2)
        if img.shape[-2:] != (image_size, image_size):
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img

    # Single grayscale image fallback. This is less informative, but keeps the
    # model input contract intact when experimenting with non-5d images.
    if img.shape[1] == 1:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return img.repeat(1, 6, 1, 1)

    raise ValueError(f"Cannot convert image batch shape {tuple(img.shape)} to 6 channels")


def prepare_mask(mask, image_size):
    """Convert masks to [B, 1, H, W] in [0, 1]."""
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


def build_model_input(condition, target_mask, sigma_min=0.0):
    """Build x=[x_t(3), condition(6), noise_channel(1)] and flow target u_t."""
    x0 = target_mask.repeat(1, 3, 1, 1)
    x1 = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    t_view = t.view(-1, 1, 1, 1)

    x_t = t_view * x0 + (1.0 - t_view) * x1
    if sigma_min > 0:
        x_t = x_t + sigma_min * torch.randn_like(x_t)
    u_t = x0 - x1

    noise_channel = torch.randn(
        x0.shape[0],
        1,
        x0.shape[2],
        x0.shape[3],
        device=x0.device,
        dtype=x0.dtype,
    )
    model_input = torch.cat([x_t, condition, noise_channel], dim=1)
    return model_input, t, u_t


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, args, device, epoch):
    model.train()
    running = {"loss": 0.0, "loss_diff": 0.0, "loss_cal": 0.0}
    progress = tqdm(loader, desc=f"epoch {epoch:03d} train", leave=False)

    for img, mask, _ in progress:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        condition = prepare_condition(img, args.image_size)
        target_mask = prepare_mask(mask, args.image_size)
        model_input, timesteps, u_t = build_model_input(condition, target_mask, args.sigma_min)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            v_pred, cal = model(model_input, timesteps)
            losses = loss_fn(v_pred, u_t, cal, target_mask)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = img.shape[0]
        for key in running:
            running[key] += float(losses[key].detach()) * batch_size
        progress.set_postfix(loss=float(loss.detach()))

    count = len(loader.dataset)
    return {key: value / max(count, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, args, device, epoch, split="val"):
    model.eval()
    running = {"loss": 0.0, "loss_diff": 0.0, "loss_cal": 0.0}
    intersection = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    progress = tqdm(loader, desc=f"epoch {epoch:03d} {split}", leave=False)

    for img, mask, _ in progress:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        condition = prepare_condition(img, args.image_size)
        target_mask = prepare_mask(mask, args.image_size)
        model_input, timesteps, u_t = build_model_input(condition, target_mask, args.sigma_min)
        v_pred, cal = model(model_input, timesteps)
        losses = loss_fn(v_pred, u_t, cal, target_mask)

        batch_size = img.shape[0]
        for key in running:
            running[key] += float(losses[key].detach()) * batch_size

        prediction = (cal >= args.threshold).float()
        intersection += float((prediction * target_mask).sum().item())
        prediction_sum += float(prediction.sum().item())
        target_sum += float(target_mask.sum().item())

    count = len(loader.dataset)
    metrics = {key: value / max(count, 1) for key, value in running.items()}
    eps = 1e-7
    metrics["dice"] = (2.0 * intersection + eps) / (prediction_sum + target_sum + eps)
    metrics["iou"] = (intersection + eps) / (
        prediction_sum + target_sum - intersection + eps
    )
    return metrics


def save_checkpoint(path, model, optimizer, scaler, epoch, best_validation_dice):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_dice": best_validation_dice,
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, split_manifest = build_loaders(args)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    model = DualBranchParallelFlowModel(
        in_channels=10,
        base_channels=args.base_channels,
        fusion_type=args.fusion_type,
    ).to(device)
    loss_fn = DualBranchLoss(weight_cal=args.weight_cal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_validation_dice = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_dice = float(
            checkpoint.get("best_validation_dice", best_validation_dice)
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, args, device, epoch
        )
        message = (
            f"epoch {epoch:03d} "
            f"train loss={train_metrics['loss']:.6f} "
            f"flow={train_metrics['loss_diff']:.6f} "
            f"cal={train_metrics['loss_cal']:.6f}"
        )

        val_metrics = evaluate(model, val_loader, loss_fn, args, device, epoch, split="val")
        message += (
            f" | val loss={val_metrics['loss']:.6f} "
            f"flow={val_metrics['loss_diff']:.6f} "
            f"cal={val_metrics['loss_cal']:.6f} "
            f"dice={val_metrics['dice']:.6f} "
            f"iou={val_metrics['iou']:.6f}"
        )

        print(message)

        if val_metrics["dice"] > best_validation_dice:
            best_validation_dice = val_metrics["dice"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_validation_dice,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_validation_dice,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_validation_dice,
            )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    selected_epoch = int(best_checkpoint["epoch"])
    validation_metrics = evaluate(
        model, val_loader, loss_fn, args, device, selected_epoch, split="val"
    )
    test_metrics = evaluate(
        model, test_loader, loss_fn, args, device, selected_epoch, split="test"
    )
    results = {
        "selected_epoch": selected_epoch,
        "best_validation_dice": float(best_checkpoint["best_validation_dice"]),
        "validation": validation_metrics,
        "test": test_metrics,
        "test_used_for_model_selection": False,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("final " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
