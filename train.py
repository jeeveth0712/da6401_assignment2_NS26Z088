"""Training entrypoint for DA6401 Assignment-2.

Supports training all three task-specific models:
    classify   → VGG11Classifier  (saved as classifier.pth)
    localize   → VGG11Localizer   (saved as localizer.pth)
    segment    → VGG11UNet        (saved as unet.pth)

Usage examples:
    # Task 1 — classification
    python train.py --task classify --data_root /path/to/pets

    # Task 2 — localization (fine-tune encoder from Task 1)
    python train.py --task localize --data_root /path/to/pets \\
                    --pretrained checkpoints/classifier.pth

    # Task 3 — segmentation, partial fine-tune (freeze first 3 blocks)
    python train.py --task segment --data_root /path/to/pets \\
                    --pretrained checkpoints/classifier.pth \\
                    --freeze_mode partial
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dice loss (segmentation) ─────────────────────────────────────────────────

def dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft multi-class Dice loss.

    Args:
        pred_logits: [B, C, H, W] raw logits.
        target:      [B, H, W] integer class labels.
        num_classes: Number of classes C.
        smooth:      Laplace smoothing constant.

    Returns:
        Scalar Dice loss in [0, 1].
    """
    pred  = F.softmax(pred_logits, dim=1)                           # [B, C, H, W]
    tgt   = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    inter = (pred * tgt).sum(dim=(2, 3))                            # [B, C]
    denom = pred.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))             # [B, C]
    dice  = (2.0 * inter + smooth) / (denom + smooth)              # [B, C]
    return 1.0 - dice.mean()


# ── Metric helpers ───────────────────────────────────────────────────────────

def compute_iou_batch(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-sample IoU between predicted and target boxes.

    Args:
        pred_boxes:   [B, 4] in (cx, cy, w, h).
        target_boxes: [B, 4] in (cx, cy, w, h).

    Returns:
        [B] IoU values in [0, 1].
    """
    def to_xyxy(b):
        return (b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2)

    px1, py1, px2, py2 = to_xyxy(pred_boxes)
    tx1, ty1, tx2, ty2 = to_xyxy(target_boxes)

    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    return inter / (p_area + t_area - inter + eps)


def compute_dice(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> float:
    """Compute mean Dice score over a batch (no gradient)."""
    with torch.no_grad():
        pred  = pred_logits.argmax(dim=1)                           # [B, H, W]
        score = 0.0
        for c in range(num_classes):
            p = (pred == c).float()
            t = (target == c).float()
            inter = (p * t).sum()
            score += (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return (score / num_classes).item()


# ── Encoder freeze helpers ───────────────────────────────────────────────────

def apply_freeze(encoder: nn.Module, mode: str) -> None:
    """Freeze encoder parameters according to the chosen strategy.

    Args:
        encoder: The VGG11Encoder module.
        mode:
            ``'full_freeze'`` — freeze entire encoder.
            ``'partial'``     — freeze blocks 1-3; unfreeze blocks 4-5.
            ``'none'``        — keep all params trainable.
    """
    if mode == "none":
        return

    if mode == "full_freeze":
        for p in encoder.parameters():
            p.requires_grad = False
        return

    if mode == "partial":
        # Freeze early blocks (generic low-level features)
        for name, p in encoder.named_parameters():
            if name.startswith(("block1", "pool1", "block2", "pool2", "block3", "pool3")):
                p.requires_grad = False
            else:
                p.requires_grad = True
        return

    raise ValueError(f"Unknown freeze_mode '{mode}'. Use 'full_freeze', 'partial', or 'none'.")


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, epoch: int, best_metric: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_metric},
        path,
    )


def load_encoder_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Load only the encoder weights from a checkpoint into ``model.encoder``."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    print(f"  Loaded encoder weights from {checkpoint_path}")


# ── Task trainers ────────────────────────────────────────────────────────────

def train_classifier(args: argparse.Namespace, device: torch.device) -> None:
    """Train VGG11 for 37-class breed classification (Task 1)."""

    train_transform = get_val_transforms() if args.no_augment else None
    train_ds = OxfordIIITPetDataset(args.data_root, split="train", seed=args.seed,
                                    transform=train_transform)
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "classifier.pth")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            labels = torch.tensor(batch["label"]).long().to(device) if not isinstance(batch["label"], torch.Tensor) else batch["label"].long().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
        train_loss /= n
        train_acc   = train_correct / n

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, all_preds, all_labels = 0.0, 0, [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                labels = batch["label"]
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                labels = labels.long().to(device)
                logits = model(imgs)
                val_loss    += criterion(logits, labels).item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()

        # ── Log & checkpoint ──────────────────────────────────────────────
        wandb.log({
            "epoch":          epoch,
            "train/cls_loss": train_loss,
            "train/acc":      train_acc,
            "val/cls_loss":   val_loss,
            "val/acc":        val_acc,
            "val/macro_f1":   val_f1,
        })
        print(f"[Classify] Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, ckpt_path, epoch, best_f1)
            print(f"  ✓ Saved best classifier (F1={best_f1:.4f}) → {ckpt_path}")


def train_localizer(args: argparse.Namespace, device: torch.device) -> None:
    """Train VGG11 bounding-box regressor (Task 2).

    Loss = MSELoss(pred, target) + IoULoss(pred, target).
    """

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", seed=args.seed)
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # Transfer encoder weights from classifier if available
    if args.pretrained:
        load_encoder_weights(model, args.pretrained)
    apply_freeze(model.encoder, args.freeze_mode)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mse_crit   = nn.MSELoss()
    iou_crit   = IoULoss(reduction="mean")

    best_iou  = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "localizer.pth")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, n = 0.0, 0
        for batch in train_dl:
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)               # [B, 4] in [0, 224]
            bboxes_norm = bboxes / 224.0                    # [B, 4] in [0, 1]
            optimizer.zero_grad()
            pred      = model(imgs) / 224.0                 # [B, 4] in [0, 1]
            loss = mse_crit(pred, bboxes_norm) + iou_crit(pred * 224.0, bboxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss /= n

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, iou_sum, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                bboxes_norm = bboxes / 224.0
                pred       = model(imgs) / 224.0
                loss  = mse_crit(pred, bboxes_norm) + iou_crit(pred * 224.0, bboxes)
                val_loss += loss.item() * imgs.size(0)
                iou_sum  += compute_iou_batch(pred * 224.0, bboxes).sum().item()
                nv += imgs.size(0)
        val_loss /= nv
        val_iou   = iou_sum / nv

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loc_loss": train_loss,
            "val/loc_loss":   val_loss,
            "val/mean_iou":   val_iou,
        })
        print(f"[Localize] Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, ckpt_path, epoch, best_iou)
            print(f"  ✓ Saved best localizer (IoU={best_iou:.4f}) → {ckpt_path}")


def train_segmentor(args: argparse.Namespace, device: torch.device) -> None:
    """Train VGG11-UNet for trimap segmentation (Task 3).

    Loss = CrossEntropyLoss + soft Dice loss.
    Justification: CE handles per-pixel class probability estimation; Dice
    explicitly optimises the overlap metric used for evaluation, and is more
    robust to foreground/background imbalance in the trimap.
    """

    train_ds = OxfordIIITPetDataset(args.data_root, split="train", seed=args.seed)
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    if args.pretrained:
        load_encoder_weights(model, args.pretrained)
    apply_freeze(model.encoder, args.freeze_mode)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    ce_crit    = nn.CrossEntropyLoss()

    best_dice = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "unet.pth")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, n = 0.0, 0
        for batch in train_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)               # [B, H, W]
            optimizer.zero_grad()
            logits = model(imgs)                           # [B, 3, H, W]
            loss   = ce_crit(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss /= n

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, dice_sum, px_correct, px_total, nv = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss   = ce_crit(logits, masks) + dice_loss(logits, masks)
                val_loss  += loss.item() * imgs.size(0)
                dice_sum  += compute_dice(logits, masks) * imgs.size(0)
                preds      = logits.argmax(1)
                px_correct += (preds == masks).sum().item()
                px_total   += masks.numel()
                nv += imgs.size(0)
        val_loss /= nv
        val_dice  = dice_sum / nv
        val_px_acc = px_correct / px_total

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/seg_loss":  train_loss,
            "val/seg_loss":    val_loss,
            "val/dice":        val_dice,
            "val/pixel_acc":   val_px_acc,
        })
        print(f"[Segment]  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"Dice={val_dice:.4f}  PixAcc={val_px_acc:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, ckpt_path, epoch, best_dice)
            print(f"  ✓ Saved best UNet (Dice={best_dice:.4f}) → {ckpt_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 training script")

    p.add_argument("--task", choices=["classify", "localize", "segment"],
                   required=True, help="Which task model to train.")
    p.add_argument("--data_root", type=str, required=True,
                   help="Root directory of the Oxford-IIIT Pet dataset.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout_p", type=float, default=0.5,
                   help="Dropout probability for the task head.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory in which to save .pth files.")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to a checkpoint from which encoder weights are loaded.")
    p.add_argument("--freeze_mode", choices=["none", "partial", "full_freeze"],
                   default="none",
                   help="Encoder freezing strategy for Tasks 2 & 3.")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable training augmentation (use resize+normalize only).")
    p.add_argument("--weight_decay", type=float, default=1e-3,
                   help="Weight decay for Adam optimizer.")
    p.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.task}_dropout{args.dropout_p}_{args.freeze_mode}",
        config=vars(args),
    )

    if args.task == "classify":
        train_classifier(args, device)
    elif args.task == "localize":
        train_localizer(args, device)
    elif args.task == "segment":
        train_segmentor(args, device)

    wandb.finish()


if __name__ == "__main__":
    main()
