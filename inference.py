"""Inference and evaluation for the unified multi-task pipeline.

Loads MultiTaskPerceptionModel and runs it on the test split of the
Oxford-IIIT Pet dataset (or arbitrary images), reporting:
    - Classification Macro F1
    - Mean IoU  (detection)
    - Mean Dice (segmentation)

Usage:
    python inference.py --data_root /path/to/pets
    python inference.py --image /path/to/cat.jpg   # single-image mode
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.multitask import MultiTaskPerceptionModel


# ── Metric helpers ────────────────────────────────────────────────────────────

def _iou_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def xyxy(b):
        return b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, \
               b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    px1, py1, px2, py2 = xyxy(pred)
    tx1, ty1, tx2, ty2 = xyxy(target)
    ix = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    iy = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    inter = ix * iy
    union = ((px2 - px1) * (py2 - py1)).clamp(0) + \
            ((tx2 - tx1) * (ty2 - ty1)).clamp(0) - inter
    return inter / (union + eps)


def _dice_batch(pred_logits: torch.Tensor, target: torch.Tensor,
                num_classes: int = 3, smooth: float = 1.0) -> float:
    pred = pred_logits.argmax(1)
    score = 0.0
    for c in range(num_classes):
        p = (pred == c).float(); t = (target == c).float()
        score += (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return (score / num_classes).item()


# ── Evaluation on dataset split ───────────────────────────────────────────────

def evaluate(
    model: MultiTaskPerceptionModel,
    data_root: str,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 4,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run the unified pipeline on a dataset split and return metric dict."""

    ds = OxfordIIITPetDataset(data_root, split=split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    all_cls_preds, all_cls_labels = [], []
    iou_sum, dice_sum, n = 0.0, 0.0, 0

    model.eval()
    with torch.no_grad():
        for batch in dl:
            imgs   = batch["image"].to(device)
            labels = batch["label"]
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels = labels.long()
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            out = model(imgs)

            # Classification
            preds = out["classification"].argmax(1).cpu()
            all_cls_preds.extend(preds.tolist())
            all_cls_labels.extend(labels.tolist())

            # Localization
            iou_sum += _iou_batch(out["localization"], bboxes).sum().item()

            # Segmentation
            dice_sum += _dice_batch(out["segmentation"], masks) * imgs.size(0)

            n += imgs.size(0)

    macro_f1 = f1_score(all_cls_labels, all_cls_preds, average="macro", zero_division=0)
    mean_iou  = iou_sum  / n
    mean_dice = dice_sum / n

    return {"macro_f1": macro_f1, "mean_iou": mean_iou, "mean_dice": mean_dice}


# ── Single-image inference ────────────────────────────────────────────────────

_INFER_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

BREED_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
    "British_Shorthair", "chihuahua", "Egyptian_Mau", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher",
    "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
    "Siamese", "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier",
    "yorkshire_terrier",
]


def infer_single(
    model: MultiTaskPerceptionModel,
    image_path: str,
    device: torch.device,
) -> dict:
    """Run the pipeline on a single image file.

    Args:
        model:      Loaded MultiTaskPerceptionModel in eval mode.
        image_path: Path to an RGB image file.
        device:     Compute device.

    Returns:
        dict with keys:
            ``breed``      — predicted breed name (str).
            ``confidence`` — softmax confidence for the top class (float).
            ``bbox``       — [x_center, y_center, width, height] in pixels.
            ``mask``       — [H, W] predicted class label map (numpy).
    """
    img_np = np.array(Image.open(image_path).convert("RGB"))
    t = _INFER_TRANSFORM(image=img_np)["image"]      # [3, 224, 224]
    inp = t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(inp)

    probs = F.softmax(out["classification"][0], dim=0)
    cls_idx = probs.argmax().item()
    breed = BREED_NAMES[cls_idx] if cls_idx < len(BREED_NAMES) else str(cls_idx)
    confidence = probs[cls_idx].item()

    bbox = out["localization"][0].cpu().tolist()                 # [4]
    mask = out["segmentation"][0].argmax(0).cpu().numpy()       # [H, W]

    return {
        "breed":      breed,
        "confidence": confidence,
        "bbox":       bbox,
        "mask":       mask,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-task inference / evaluation")
    p.add_argument("--data_root", type=str, default=None,
                   help="Dataset root for split-level evaluation.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--image", type=str, default=None,
                   help="Path to a single image for inference.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--classifier_path",  type=str, default="checkpoints/classifier.pth")
    p.add_argument("--localizer_path",   type=str, default="checkpoints/localizer.pth")
    p.add_argument("--unet_path",        type=str, default="checkpoints/unet.pth")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MultiTaskPerceptionModel …")
    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    if args.image:
        result = infer_single(model, args.image, device)
        print(f"\n── Inference: {args.image} ──────────────────")
        print(f"  Breed      : {result['breed']}")
        print(f"  Confidence : {result['confidence']:.4f}")
        print(f"  BBox (cx,cy,w,h): {[f'{v:.1f}' for v in result['bbox']]}")
        print(f"  Mask shape : {result['mask'].shape}")

    if args.data_root:
        print(f"\nEvaluating on split='{args.split}' …")
        metrics = evaluate(
            model, args.data_root,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        print(f"\n── Evaluation Results ({'%s split' % args.split}) ─────────")
        print(f"  Macro F1  (classification) : {metrics['macro_f1']:.4f}")
        print(f"  Mean IoU  (localization)   : {metrics['mean_iou']:.4f}")
        print(f"  Mean Dice (segmentation)   : {metrics['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
