"""Dataset loader for Oxford-IIIT Pet Dataset.

Supports all three tasks simultaneously:
    - Classification  : 37 breed classes (0-indexed)
    - Localization    : bounding box [x_center, y_center, width, height] in pixel space
    - Segmentation    : trimap mask with values in {0, 1, 2}
                        (0=foreground/pet, 1=background, 2=border/uncertain)
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


# VGG11 paper uses 224×224 inputs
IMG_SIZE = 224

# ImageNet statistics — used since VGG11 was designed for ImageNet-normalised inputs
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    """Training augmentation: geometric + colour jitter + normalisation.

    All geometric ops (flip, shift, rotate) are applied identically to the
    image, the bounding box, and the segmentation mask to keep them aligned.
    """
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,          # constant padding (black)
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),               # HWC → CHW, numpy → torch
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",        # expects [xmin, ymin, xmax, ymax]
            label_fields=["bbox_labels"],
            clip=True,                  # clip boxes to image boundary after transform
            min_visibility=0.3,         # drop box if <30% visible after crop/rotate
        ),
    )


def get_val_transforms() -> A.Compose:
    """Validation / test augmentation: deterministic resize + normalisation only."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_labels"],
            clip=True,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset.

    Args:
        root:         Root directory containing ``images/`` and ``annotations/``
                      sub-directories (standard Oxford-IIIT layout).
        split:        One of ``'train'``, ``'val'``, or ``'test'``.
        val_fraction: Fraction of the official *trainval* set reserved for
                      validation.  Ignored when ``split='test'``.
        seed:         RNG seed for the reproducible trainval → train/val split.
        transform:    Custom ``albumentations.Compose`` pipeline.  When *None*
                      a default pipeline is selected based on ``split``.

    Returns (per ``__getitem__``):
        A ``dict`` with keys:

        * ``"image"``  – ``FloatTensor [3, 224, 224]``, ImageNet-normalised.
        * ``"label"``  – ``int``, breed class index in ``[0, 36]``.
        * ``"bbox"``   – ``FloatTensor [4]``, ``[x_center, y_center, w, h]``
                         in **pixel coordinates** of the 224×224 image.
        * ``"mask"``   – ``LongTensor [224, 224]``, trimap values in
                         ``{0=foreground, 1=background, 2=border}``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        transform: Optional[A.Compose] = None,
    ) -> None:
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test'; got '{split}'"
        )

        self.root     = root
        self.split    = split
        self.img_dir  = os.path.join(root, "images")
        self.ann_dir  = os.path.join(root, "annotations")
        self.xml_dir  = os.path.join(self.ann_dir, "xmls")
        self.mask_dir = os.path.join(self.ann_dir, "trimaps")

        # Map image_name → 0-based class index
        label_map = self._parse_list_txt()

        # Determine which image names belong to this split
        if split == "test":
            names = self._read_split_file("test.txt")
        else:
            all_tv = self._read_split_file("trainval.txt")
            rng    = np.random.default_rng(seed)
            perm   = rng.permutation(len(all_tv))
            cut    = int(len(all_tv) * val_fraction)
            val_set = set(perm[:cut].tolist())
            if split == "val":
                names = [all_tv[i] for i in range(len(all_tv)) if i in val_set]
            else:
                names = [all_tv[i] for i in range(len(all_tv)) if i not in val_set]

        # Build sample list, keeping only entries with valid annotation files
        self.samples: List[Dict] = []
        for name in names:
            img_path  = self._find_image(name)
            xml_path  = os.path.join(self.xml_dir,  name + ".xml")
            mask_path = os.path.join(self.mask_dir, name + ".png")

            if img_path is None:
                continue
            if name not in label_map:
                continue
            if not os.path.exists(xml_path) or not os.path.exists(mask_path):
                continue

            self.samples.append(
                {
                    "name":      name,
                    "img_path":  img_path,
                    "xml_path":  xml_path,
                    "mask_path": mask_path,
                    "label":     label_map[name],
                }
            )

        # Select / store augmentation pipeline
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_image(self, name: str) -> Optional[str]:
        """Return the image path for *name*, trying common extensions."""
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.img_dir, name + ext)
            if os.path.exists(p):
                return p
        return None

    def _parse_list_txt(self) -> Dict[str, int]:
        """Parse ``annotations/list.txt`` → ``{image_name: class_index}`` (0-based)."""
        path    = os.path.join(self.ann_dir, "list.txt")
        mapping: Dict[str, int] = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts    = line.split()
                name     = parts[0]
                class_id = int(parts[1]) - 1   # list.txt is 1-indexed
                mapping[name] = class_id
        return mapping

    def _read_split_file(self, filename: str) -> List[str]:
        """Return image names (no extension) listed in an annotations split file."""
        path  = os.path.join(self.ann_dir, filename)
        names: List[str] = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.append(line.split()[0])
        return names

    def _parse_bbox_xml(self, xml_path: str) -> Tuple[float, float, float, float]:
        """Parse a Pascal-VOC XML file and return (xmin, ymin, xmax, ymax)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find("object")
        bb   = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        return xmin, ymin, xmax, ymax

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # ── Image ──────────────────────────────────────────────────────
        image   = np.array(Image.open(sample["img_path"]).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # ── Trimap mask ────────────────────────────────────────────────
        # Oxford trimap pixel values: 1=foreground, 2=background, 3=border
        # Remap to 0-based: 0=foreground, 1=background, 2=border
        mask = np.array(Image.open(sample["mask_path"]).convert("L"), dtype=np.uint8)
        mask = np.clip(mask.astype(np.int32) - 1, 0, 2).astype(np.uint8)

        # ── Bounding box (Pascal VOC format) ──────────────────────────
        xmin, ymin, xmax, ymax = self._parse_bbox_xml(sample["xml_path"])
        # Clip to valid image extent before transforming
        xmin = float(np.clip(xmin, 0, orig_w - 1))
        ymin = float(np.clip(ymin, 0, orig_h - 1))
        xmax = float(np.clip(xmax, xmin + 1, orig_w))
        ymax = float(np.clip(ymax, ymin + 1, orig_h))

        # ── Apply albumentations pipeline ─────────────────────────────
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[[xmin, ymin, xmax, ymax]],
            bbox_labels=[0],                # dummy category; we only have one box
        )

        image_t = transformed["image"]          # FloatTensor [3, 224, 224]
        mask_t  = transformed["mask"].long()    # LongTensor  [224, 224]

        # Recover transformed bbox (albumentations may drop it if invisible)
        bboxes_out = transformed["bboxes"]
        if len(bboxes_out) > 0:
            tx1, ty1, tx2, ty2 = bboxes_out[0]
        else:
            # Fallback: full image extent
            tx1, ty1, tx2, ty2 = 0.0, 0.0, float(IMG_SIZE), float(IMG_SIZE)

        # Convert [xmin, ymin, xmax, ymax] → [x_center, y_center, width, height]
        # All values are in pixel coordinates of the resized 224×224 image
        x_center = (tx1 + tx2) / 2.0
        y_center  = (ty1 + ty2) / 2.0
        width     =  tx2 - tx1
        height    =  ty2 - ty1
        bbox_t = torch.tensor(
            [x_center, y_center, width, height], dtype=torch.float32
        )

        return {
            "image": image_t,           # FloatTensor [3, 224, 224]
            "label": sample["label"],   # int in [0, 36]
            "bbox":  bbox_t,            # FloatTensor [4]
            "mask":  mask_t,            # LongTensor  [224, 224]
        }
