"""Dataset loader for Oxford-IIIT Pet Dataset.

Handles all three tasks at once:
    - Classification  : 37 breed classes (0-indexed)
    - Localization    : bounding box [x_center, y_center, width, height] in pixel space
    - Segmentation    : trimap mask — 0=foreground/pet, 1=background, 2=border/uncertain
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


# VGG11 takes 224x224 inputs
IMG_SIZE = 224

# ImageNet mean and std — needed since our VGG11 was pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------


def get_train_transforms() -> A.Compose:
    """Training augmentations — flips, affine, colour jitter, blur, dropout, normalize.

    All geometric transforms are applied to image, bbox, and mask together
    so they stay aligned.
    """
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),  # converts HWC numpy to CHW torch tensor
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",      # [xmin, ymin, xmax, ymax] format
            label_fields=["bbox_labels"],
            clip=True,                # clip boxes to image boundary after transforms
            min_visibility=0.3,       # drop box if less than 30% visible after crop/rotate
        ),
    )


def get_val_transforms() -> A.Compose:
    """Val/test transforms — just resize and normalize, no augmentation."""
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
    """Oxford-IIIT Pet dataset that returns image, label, bbox and mask for each sample.

    Args:
        root:         Root folder containing images/ and annotations/ subdirectories.
        split:        'train', 'val', or 'test'.
        val_fraction: What fraction of trainval to use as validation. Ignored for test.
        seed:         Random seed for train/val split reproducibility.
        transform:    Custom albumentations pipeline. If None, we pick default based on split.

    Each __getitem__ returns a dict:
        "image"  — FloatTensor [3, 224, 224], ImageNet normalized
        "label"  — int, breed class index in [0, 36]
        "bbox"   — FloatTensor [4], (x_center, y_center, w, h) in pixel coords
        "mask"   — LongTensor [224, 224], trimap values {0=foreground, 1=bg, 2=border}
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        transform: Optional[A.Compose] = None,
    ) -> None:
        assert split in (
            "train",
            "val",
            "test",
        ), f"split must be 'train', 'val', or 'test'; got '{split}'"

        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.xml_dir = os.path.join(self.ann_dir, "xmls")
        self.mask_dir = os.path.join(self.ann_dir, "trimaps")

        # get image name -> class index mapping
        label_map = self._parse_list_txt()

        # figure out which image names belong to this split
        if split == "test":
            names = self._read_split_file("test.txt")
        else:
            all_tv = self._read_split_file("trainval.txt")
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(all_tv))
            cut = int(len(all_tv) * val_fraction)
            val_set = set(perm[:cut].tolist())
            if split == "val":
                names = [all_tv[i] for i in range(len(all_tv)) if i in val_set]
            else:
                names = [all_tv[i] for i in range(len(all_tv)) if i not in val_set]

        # build the sample list, skip any entries with missing files
        self.samples: List[Dict] = []
        for name in names:
            img_path = self._find_image(name)
            xml_path = os.path.join(self.xml_dir, name + ".xml")
            mask_path = os.path.join(self.mask_dir, name + ".png")

            if img_path is None:
                continue
            if name not in label_map:
                continue
            if not os.path.exists(xml_path) or not os.path.exists(mask_path):
                continue

            self.samples.append(
                {
                    "name": name,
                    "img_path": img_path,
                    "xml_path": xml_path,
                    "mask_path": mask_path,
                    "label": label_map[name],
                }
            )

        # set the transform pipeline
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
        """Try common image extensions and return the path if found."""
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.img_dir, name + ext)
            if os.path.exists(p):
                return p
        return None

    def _parse_list_txt(self) -> Dict[str, int]:
        """Read annotations/list.txt and return {image_name: class_index} (0-based)."""
        path = os.path.join(self.ann_dir, "list.txt")
        mapping: Dict[str, int] = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name = parts[0]
                class_id = int(parts[1]) - 1  # list.txt uses 1-based indexing
                mapping[name] = class_id
        return mapping

    def _read_split_file(self, filename: str) -> List[str]:
        """Read a split file and return list of image names (no extension)."""
        path = os.path.join(self.ann_dir, filename)
        names: List[str] = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.append(line.split()[0])
        return names

    def _parse_bbox_xml(self, xml_path: str) -> Tuple[float, float, float, float]:
        """Parse Pascal VOC XML and return (xmin, ymin, xmax, ymax) of first object."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        bb = obj.find("bndbox")
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

        # load image as numpy array
        image = np.array(Image.open(sample["img_path"]).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # load trimap mask
        # Oxford trimaps use pixel values 1=foreground, 2=background, 3=border
        # we remap to 0-based: 0=foreground, 1=background, 2=border
        mask = np.array(Image.open(sample["mask_path"]).convert("L"), dtype=np.uint8)
        mask = np.clip(mask.astype(np.int32) - 1, 0, 2).astype(np.uint8)

        # load and clip bounding box
        xmin, ymin, xmax, ymax = self._parse_bbox_xml(sample["xml_path"])
        xmin = float(np.clip(xmin, 0, orig_w - 1))
        ymin = float(np.clip(ymin, 0, orig_h - 1))
        xmax = float(np.clip(xmax, xmin + 1, orig_w))
        ymax = float(np.clip(ymax, ymin + 1, orig_h))

        # apply augmentation pipeline
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[[xmin, ymin, xmax, ymax]],
            bbox_labels=[0],  # dummy label, we only have one box per image
        )

        image_t = transformed["image"]        # FloatTensor [3, 224, 224]
        mask_t = transformed["mask"].long()   # LongTensor  [224, 224]

        # get transformed bbox — albumentations might drop it if it goes out of frame
        bboxes_out = transformed["bboxes"]
        if len(bboxes_out) > 0:
            tx1, ty1, tx2, ty2 = bboxes_out[0]
        else:
            # fallback to full image if bbox got dropped
            tx1, ty1, tx2, ty2 = 0.0, 0.0, float(IMG_SIZE), float(IMG_SIZE)

        # convert [xmin, ymin, xmax, ymax] -> [x_center, y_center, width, height]
        x_center = (tx1 + tx2) / 2.0
        y_center = (ty1 + ty2) / 2.0
        width = tx2 - tx1
        height = ty2 - ty1
        bbox_t = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

        return {
            "image": image_t,        # FloatTensor [3, 224, 224]
            "label": sample["label"],# int in [0, 36]
            "bbox": bbox_t,          # FloatTensor [4]
            "mask": mask_t,          # LongTensor  [224, 224]
        }
