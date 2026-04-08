"""Unified multi-task perception model.

Task 4: Single forward pass → classification logits + bounding box + segmentation mask.

Backbone sharing strategy:
    The UNet checkpoint is used as the shared backbone because Task 3 (segmentation)
    requires the richest feature representations and is the most demanding task.
    The classifier head is lifted from classifier.pth and the localization head
    from localizer.pth, keeping all task-specific weights intact.
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def _load_state_dict(path: str, map_location="cpu") -> dict:
    """Load a checkpoint that may be either a plain state_dict or a wrapped dict."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    On ``__init__``, the three task-specific checkpoints are downloaded from
    Google Drive (via ``gdown``) and the weights are loaded into a unified
    architecture:

    .. code-block::

        Input image [B, 3, 224, 224]
              │
        ┌─────┴──────────────────────────┐
        │   Shared VGG11Encoder          │
        │   (weights from unet.pth)      │
        └─────┬──────────┬──────────┬───┘
              │          │          │
        cls_head      loc_head   decoder
        (clf.pth)   (loc.pth)  (unet.pth)
              │          │          │
        [B,37]       [B,4]   [B,3,H,W]

    Args:
        num_breeds:       Number of breed classes (default 37).
        seg_classes:      Number of segmentation classes (default 3).
        in_channels:      Number of input channels (default 3).
        classifier_path:  Relative path to ``classifier.pth``.
        localizer_path:   Relative path to ``localizer.pth``.
        unet_path:        Relative path to ``unet.pth``.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ) -> None:
        super().__init__()

        # ── Download checkpoints from Google Drive ────────────────────────
        import gdown

        gdown.download(
            id="1Flr4x8YI1BH-KCYo0pV5G2L1h4ng6_K4", output=classifier_path, quiet=False
        )
        gdown.download(
            id="1726sAG0UGdmUbR5v1Xc9WeNQ7EKkC9d8", output=localizer_path, quiet=False
        )
        gdown.download(
            id="1HXr-w2ei7OiKXI5m9oH-gegb43eBdrMZ", output=unet_path, quiet=False
        )

        # ── Instantiate and load each task model ──────────────────────────
        clf = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        clf.load_state_dict(_load_state_dict(classifier_path))
        clf.eval()

        loc = VGG11Localizer(in_channels=in_channels)
        loc.load_state_dict(_load_state_dict(localizer_path))
        loc.eval()

        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        unet.load_state_dict(_load_state_dict(unet_path))
        unet.eval()

        # ── Build the shared backbone + task-specific heads ───────────────
        # Shared encoder: taken from the classifier since both classifier_head
        # and localization_head were trained starting from classifier weights.
        # Using classifier encoder avoids feature mismatch for clf and loc heads.
        self.encoder: VGG11Encoder = clf.encoder

        # Classification head from Task 1
        self.classifier_head: nn.Sequential = clf.classifier_head

        # Localization head from Task 2
        self.localization_head: nn.Sequential = loc.localization_head

        # Segmentation decoder from Task 3
        self.decoder = unet.decoder

    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass producing all three task outputs.

        Args:
            x: Input tensor ``[B, in_channels, 224, 224]``.

        Returns:
            A ``dict`` with keys:

            * ``'classification'``: ``[B, num_breeds]`` logits.
            * ``'localization'``:   ``[B, 4]`` bounding box
              ``(x_center, y_center, width, height)`` in pixel space.
            * ``'segmentation'``:   ``[B, seg_classes, 224, 224]`` logits.
        """
        bottleneck, skips = self.encoder(x, return_features=True)
        flat = bottleneck.view(bottleneck.size(0), -1)  # [B, 25088]

        cls_out = self.classifier_head(flat)  # [B, num_breeds]
        loc_out = self.localization_head(flat)  # [B, 4]
        seg_out = self.decoder(bottleneck, skips)  # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
