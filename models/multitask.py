"""Unified multi-task model for Task 4.

Single forward pass gives all three outputs — classification, localization, segmentation.

Each task keeps its own separately trained encoder so we don't mess up the
feature distributions. The heads are directly lifted from the individual checkpoints.
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def _load_state_dict(path: str, map_location="cpu") -> dict:
    """Load checkpoint file — handles both plain state_dict and wrapped dicts."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class MultiTaskPerceptionModel(nn.Module):
    """Multi-task model that runs classification, localization, and segmentation together.

    On init, we download the three checkpoints from Google Drive and load
    the weights. Each task gets its own encoder (no sharing) to avoid
    feature distribution mismatch.

    The architecture looks like:

        Input image [B, 3, 224, 224]
              |           |           |
        clf.encoder   loc.encoder  unet.encoder
              |           |           |
        cls_head      loc_head     decoder
              |           |           |
           [B,37]       [B,4]    [B,3,H,W]

    Args:
        num_breeds:       Number of breed classes, default 37.
        seg_classes:      Number of segmentation classes, default 3.
        in_channels:      Input channels, default 3.
        classifier_path:  Path to save/load classifier.pth.
        localizer_path:   Path to save/load localizer.pth.
        unet_path:        Path to save/load unet.pth.
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

        # download checkpoints from Google Drive
        import gdown

        gdown.download(
            id="1SHFBmGRvpbikYZp6DyvUjUNp_DVhRdbL", output=classifier_path, quiet=False
        )
        gdown.download(
            id="1UQxT9VIPxIopPnogHDXtg5YkJM2fguH6", output=localizer_path, quiet=False
        )
        gdown.download(
            id="1RLY5TM0M901MfjEpVsYrfMhm8TvpqjAg", output=unet_path, quiet=False
        )

        # load each task model from its checkpoint
        clf = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        clf.load_state_dict(_load_state_dict(classifier_path))
        clf.eval()

        loc = VGG11Localizer(in_channels=in_channels)
        loc.load_state_dict(_load_state_dict(localizer_path))
        loc.eval()

        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        unet.load_state_dict(_load_state_dict(unet_path))
        unet.eval()

        # each task has its own encoder with its own fine-tuned weights
        self.encoder: VGG11Encoder = clf.encoder       # classification encoder
        self.loc_encoder: VGG11Encoder = loc.encoder   # localization encoder
        self.seg_encoder: VGG11Encoder = unet.encoder  # segmentation encoder

        # pull the heads from each task model
        self.classifier_head: nn.Sequential = clf.classifier_head
        self.localization_head: nn.Sequential = loc.localization_head
        self.decoder = unet.decoder

    def forward(self, x: torch.Tensor) -> dict:
        """One forward pass — returns all three task outputs.

        Args:
            x: Input image [B, in_channels, 224, 224].

        Returns:
            Dict with:
                'classification': [B, num_breeds] logits
                'localization':   [B, 4] bbox (cx, cy, w, h) in pixel space
                'segmentation':   [B, seg_classes, 224, 224] logits
        """
        # classification — use clf encoder
        bottleneck, skips = self.encoder(x, return_features=True)
        flat = bottleneck.view(bottleneck.size(0), -1)  # [B, 25088]
        cls_out = self.classifier_head(flat)             # [B, num_breeds]

        # localization — use its own encoder
        loc_bottleneck, _ = self.loc_encoder(x, return_features=True)
        loc_flat = loc_bottleneck.view(loc_bottleneck.size(0), -1)  # [B, 25088]
        loc_out = self.localization_head(loc_flat) * 224            # [B, 4] in pixels

        # segmentation — use unet encoder + decoder
        seg_bottleneck, seg_skips = self.seg_encoder(x, return_features=True)
        seg_out = self.decoder(seg_bottleneck, seg_skips)           # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }
