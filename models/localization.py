"""Localization model — VGG11 encoder + regression head.

Task 2: Predict a single bounding box per image in
``(x_center, y_center, width, height)`` pixel coordinates.

Training loss (applied externally in train.py):
    L = MSELoss(pred, target) + IoULoss(pred, target)
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """Single-object bounding-box regressor built on VGG11.

    Architecture:
        VGG11Encoder  →  flatten [B, 25088]
            → Linear(25088, 4096) → BN → ReLU → Dropout
            → Linear(4096, 1024)  → ReLU
            → Linear(1024, 4)     [raw pixel output]

    The output is **not** passed through a sigmoid or any other bounded
    activation; the MSE + IoU training loss drives the predictions into
    the correct pixel-coordinate range naturally.

    Args:
        in_channels: Number of input channels (default 3).
        dropout_p:   Dropout probability in the regression head (default 0.5).

    Example::
        >>> model = VGG11Localizer()
        >>> bbox = model(torch.randn(2, 3, 224, 224))
        >>> bbox.shape
        torch.Size([2, 4])
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5) -> None:
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        self.localization_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),      # (x_center, y_center, width, height)
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.localization_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[B, in_channels, 224, 224]``.

        Returns:
            Bounding box ``[B, 4]`` as
            ``(x_center, y_center, width, height)`` in pixel space
            (not normalised).
        """
        features = self.encoder(x)                         # [B, 512, 7, 7]
        flat     = features.view(features.size(0), -1)     # [B, 25088]
        out      = self.localization_head(flat)             # [B, 4] in [0, 1]
        return out * 224                                    # scale to pixel space [0, 224]
