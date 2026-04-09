"""Localization model for Task 2 — predicting bounding box around the pet.

Output is (x_center, y_center, width, height) in pixel coordinates.

We trained this with MSELoss + IoULoss combined.
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """Bounding box regressor using VGG11 as backbone.

    Takes an image, runs it through VGG11 encoder, flattens,
    then passes through a regression head to predict 4 values (cx, cy, w, h).

    Sigmoid at the end keeps output in [0, 1], then we multiply by 224
    to get pixel coordinates.

    Args:
        in_channels: Input channels, default 3.
        dropout_p:   Dropout probability for regression head, default 0.5.

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
            nn.Linear(512, 4),      # outputs (x_center, y_center, width, height)
            nn.Sigmoid(),           # clamp to [0, 1] before scaling to pixels
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
        """Run image through encoder and regression head.

        Args:
            x: Input image [B, in_channels, 224, 224].

        Returns:
            Bounding box [B, 4] as (x_center, y_center, width, height) in pixels.
        """
        features = self.encoder(x)                         # [B, 512, 7, 7]
        flat     = features.view(features.size(0), -1)     # [B, 25088]
        out      = self.localization_head(flat)             # [B, 4] values in [0, 1]
        return out * 224                                    # scale to pixel space [0, 224]
