"""VGG11 encoder backbone and full VGG11 classification model.

Architecture follows Table 1, column "A" of:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
    Image Recognition", arXiv:1409.1556.

Modifications vs. the paper:
    * BatchNorm2d inserted after every Conv2d (before ReLU) in the encoder.
    * BatchNorm1d inserted after every Linear (before ReLU) in the FC head.
    * CustomDropout replaces nn.Dropout in the FC head.
    * Final FC outputs ``num_classes`` logits instead of 1000.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout

# Fixed input size per VGG paper
IMG_SIZE: int = 224


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 conv → BN → ReLU building block (same-padding)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11 convolutional backbone with optional skip-connection output.

    Spatial resolution at each block boundary (224×224 input):
        block1 output : [B, 64,  224, 224]   pool1 → [B, 64,  112, 112]
        block2 output : [B, 128, 112, 112]   pool2 → [B, 128,  56,  56]
        block3 output : [B, 256,  56,  56]   pool3 → [B, 256,  28,  28]
        block4 output : [B, 512,  28,  28]   pool4 → [B, 512,  14,  14]
        block5 output : [B, 512,  14,  14]   pool5 → [B, 512,   7,   7]  ← bottleneck

    Args:
        in_channels: Number of input image channels (default 3).
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # ── Block 1 — 1 conv, 64 filters ────────────────────────────────
        self.block1 = _conv_bn_relu(in_channels, 64)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 2 — 1 conv, 128 filters ───────────────────────────────
        self.block2 = _conv_bn_relu(64, 128)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 3 — 2 convs, 256 filters ──────────────────────────────
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 4 — 2 convs, 512 filters ──────────────────────────────
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 5 — 2 convs, 512 filters ──────────────────────────────
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5  = nn.MaxPool2d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the VGG11 convolutional backbone.

        Args:
            x:               Input image tensor ``[B, in_channels, H, W]``.
            return_features: When *True*, also return a dict of skip-connection
                             feature maps (pre-pool) for U-Net decoder fusion.

        Returns:
            * ``return_features=False``: bottleneck tensor ``[B, 512, 7, 7]``.
            * ``return_features=True``: ``(bottleneck, feature_dict)`` where
              ``feature_dict`` has keys ``'skip1' … 'skip5'``.
        """
        s1 = self.block1(x)     # [B,  64, 224, 224]
        x  = self.pool1(s1)     # [B,  64, 112, 112]

        s2 = self.block2(x)     # [B, 128, 112, 112]
        x  = self.pool2(s2)     # [B, 128,  56,  56]

        s3 = self.block3(x)     # [B, 256,  56,  56]
        x  = self.pool3(s3)     # [B, 256,  28,  28]

        s4 = self.block4(x)     # [B, 512,  28,  28]
        x  = self.pool4(s4)     # [B, 512,  14,  14]

        s5 = self.block5(x)     # [B, 512,  14,  14]
        x  = self.pool5(s5)     # [B, 512,   7,   7]  ← bottleneck

        if return_features:
            return x, {
                "skip1": s1,
                "skip2": s2,
                "skip3": s3,
                "skip4": s4,
                "skip5": s5,
            }
        return x


class VGG11(nn.Module):
    """Full VGG11 model: convolutional backbone + FC classification head.

    The FC head follows the VGG paper (two 4096-unit hidden layers) with
    BatchNorm1d and CustomDropout added after each hidden layer.

    Args:
        num_classes: Number of output logits (default 37 for Oxford-IIIT Pet).
        in_channels: Number of input channels (default 3).
        dropout_p:   Dropout probability for the FC head (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        # FC head: 512×7×7 = 25 088 → 4096 → 4096 → num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

        self._init_fc_weights()

    def _init_fc_weights(self) -> None:
        for m in self.classifier_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``[B, in_channels, 224, 224]``.

        Returns:
            Classification logits ``[B, num_classes]``.
        """
        features = self.encoder(x)                         # [B, 512, 7, 7]
        flat     = features.view(features.size(0), -1)     # [B, 25088]
        return self.classifier_head(flat)                  # [B, num_classes]
