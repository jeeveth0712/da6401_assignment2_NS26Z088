"""VGG11 encoder and full classifier model.

Based on the VGG paper (Table 1, column A) by Simonyan & Zisserman.
We made a few changes compared to original:
    - Added BatchNorm2d after every Conv2d in the encoder (before ReLU)
    - Added BatchNorm1d after every Linear in the FC head (before ReLU)
    - Used our own CustomDropout instead of nn.Dropout
    - Output is num_classes logits instead of 1000
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout

# VGG paper uses 224x224 input
IMG_SIZE: int = 224


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Basic conv block: 3x3 conv -> BatchNorm -> ReLU (padding=1 to keep size same)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11 convolutional backbone. Can also return skip connections for U-Net.

    What the spatial size looks like after each block (input is 224x224):
        block1 -> [B, 64,  224, 224]   after pool1 -> [B, 64,  112, 112]
        block2 -> [B, 128, 112, 112]   after pool2 -> [B, 128,  56,  56]
        block3 -> [B, 256,  56,  56]   after pool3 -> [B, 256,  28,  28]
        block4 -> [B, 512,  28,  28]   after pool4 -> [B, 512,  14,  14]
        block5 -> [B, 512,  14,  14]   after pool5 -> [B, 512,   7,   7]  <- bottleneck

    Args:
        in_channels: Number of input channels, default is 3 for RGB images.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # block 1 — single conv, 64 filters
        self.block1 = _conv_bn_relu(in_channels, 64)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2 — single conv, 128 filters
        self.block2 = _conv_bn_relu(64, 128)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 — 2 convs, 256 filters
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4 — 2 convs, 512 filters
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 5 — 2 convs, 512 filters
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
        """Run the input through all 5 conv blocks.

        Args:
            x:               Input image tensor [B, in_channels, H, W].
            return_features: If True, also return the pre-pool feature maps
                             from each block — needed for U-Net skip connections.

        Returns:
            - If return_features=False: just the bottleneck [B, 512, 7, 7].
            - If return_features=True: (bottleneck, dict) where dict has
              keys 'skip1' through 'skip5'.
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
        x  = self.pool5(s5)     # [B, 512,   7,   7]  <- bottleneck

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
    """Full VGG11: encoder + FC head for classification.

    The FC head has two hidden layers of 4096 units each, with BatchNorm
    and CustomDropout after each one — as described in the VGG paper.

    Args:
        num_classes: How many output classes. Default is 37 for Oxford Pet dataset.
        in_channels: Input channels, default 3.
        dropout_p:   Dropout rate in the FC head, default 0.5.
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        # FC head: flatten 512x7x7 = 25088 -> 4096 -> 4096 -> num_classes
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
        """Forward pass — encoder then FC head.

        Args:
            x: Input image tensor [B, in_channels, 224, 224].

        Returns:
            Class logits [B, num_classes].
        """
        features = self.encoder(x)                         # [B, 512, 7, 7]
        flat     = features.view(features.size(0), -1)     # [B, 25088]
        return self.classifier_head(flat)                  # [B, num_classes]
