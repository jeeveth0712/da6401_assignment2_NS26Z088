"""U-Net segmentation model for Task 3 — trimap prediction.

3 classes: 0 = foreground (the pet), 1 = background, 2 = border/uncertain region.

Architecture:
    Encoder : VGG11Encoder  (5 conv blocks + 5 max pool layers)
    Decoder : UNetDecoder   (5 transposed conv upsamplings with skip connections)

Important constraint: upsampling must be done only with ConvTranspose2d,
no bilinear interpolation allowed.

Loss used during training: CrossEntropyLoss + soft Dice loss (handles class imbalance).
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


def _dec_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 convs with BN and ReLU — standard decoder block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetDecoder(nn.Module):
    """Decoder that mirrors VGG11 encoder — takes bottleneck + skip connections
    and gradually upsamples back to full 224x224 resolution.

    Skip connection sizes coming from the encoder:
        skip5 : [B, 512, 14, 14]
        skip4 : [B, 512, 28, 28]
        skip3 : [B, 256, 56, 56]
        skip2 : [B, 128, 112, 112]
        skip1 : [B, 64,  224, 224]

    How upsampling happens step by step (bottleneck is [B, 512, 7, 7]):
        up1 -> [B,512,14,14]   concat skip5 -> [B,1024,14,14]  -> dec1 -> [B,512,14,14]
        up2 -> [B,512,28,28]   concat skip4 -> [B,1024,28,28]  -> dec2 -> [B,512,28,28]
        up3 -> [B,256,56,56]   concat skip3 -> [B, 512,56,56]  -> dec3 -> [B,256,56,56]
        up4 -> [B,128,112,112] concat skip2 -> [B, 256,112,112]-> dec4 -> [B,128,112,112]
        up5 -> [B, 64,224,224] concat skip1 -> [B, 128,224,224]-> dec5 -> [B, 64,224,224]
        final_conv -> [B, num_classes, 224, 224]

    Args:
        num_classes: Number of output segmentation classes, default 3.
        dropout_p:   Dropout before final conv, default 0.5.
    """

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5) -> None:
        super().__init__()

        # transposed convolutions for upsampling — no bilinear allowed
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # 7  -> 14
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   # 14 -> 28
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 28 -> 56
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 56 -> 112
        self.up5 = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)   # 112-> 224

        # after each upsample we concat the skip and run through a conv block
        self.dec1 = _dec_conv_block(512 + 512, 512)   # skip5 has 512 channels
        self.dec2 = _dec_conv_block(512 + 512, 512)   # skip4 has 512 channels
        self.dec3 = _dec_conv_block(256 + 256, 256)   # skip3 has 256 channels
        self.dec4 = _dec_conv_block(128 + 128, 128)   # skip2 has 128 channels
        self.dec5 = _dec_conv_block( 64 +  64,  64)   # skip1 has 64 channels

        self.dropout    = CustomDropout(dropout_p)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: dict,
    ) -> torch.Tensor:
        """Upsample bottleneck back to full resolution using skip connections.

        Args:
            bottleneck: [B, 512, 7, 7] from encoder.
            skips:      Dict with 'skip1' to 'skip5' feature maps.

        Returns:
            Segmentation logits [B, num_classes, 224, 224].
        """
        x = self.up1(bottleneck)                                    # [B,512,14,14]
        x = self.dec1(torch.cat([x, skips["skip5"]], dim=1))       # [B,512,14,14]

        x = self.up2(x)                                             # [B,512,28,28]
        x = self.dec2(torch.cat([x, skips["skip4"]], dim=1))       # [B,512,28,28]

        x = self.up3(x)                                             # [B,256,56,56]
        x = self.dec3(torch.cat([x, skips["skip3"]], dim=1))       # [B,256,56,56]

        x = self.up4(x)                                             # [B,128,112,112]
        x = self.dec4(torch.cat([x, skips["skip2"]], dim=1))       # [B,128,112,112]

        x = self.up5(x)                                             # [B,64,224,224]
        x = self.dec5(torch.cat([x, skips["skip1"]], dim=1))       # [B,64,224,224]

        x = self.dropout(x)
        return self.final_conv(x)                                   # [B,C,224,224]


class VGG11UNet(nn.Module):
    """Full U-Net with VGG11 encoder for trimap segmentation.

    Args:
        num_classes: How many classes to predict, default 3.
        in_channels: Input channels, default 3.
        dropout_p:   Dropout in the decoder, default 0.5.

    Example::
        >>> model = VGG11UNet(num_classes=3)
        >>> mask_logits = model(torch.randn(2, 3, 224, 224))
        >>> mask_logits.shape
        torch.Size([2, 3, 224, 224])
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.decoder = UNetDecoder(num_classes, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run image through encoder then decoder.

        Args:
            x: Input image [B, in_channels, 224, 224].

        Returns:
            Segmentation logits [B, num_classes, 224, 224].
        """
        bottleneck, skips = self.encoder(x, return_features=True)
        return self.decoder(bottleneck, skips)
