"""Classification model for Task 1 — breed prediction.

VGG11Classifier is just VGG11 with a proper name so we can import it
cleanly from models.classification in training and inference scripts.
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11


class VGG11Classifier(VGG11):
    """VGG11 model for classifying pet breeds. Inherits everything from VGG11.

    We basically just renamed it so the import path makes more sense.
    All the actual logic (encoder + classifier_head) comes from VGG11 itself.

    Args:
        num_classes: Number of breed classes, default 37.
        in_channels: Input channels, default 3.
        dropout_p:   Dropout probability in FC head, default 0.5.

    Example::
        >>> model = VGG11Classifier(num_classes=37)
        >>> logits = model(torch.randn(2, 3, 224, 224))
        >>> logits.shape
        torch.Size([2, 37])
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
        )
