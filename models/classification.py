"""Classification components.

VGG11Classifier is the full VGG11 model (encoder + FC head) used for
Task 1 — breed classification on the Oxford-IIIT Pet dataset.
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11


class VGG11Classifier(VGG11):
    """Full VGG11 classifier.

    Inherits the complete VGG11 architecture (convolutional backbone +
    FC head).  Provided as a named alias so training and inference scripts
    can import it from ``models.classification`` without depending directly
    on ``models.vgg11``.

    Attributes (inherited from VGG11):
        encoder (VGG11Encoder): Convolutional backbone.
        classifier_head (nn.Sequential): FC classification head.

    Args:
        num_classes: Number of breed classes (default 37).
        in_channels: Number of input channels (default 3).
        dropout_p:   Dropout probability for the FC head (default 0.5).

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
