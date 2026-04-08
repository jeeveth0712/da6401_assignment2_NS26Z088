"""Reusable custom layers."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Inverted dropout implemented from scratch (no torch.nn.Dropout).

    During **training**: each element is independently zeroed with probability
    ``p``, and the surviving activations are scaled by ``1 / (1 - p)`` so that
    the expected value of each unit is preserved (inverted / scaled dropout).

    During **evaluation** (``self.training == False``): acts as an identity —
    no masking, no scaling.

    Args:
        p (float): Probability of zeroing an element. Must be in ``[0, 1)``.

    Example::
        >>> drop = CustomDropout(p=0.5)
        >>> x = torch.ones(2, 4)
        >>> drop.train(); drop(x)   # ~half zeros, rest ≈ 2.0
        >>> drop.eval();  drop(x)   # identical to x
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverted dropout during training; identity during eval.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as ``x``.
        """
        if not self.training or self.p == 0.0:
            return x

        # Bernoulli mask: 1 with prob (1 - p), 0 with prob p
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # Inverted scaling: divide by keep_prob to preserve expected activation
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
