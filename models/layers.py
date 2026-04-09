"""Custom layer definitions — currently only has CustomDropout."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Dropout written from scratch without using nn.Dropout.

    During training, each value is randomly set to zero with probability p.
    The remaining values are scaled up by 1/(1-p) so the overall magnitude
    stays the same — this is called inverted dropout.

    During eval mode it just returns the input as-is, no changes.

    Args:
        p (float): How likely each elem
        ent gets zeroed out. Should be in [0, 1).

    Example::
        >>> drop = CustomDropout(p=0.5)
        >>> x = torch.ones(2, 4)
        >>> drop.train(); drop(x)   # roughly half will be zero, rest will be 2.0
        >>> drop.eval();  drop(x)   # same as input, no dropout
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies dropout during training, does nothing during eval.

        Args:
            x: Any tensor, shape doesn't matter.

        Returns:
            Same shape tensor as input.
        """
        if not self.training or self.p == 0.0:
            return x

        # randomly keep each element with probability (1 - p)
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # scale up the surviving values so expected value is preserved
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
