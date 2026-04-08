"""Custom Intersection-over-Union (IoU) loss for bounding box regression.

Boxes must be supplied in ``(x_center, y_center, width, height)`` format,
in pixel coordinates.  The loss equals ``1 − IoU`` and is therefore in
``[0, 1]``.  Gradients flow back through both predicted and target boxes
(though targets are typically detached during training).
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss between predicted and target bounding boxes.

    .. math::
        \\mathcal{L}_{IoU} = 1 - \\frac{\\text{Area}(\\text{Intersection})}{
            \\text{Area}(\\text{Union}) + \\varepsilon}

    The loss is in ``[0, 1]``:
        * ``0``  — perfect overlap (prediction == target).
        * ``1``  — zero overlap.

    Args:
        eps:       Small constant added to the denominator for numerical
                   stability (avoids division by zero).  Default: ``1e-6``.
        reduction: How to reduce the per-sample losses:
                   ``'mean'`` (default) | ``'sum'`` | ``'none'``.

    Raises:
        ValueError: If ``reduction`` is not one of the three supported values.

    Example::
        >>> criterion = IoULoss(reduction='mean')
        >>> pred   = torch.tensor([[112., 112., 100., 80.]])
        >>> target = torch.tensor([[112., 112., 100., 80.]])
        >>> criterion(pred, target)   # ≈ 0.0  (perfect overlap)
        tensor(0.)
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got '{reduction}'"
            )
        self.eps       = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes:   torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the IoU loss.

        Args:
            pred_boxes:   ``[B, 4]`` predicted boxes
                          ``(x_center, y_center, width, height)`` in pixels.
            target_boxes: ``[B, 4]`` ground-truth boxes in the same format.

        Returns:
            * Scalar tensor if ``reduction`` is ``'mean'`` or ``'sum'``.
            * ``[B]`` tensor of per-sample losses if ``reduction='none'``.
        """
        # ── Convert (cx, cy, w, h) → (x1, y1, x2, y2) ─────────────────
        def to_xyxy(boxes: torch.Tensor):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        p_x1, p_y1, p_x2, p_y2 = to_xyxy(pred_boxes)
        t_x1, t_y1, t_x2, t_y2 = to_xyxy(target_boxes)

        # ── Intersection ────────────────────────────────────────────────
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        inter_w    = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h

        # ── Union ────────────────────────────────────────────────────────
        pred_area   = (p_x2 - p_x1).clamp(min=0.0) * (p_y2 - p_y1).clamp(min=0.0)
        target_area = (t_x2 - t_x1).clamp(min=0.0) * (t_y2 - t_y1).clamp(min=0.0)
        union_area  = pred_area + target_area - inter_area

        # ── IoU and loss (both in [0, 1]) ────────────────────────────────
        iou  = inter_area / (union_area + self.eps)
        loss = 1.0 - iou          # shape [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss               # 'none' — return per-sample tensor [B]

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
