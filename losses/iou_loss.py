"""IoU loss for bounding box regression.

Boxes should be in (x_center, y_center, width, height) format, in pixels.
Loss = 1 - IoU, so it's between 0 and 1.
0 means perfect overlap, 1 means no overlap at all.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """Computes 1 - IoU between predicted and target bounding boxes.

    Basically we compute how much the predicted box overlaps with the ground truth,
    and then subtract from 1 so that lower overlap = higher loss.

    Loss formula:
        L_IoU = 1 - (Intersection Area) / (Union Area + eps)

    Args:
        eps:       Small value to avoid divide by zero. Default 1e-6.
        reduction: How to combine losses across the batch.
                   'mean' (default), 'sum', or 'none' (per sample).

    Raises:
        ValueError: If reduction is not one of the three valid options.

    Example::
        >>> criterion = IoULoss(reduction='mean')
        >>> pred   = torch.tensor([[112., 112., 100., 80.]])
        >>> target = torch.tensor([[112., 112., 100., 80.]])
        >>> criterion(pred, target)   # should be 0.0 since boxes are identical
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
        """Calculate IoU loss between predicted and ground truth boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes in (cx, cy, w, h) pixel format.
            target_boxes: [B, 4] ground truth boxes in same format.

        Returns:
            Scalar if reduction is 'mean' or 'sum'.
            [B] tensor if reduction is 'none'.
        """
        # convert (cx, cy, w, h) to (x1, y1, x2, y2) for easier area calculation
        def to_xyxy(boxes: torch.Tensor):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        p_x1, p_y1, p_x2, p_y2 = to_xyxy(pred_boxes)
        t_x1, t_y1, t_x2, t_y2 = to_xyxy(target_boxes)

        # find intersection rectangle
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        inter_w    = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h    = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h

        # union = sum of both areas minus the overlap
        pred_area   = (p_x2 - p_x1).clamp(min=0.0) * (p_y2 - p_y1).clamp(min=0.0)
        target_area = (t_x2 - t_x1).clamp(min=0.0) * (t_y2 - t_y1).clamp(min=0.0)
        union_area  = pred_area + target_area - inter_area

        # iou and final loss
        iou  = inter_area / (union_area + self.eps)
        loss = 1.0 - iou          # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss               # 'none' — return per-sample losses

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
