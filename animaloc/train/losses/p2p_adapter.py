"""
P2P Loss Adapter for auxiliary point-to-point supervision.

Bridges the standard HerdNet loss interface (pred_tensor, target_tensor)
with the HungarianLoss interface (outputs_dict, targets_list).

The P2P model output dict is stored on the model as `_p2p_output` during
the forward pass. This adapter retrieves it and extracts point targets
from the FIDT heatmap ground truth.
"""

import torch
import torch.nn as nn
from .p2p import HungarianLoss


class P2PLossAdapter(nn.Module):
    """
    Adapter that computes HungarianLoss from stored P2P outputs and FIDT GT.

    During training, the model stores its P2P predictions in `model._p2p_output`.
    This loss:
    1. Retrieves the stored P2P output dict
    2. Extracts point locations from the FIDT heatmap target (local maxima)
    3. Calls HungarianLoss with the proper format

    Args:
        cost_class: Hungarian matching class cost weight
        cost_point: Hungarian matching point cost weight
        cls_weight: Classification loss weight in total
        reg_weight: Regression loss weight in total
        image_size: Image size for coordinate normalization
        peak_threshold: Threshold for extracting GT points from FIDT heatmap
    """

    def __init__(
            self,
            cost_class: float = 2.0,
            cost_point: float = 5.0,
            cls_weight: float = 1.0,
            reg_weight: float = 5.0,
            image_size: int = 512,
            peak_threshold: float = 0.5,
    ):
        super().__init__()
        self.hungarian = HungarianLoss(
            cost_class=cost_class,
            cost_point=cost_point,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            image_size=image_size,
            debug=False,
        )
        self.peak_threshold = peak_threshold
        self._model_ref = None

    def set_model(self, model):
        """Store a reference to the model to access _p2p_output."""
        self._model_ref = model

    def _extract_points_from_fidt(self, fidt_map: torch.Tensor) -> list:
        """Extract point locations from FIDT heatmap by finding local maxima.

        Uses max-pooling to find peaks (same approach as LMDS detection).

        Args:
            fidt_map: [B, 1, H, W] FIDT ground truth heatmap

        Returns:
            List of dicts with 'points' key (normalized [0,1]), one per batch item
        """
        import torch.nn.functional as F

        B, _, H, W = fidt_map.shape
        targets = []

        for b in range(B):
            hm = fidt_map[b:b+1]  # [1, 1, H, W]

            # Find local maxima via max pooling (3x3 kernel)
            pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
            # A pixel is a local max if it equals the pooled value AND is above threshold
            peaks = (hm == pooled) & (hm > self.peak_threshold)
            peaks = peaks[0, 0]  # [H, W]

            if peaks.sum() == 0:
                targets.append({'points': torch.zeros(0, 2, device=hm.device)})
                continue

            ys, xs = torch.where(peaks)
            # Normalize to [0, 1] as (x, y)
            points_norm = torch.stack([xs.float() / W, ys.float() / H], dim=1)
            targets.append({'points': points_norm})

        return targets

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Ignored (P2P output is read from model._p2p_output)
            gt: FIDT heatmap target [B, 1, H, W] — used to extract point locations

        Returns:
            Scalar P2P loss
        """
        # Get P2P output from model
        model = self._model_ref
        if model is None or not hasattr(model, '_p2p_output') or model._p2p_output is None:
            return torch.tensor(0.0, device=gt.device, requires_grad=True)

        p2p_out = model._p2p_output

        # Extract GT points from FIDT heatmap
        targets = self._extract_points_from_fidt(gt)

        # Compute Hungarian loss
        loss = self.hungarian(p2p_out, targets)

        # Clear stored output
        model._p2p_output = None

        return loss
