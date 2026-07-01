from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def normalize_heatmap(heatmap: Tensor | np.ndarray) -> Tensor:
    """Clamp negatives to zero and rescale so the heatmap sums to 1."""
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    tensor = torch.clamp(tensor, min=0.0)
    total = tensor.sum().item()
    if total <= 0.0:
        return torch.zeros_like(tensor)
    return tensor / total


def pointing_score(heatmap: Tensor, gt_mask: Tensor) -> float:
    """1.0 if the heatmap's peak cell falls inside the GT mask, else 0.0."""
    peak_index = torch.argmax(heatmap.reshape(-1)).item()
    row_index = peak_index // heatmap.shape[1]
    col_index = peak_index % heatmap.shape[1]
    return float(gt_mask[row_index, col_index].item())


def topk_region_overlap(heatmap: Tensor, gt_mask: Tensor) -> float:
    """IoU between the GT mask and the top-|GT| highest-activation cells."""
    target_cells = max(gt_mask.sum().item(), 1)
    flattened = heatmap.reshape(-1)
    topk_indices = torch.topk(flattened, k=min(target_cells, flattened.numel())).indices
    predicted_mask = torch.zeros_like(flattened, dtype=torch.bool)
    predicted_mask[topk_indices] = True
    predicted_mask = predicted_mask.reshape_as(gt_mask)
    intersection = torch.logical_and(predicted_mask, gt_mask).sum().item()
    union = torch.logical_or(predicted_mask, gt_mask).sum().item()
    if union == 0:
        return 0.0
    return float(intersection / union)
