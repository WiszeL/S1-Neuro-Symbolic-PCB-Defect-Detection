from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .geometry import project_gt_box_to_roi_grid

_NAN = float("nan")
LOW_GT_COVERAGE_THRESHOLD = 0.5


def normalize_heatmap(heatmap: Tensor | np.ndarray) -> Tensor:
    """Common scale so heatmaps from different methods compare fairly."""
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    tensor = torch.clamp(tensor, min=0.0)
    total = tensor.sum().item()
    if total <= 0.0:
        return torch.zeros_like(tensor)
    return tensor / total


def pointing_score(heatmap: Tensor, gt_mask: Tensor) -> float:
    """1.0 when the hottest cell lands on the defect, else 0.0."""
    peak_index = torch.argmax(heatmap.reshape(-1)).item()
    row_index = peak_index // heatmap.shape[1]
    col_index = peak_index % heatmap.shape[1]
    return float(gt_mask[row_index, col_index].item())


def importance_ranking(heatmap: Tensor | np.ndarray) -> Tensor:
    """One ranking rule for every method, so the comparison stays fair."""
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    return torch.argsort(tensor.reshape(-1), descending=True)


def topk_region_overlap(heatmap: Tensor, gt_mask: Tensor) -> float:
    """Overlap between the hottest cells and the defect area."""
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


def spatial_result(count: int, overlap: float, pointing: float) -> dict[str, Any]:
    """One result shape so every method reports side by side."""
    return {
        "evaluated_roi_count": count,
        "box_grounded_roi_overlap": overlap,
        "pointing_score": pointing,
    }


_EMPTY_SPATIAL_RESULT = spatial_result(0, _NAN, _NAN)


def stratified_spatial_result(
    overlap_scores: list[float],
    pointing_scores: list[float],
    gt_coverage: list[float],
) -> dict[str, Any]:
    """Overall scores plus a low-coverage breakdown.

    Tight proposals saturate these metrics near chance, so the
    low-coverage subset is the only part that separates methods.
    """
    if not overlap_scores:
        result = dict(_EMPTY_SPATIAL_RESULT)
        result["low_gt_coverage"] = dict(_EMPTY_SPATIAL_RESULT)
        return result

    result = spatial_result(
        len(overlap_scores),
        float(np.mean(overlap_scores)),
        float(np.mean(pointing_scores)),
    )

    coverage = np.asarray(gt_coverage, dtype=np.float64)
    low_mask = coverage < LOW_GT_COVERAGE_THRESHOLD
    low_overlap = [score for score, keep in zip(overlap_scores, low_mask) if keep]
    low_pointing = [score for score, keep in zip(pointing_scores, low_mask) if keep]
    result["low_gt_coverage"] = (
        spatial_result(
            len(low_overlap), float(np.mean(low_overlap)), float(np.mean(low_pointing))
        )
        if low_overlap
        else dict(_EMPTY_SPATIAL_RESULT)
    )
    return result


def evaluate_random_baseline_spatial_metrics(
    proposal_boxes: Tensor,
    matched_gt_boxes: Tensor | None,
    has_matched_gt: Tensor | None,
    gt_iou: Tensor | None,
    grid_shape: tuple[int, int] = (7, 7),
    min_proposal_iou: float = 0.0,
    max_proposal_iou: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Chance-level reference on the same RoIs, so high-looking scores can't pass as wins."""
    if matched_gt_boxes is None or has_matched_gt is None:
        return stratified_spatial_result([], [], [])

    rng = np.random.default_rng(random_state)
    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    gt_coverage: list[float] = []

    for index in range(proposal_boxes.shape[0]):
        if not bool(has_matched_gt[index]):
            continue
        if gt_iou is not None and not (min_proposal_iou <= float(gt_iou[index]) <= max_proposal_iou):
            continue

        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=grid_shape,
        )
        if gt_mask.sum().item() == 0:
            continue

        random_heatmap = torch.from_numpy(rng.random(grid_shape).astype(np.float32))
        normalized_heatmap = normalize_heatmap(random_heatmap)
        overlap_scores.append(topk_region_overlap(normalized_heatmap, gt_mask))
        pointing_scores.append(pointing_score(normalized_heatmap, gt_mask))
        gt_coverage.append(float(gt_mask.float().mean().item()))

    return stratified_spatial_result(overlap_scores, pointing_scores, gt_coverage)
