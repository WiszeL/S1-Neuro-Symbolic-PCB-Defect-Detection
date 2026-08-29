from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .geometry import project_gt_box_to_roi_grid

_NAN = float("nan")
LOW_GT_COVERAGE_THRESHOLD = 0.5


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


def importance_ranking(heatmap: Tensor | np.ndarray) -> Tensor:
    """Flattened spatial cell indices, sorted by descending heatmap value.

    Shared by the symbolic and Grad-CAM faithfulness evaluators so both
    sides rank cells with the exact same rule — see the perturbation
    protocol docstrings in symbolic/evaluation.py and gradcam/evaluation.py.
    """
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    return torch.argsort(tensor.reshape(-1), descending=True)


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


def spatial_result(count: int, overlap: float, pointing: float) -> dict[str, Any]:
    """Shared result shape for pointing/IoU spatial metrics — used by the
    symbolic, Grad-CAM, and random-baseline evaluators so the three are
    reported side by side in the same schema."""
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
    """Overall spatial metrics plus a ``low_gt_coverage`` breakdown.

    GT boxes typically cover most of the RoI grid (the proposal is tight on
    the defect), which saturates pointing/IoU near a random baseline. The
    low-coverage subset (GT covers < LOW_GT_COVERAGE_THRESHOLD of the grid)
    is where these metrics still have room to discriminate between methods.
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
    """Uniform-random heatmap baseline, scored on the exact same RoI
    population and GT projection as the symbolic/Grad-CAM spatial metrics.

    Both real methods can land near this baseline purely because the GT box
    typically covers most of the grid — this control makes that visible
    instead of letting a high-looking number pass as a win. See
    stratified_spatial_result for the low-coverage subset where the metrics
    can still discriminate. `max_proposal_iou` narrows to loose proposals
    (e.g. 0.05-0.35) where the GT box covers only part of the grid — at the
    default min_proposal_iou=0.5 that subset is empty (tight proposals only),
    which is why the low-coverage table can otherwise report n=0.
    """
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
