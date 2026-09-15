"""How good the tree's heatmaps are, scored exactly like Grad-CAM's.

- Same test unit both sides: one grid cell, all channels at once.
- Each side checked against its own model, never the other's.
- Skipped on purpose: scoring the tree's own active features would always
  pass by construction, so it would prove nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .sodt import SparseObliqueDecisionTreeClassifier
from util.features import ensure_float32
from util.geometry import project_gt_box_to_roi_grid
from util.heatmap_metrics import (
    importance_ranking,
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)

_FAITHFULNESS_CELL_BUDGET_FRACTION = 0.5


def _safe_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> float:
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for symbolic evaluation metrics."
        ) from exc

    return float(
        f1_score(
            labels,
            predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )


def _per_class_agreement(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    per_class: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not np.any(mask):
            per_class[class_name] = float("nan")
            continue
        per_class[class_name] = float((predictions[mask] == labels[mask]).mean())
    return per_class


def _ensure_writable_tensor(
    data: Tensor | np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Copy first so torch never writes into read-only memory."""
    if isinstance(data, np.ndarray):
        data = data.copy()
    return torch.as_tensor(data, dtype=dtype)


def _build_selected_mask(
    n_rows: int,
    n_features: int,
    selected_per_row: list[np.ndarray],
) -> np.ndarray:
    mask = np.zeros((n_rows, n_features), dtype=bool)
    lengths = np.array([len(sel) for sel in selected_per_row], dtype=np.int64)
    total = lengths.sum()
    if total > 0:
        row_idx = np.repeat(np.arange(n_rows), lengths)
        col_idx = np.concatenate(selected_per_row)
        mask[row_idx, col_idx] = True
    return mask


def _row_cell_ranking(
    tree: SparseObliqueDecisionTreeClassifier, feature_row: np.ndarray
) -> np.ndarray:
    """Rank grid cells the same way the displayed heatmaps do."""
    grid = feature_row.reshape(tree.feature_shape)
    heatmap = _compute_local_instance_heatmap(tree, grid, mode="leaf_only")
    return importance_ranking(heatmap).cpu().numpy()


def _full_channel_flat_indices(
    cells: np.ndarray, feature_shape: tuple[int, int, int]
) -> np.ndarray:
    """One grid cell means every channel — the same unit Grad-CAM masks."""
    channels, height, width = feature_shape
    channel_offsets = (np.arange(channels) * height * width)[:, None]
    return (channel_offsets + cells[None, :]).reshape(-1).astype(np.int64)


def _deletion_insertion_auc(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    predictions: np.ndarray,
    num_samples: int = 1000,
    steps: int = 10,
    random_state: int = 42,
) -> tuple[float, float]:
    if tree.feature_shape is None:
        raise ValueError(
            "_deletion_insertion_auc requires tree.feature_shape to rank spatial cells."
        )
    _, height, width = tree.feature_shape
    num_cells = height * width

    N, D = features.shape
    if N > num_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(N, size=num_samples, replace=False)
    else:
        indices = np.arange(N)

    n = len(indices)
    auc_features = features[indices]
    auc_preds = predictions[indices]

    cell_rankings = [_row_cell_ranking(tree, auc_features[i]) for i in range(n)]

    deletion_curves = np.ones((n, steps + 1))
    insertion_curves = np.zeros((n, steps + 1))

    batch_size = 1024
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        B = batch_end - batch_start
        full_scores = tree.predict_scores(auc_features[batch_start:batch_end])
        empty_scores = tree.predict_scores(np.zeros((B, D), dtype=np.float32))
        for bi in range(B):
            gi = batch_start + bi
            deletion_curves[gi, 0] = full_scores[bi, auc_preds[gi]]
            insertion_curves[gi, 0] = empty_scores[bi, auc_preds[gi]]

        del_batch = auc_features[batch_start:batch_end].copy()
        ins_batch = np.zeros_like(del_batch)
        for s in range(1, steps + 1):
            k = int(s / steps * num_cells)
            for bi in range(B):
                gi = batch_start + bi
                top_cells = cell_rankings[gi][:k]
                flat_indices = _full_channel_flat_indices(
                    top_cells, tree.feature_shape
                )

                del_batch[bi] = auc_features[gi].copy()
                del_batch[bi, flat_indices] = 0.0
                ins_batch[bi] = 0.0
                ins_batch[bi, flat_indices] = auc_features[gi, flat_indices]

            del_scores = tree.predict_scores(del_batch)
            ins_scores = tree.predict_scores(ins_batch)
            for bi in range(B):
                gi = batch_start + bi
                deletion_curves[gi, s] = del_scores[bi, auc_preds[gi]]
                insertion_curves[gi, s] = ins_scores[bi, auc_preds[gi]]

    dx = 1.0 / steps
    return float(np.trapz(deletion_curves, dx=dx, axis=1).mean()), float(
        np.trapz(insertion_curves, dx=dx, axis=1).mean()
    )


def evaluate_symbolic_model(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_matrix: np.ndarray,
    teacher_labels: np.ndarray,
    class_names: tuple[str, ...],
    ranking: str = "tree",
    random_state: int = 42,
    compute_auc: bool = True,
) -> dict[str, Any]:
    """`ranking="random"` swaps the tree's own cell ranking for a random one,
    on the identical masking budget — the baseline sufficiency needs to show
    whether tumbling ~40/49 cells to zero is discriminative at all, or just
    collapses routing to the bias term regardless of which cells are picked.

    Deletion/insertion curves read the tree's routing-margin product at the
    predicted leaf, not a class probability — leaves hold a label, not a
    distribution. Interpret them against the `ranking="random"` control,
    never alone.
    """
    rng = np.random.default_rng(random_state)
    features = ensure_float32(feature_matrix)
    labels = np.asarray(teacher_labels, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("evaluate_symbolic_model expects a 2D feature matrix.")
    if tree.feature_shape is None:
        raise ValueError(
            "evaluate_symbolic_model requires tree.feature_shape to rank spatial cells."
        )

    N, D = features.shape
    _, grid_height, grid_width = tree.feature_shape
    cell_budget = max(
        int(grid_height * grid_width * _FAITHFULNESS_CELL_BUDGET_FRACTION), 1
    )

    # Setup
    predictions = np.empty(N, dtype=np.int64)
    necessity_prediction_flip = np.empty(N, dtype=np.float64)
    sufficiency_prediction_preservation = np.empty(N, dtype=np.float64)

    batch_size = 1024
    for start_idx in tqdm(
        range(0, N, batch_size),
        total=(N + batch_size - 1) // batch_size,
        desc="Symbolic metrics",
    ):
        end_idx = min(start_idx + batch_size, N)
        batch_features = features[start_idx:end_idx]
        B = end_idx - start_idx

        # Predict
        batch_preds = tree.predict(batch_features)
        predictions[start_idx:end_idx] = batch_preds

        # Mask the same cells Grad-CAM masks, so scores compare.
        selected_per_row: list[np.ndarray] = []
        for i in range(B):
            if ranking == "random":
                cell_ranking = rng.permutation(grid_height * grid_width)
            else:
                cell_ranking = _row_cell_ranking(tree, batch_features[i])
            top_cells = cell_ranking[:cell_budget]
            selected_per_row.append(
                _full_channel_flat_indices(top_cells, tree.feature_shape)
            )

        # Build masks
        selected_mask = _build_selected_mask(B, D, selected_per_row)

        necessity_features = batch_features.copy()
        necessity_features[selected_mask] = 0.0

        sufficiency_features = np.zeros_like(batch_features)
        sufficiency_features[selected_mask] = batch_features[selected_mask]

        # Score masked
        necessity_labels = tree.predict(necessity_features)
        sufficiency_labels = tree.predict(sufficiency_features)

        batch_nec_flip = (necessity_labels != batch_preds).astype(np.float64)
        necessity_prediction_flip[start_idx:end_idx] = batch_nec_flip

        batch_suf_pres = (sufficiency_labels == batch_preds).astype(np.float64)
        sufficiency_prediction_preservation[start_idx:end_idx] = batch_suf_pres

    # Aggregate
    if compute_auc:
        deletion_auc, insertion_auc = _deletion_insertion_auc(
            tree, features, predictions, random_state=42
        )
    else:
        deletion_auc = insertion_auc = float("nan")
    return {
        "mimic_accuracy": float((predictions == labels).mean()),
        "macro_f1_vs_teacher": _safe_macro_f1(
            labels, predictions, num_classes=len(class_names)
        ),
        "per_class_agreement_vs_teacher": _per_class_agreement(
            labels, predictions, class_names
        ),
        "necessity_prediction_flip_rate": float(necessity_prediction_flip.mean()),
        "sufficiency_prediction_preservation": float(
            sufficiency_prediction_preservation.mean()
        ),
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
    }


def _compute_local_instance_heatmap(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    mode: str = "all",
) -> Tensor:
    grid = _ensure_writable_tensor(feature_grid)
    feature_vector = grid.reshape(-1).detach().cpu().numpy().astype(np.float32)
    path = tree.decision_path(feature_vector)
    if not path:
        return torch.zeros(grid.shape[-2:], dtype=torch.float32)

    if mode == "leaf_only":
        path = [path[-1]]

    weight_grids = np.stack(
        [tree.node_weight_grid(step.node_index) for step in path], axis=0
    )
    path_directions = np.asarray(
        [1.0 if step.went_left else -1.0 for step in path],
        dtype=np.float32,
    ).reshape(-1, 1, 1, 1)
    local_signed = (
        path_directions * weight_grids * grid.detach().cpu().numpy()[None, ...]
    )
    positive_map = np.sum(np.maximum(local_signed, 0.0), axis=(0, 1)).astype(np.float32)
    return torch.from_numpy(positive_map)


def evaluate_symbolic_spatial_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grids: Tensor | np.ndarray,
    proposal_boxes: Tensor,
    matched_gt_boxes: Tensor | None,
    has_matched_gt: Tensor | None,
    gt_iou: Tensor | None,
    heatmap_mode: str = "leaf_only",
    min_proposal_iou: float = 0.0,
    max_proposal_iou: float = 1.0,
) -> dict[str, Any]:
    if matched_gt_boxes is None or has_matched_gt is None:
        return stratified_spatial_result([], [], [])

    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    gt_coverage: list[float] = []

    for index in tqdm(
        range(feature_grids.shape[0]),
        total=feature_grids.shape[0],
        desc="Spatial grounding",
    ):
        if not bool(has_matched_gt[index]):
            continue
        if gt_iou is not None and not (min_proposal_iou <= float(gt_iou[index]) <= max_proposal_iou):
            continue

        heatmap = _compute_local_instance_heatmap(
            tree, feature_grids[index], mode=heatmap_mode
        )
        normalized_heatmap = normalize_heatmap(heatmap)
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=tuple(normalized_heatmap.shape),
        )
        if gt_mask.sum().item() == 0:
            continue

        overlap_scores.append(topk_region_overlap(normalized_heatmap, gt_mask))
        pointing_scores.append(pointing_score(normalized_heatmap, gt_mask))
        gt_coverage.append(float(gt_mask.float().mean().item()))

    return stratified_spatial_result(overlap_scores, pointing_scores, gt_coverage)
