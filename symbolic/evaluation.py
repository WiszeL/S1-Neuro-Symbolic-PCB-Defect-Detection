"""Symbolic (SODT) explanation-quality evaluation.

Faithfulness perturbation protocol — shared with gradcam/evaluation.py so the
two sides are directly comparable:

  * unit:    one spatial cell of the pooled RoI-Align grid, ALL channels
             zeroed/kept together (not a raw per-feature mask)
  * ranking: each method's own heatmap over every cell in the grid
  * budget:  identical cell counts and step schedule on both sides
             (top-50% for sufficiency/necessity; k = s/steps * num_cells
             for deletion/insertion)
  * probe:   each explainer's own model — the SODT probes `tree.predict_proba`,
             Grad-CAM probes the FRCNN box head. This is a self-consistency
             check ("is the explanation faithful to what THIS model did"),
             not a cross-model comparison.

Note what this does NOT test: the SODT's decision is exactly reconstructible
from its path (every node on the path has nonzero weight only inside its own
feature set), so masking by "the tree's own active path features" would be
tautologically sufficient/necessary by construction — that structural
property is real but belongs in the thesis as a proposition, not a measured
number next to Grad-CAM's. Masking by spatial cell — the same interface both
the SODT and the FRCNN classifier head consume — avoids that trivially-true
result and asks the intended question instead: is the *heatmap* (a lossy
spatial rendering of the tree's decision) a faithful proxy for it.
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
    """Convert data to tensor, copying numpy arrays to avoid write warnings."""
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
    """Spatial cells of the pooled grid (flattened H*W index), ranked by
    descending importance under the tree's own leaf-node heatmap — the same
    reduction the displayed explanations use (neurosym/heatmap.py's default
    "leaf_only" mode)."""
    grid = feature_row.reshape(tree.feature_shape)
    heatmap = _compute_local_instance_heatmap(tree, grid, mode="leaf_only")
    return importance_ranking(heatmap).cpu().numpy()


def _full_channel_flat_indices(
    cells: np.ndarray, feature_shape: tuple[int, int, int]
) -> np.ndarray:
    """Flat feature indices covering ALL channels at the given spatial
    cells — the perturbation unit (one grid cell, every channel), matching
    what gradcam/evaluation.py masks on the pooled features."""
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
        full_probs = tree.predict_proba(auc_features[batch_start:batch_end])
        empty_probs = tree.predict_proba(np.zeros((B, D), dtype=np.float32))
        for bi in range(B):
            gi = batch_start + bi
            deletion_curves[gi, 0] = full_probs[bi, auc_preds[gi]]
            insertion_curves[gi, 0] = empty_probs[bi, auc_preds[gi]]

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

            del_probs = tree.predict_proba(del_batch)
            ins_probs = tree.predict_proba(ins_batch)
            for bi in range(B):
                gi = batch_start + bi
                deletion_curves[gi, s] = del_probs[bi, auc_preds[gi]]
                insertion_curves[gi, s] = ins_probs[bi, auc_preds[gi]]

    dx = 1.0 / steps
    return float(np.trapz(deletion_curves, dx=dx, axis=1).mean()), float(
        np.trapz(insertion_curves, dx=dx, axis=1).mean()
    )


def evaluate_symbolic_model(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_matrix: np.ndarray,
    teacher_labels: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
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

    # Pre-allocate output arrays for N samples
    predictions = np.empty(N, dtype=np.int64)
    necessity_confidence_drop = np.empty(N, dtype=np.float64)
    necessity_prediction_flip = np.empty(N, dtype=np.float64)
    sufficiency_prediction_preservation = np.empty(N, dtype=np.float64)
    sufficiency_confidence_retention = np.empty(N, dtype=np.float64)

    batch_size = 1024
    for start_idx in tqdm(
        range(0, N, batch_size),
        total=(N + batch_size - 1) // batch_size,
        desc="Symbolic metrics",
    ):
        end_idx = min(start_idx + batch_size, N)
        batch_features = features[start_idx:end_idx]
        B = end_idx - start_idx
        batch_row_range = np.arange(B, dtype=np.int64)

        # --- batch predict on original features ---
        probabilities = tree.predict_proba(batch_features)
        batch_preds = probabilities.argmax(axis=1).astype(np.int64)
        batch_conf = probabilities[batch_row_range, batch_preds]
        predictions[start_idx:end_idx] = batch_preds

        # --- rank spatial cells by the tree's own heatmap (same protocol
        # as gradcam/evaluation.py), then mask the top cell_budget cells
        # across ALL channels — see the module docstring above ---
        selected_per_row: list[np.ndarray] = []
        for i in range(B):
            ranking = _row_cell_ranking(tree, batch_features[i])
            top_cells = ranking[:cell_budget]
            selected_per_row.append(
                _full_channel_flat_indices(top_cells, tree.feature_shape)
            )

        # --- build boolean mask and masked feature matrices ---
        selected_mask = _build_selected_mask(B, D, selected_per_row)

        necessity_features = batch_features.copy()
        necessity_features[selected_mask] = 0.0

        sufficiency_features = np.zeros_like(batch_features)
        sufficiency_features[selected_mask] = batch_features[selected_mask]

        # --- batch predict necessity / sufficiency ---
        necessity_probs = tree.predict_proba(necessity_features)
        sufficiency_probs = tree.predict_proba(sufficiency_features)

        necessity_labels = necessity_probs.argmax(axis=1).astype(np.int64)
        necessity_confidences = necessity_probs[batch_row_range, batch_preds]
        batch_nec_drop = np.maximum(batch_conf - necessity_confidences, 0.0).astype(
            np.float64
        )
        batch_nec_flip = (necessity_labels != batch_preds).astype(np.float64)

        necessity_confidence_drop[start_idx:end_idx] = batch_nec_drop
        necessity_prediction_flip[start_idx:end_idx] = batch_nec_flip

        sufficiency_labels = sufficiency_probs.argmax(axis=1).astype(np.int64)
        sufficiency_confidences = sufficiency_probs[batch_row_range, batch_preds]
        batch_suf_pres = (sufficiency_labels == batch_preds).astype(np.float64)
        batch_suf_ret = (sufficiency_confidences / np.maximum(batch_conf, 1e-8)).astype(
            np.float64
        )

        sufficiency_prediction_preservation[start_idx:end_idx] = batch_suf_pres
        sufficiency_confidence_retention[start_idx:end_idx] = batch_suf_ret

    # --- aggregate ---
    deletion_auc, insertion_auc = _deletion_insertion_auc(
        tree, features, predictions, random_state=42
    )
    return {
        "mimic_accuracy": float((predictions == labels).mean()),
        "macro_f1_vs_teacher": _safe_macro_f1(
            labels, predictions, num_classes=len(class_names)
        ),
        "per_class_agreement_vs_teacher": _per_class_agreement(
            labels, predictions, class_names
        ),
        "necessity_confidence_drop": float(necessity_confidence_drop.mean()),
        "necessity_prediction_flip_rate": float(necessity_prediction_flip.mean()),
        "sufficiency_prediction_preservation": float(
            sufficiency_prediction_preservation.mean()
        ),
        "sufficiency_confidence_retention": float(
            sufficiency_confidence_retention.mean()
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
        if gt_iou is not None and float(gt_iou[index]) < min_proposal_iou:
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
