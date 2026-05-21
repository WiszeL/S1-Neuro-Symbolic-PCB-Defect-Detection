from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .sodt import SparseObliqueDecisionTreeClassifier
from util.geometry import project_gt_box_to_roi_grid

_NAN = float("nan")




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


def evaluate_symbolic_model(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_matrix: np.ndarray,
    teacher_labels: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    features = feature_matrix if feature_matrix.dtype == np.float32 else np.asarray(feature_matrix, dtype=np.float32)
    labels = np.asarray(teacher_labels, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("evaluate_symbolic_model expects a 2D feature matrix.")

    N, D = features.shape

    # Pre-allocate output arrays for N samples
    predictions = np.empty(N, dtype=np.int64)
    subset_sizes = np.empty(N, dtype=np.int64)
    necessity_confidence_drop = np.empty(N, dtype=np.float64)
    necessity_prediction_flip = np.empty(N, dtype=np.float64)
    sufficiency_prediction_preservation = np.empty(N, dtype=np.float64)
    sufficiency_confidence_retention = np.empty(N, dtype=np.float64)

    batch_size = 1024
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_features = features[start_idx:end_idx]
        B = end_idx - start_idx
        batch_row_range = np.arange(B, dtype=np.int64)

        # --- batch predict on original features ---
        probabilities = tree.predict_proba(batch_features)
        batch_preds = probabilities.argmax(axis=1).astype(np.int64)
        batch_conf = probabilities[batch_row_range, batch_preds]
        predictions[start_idx:end_idx] = batch_preds

        # --- collect path feature indices per row ---
        selected_per_row: list[np.ndarray] = []
        batch_subset_sizes = np.empty(B, dtype=np.int64)
        for i in range(B):
            sel = tree.path_feature_indices(batch_features[i])
            selected_per_row.append(sel)
            batch_subset_sizes[i] = sel.size

        subset_sizes[start_idx:end_idx] = batch_subset_sizes

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
    return {
        "mimic_accuracy": float((predictions == labels).mean()),
        "macro_f1_vs_teacher": _safe_macro_f1(labels, predictions, num_classes=len(class_names)),
        "per_class_agreement_vs_teacher": _per_class_agreement(
            labels, predictions, class_names
        ),
        "mean_path_feature_count": float(np.mean(subset_sizes)),
        "median_path_feature_count": float(np.median(subset_sizes)),
        "necessity_confidence_drop": float(necessity_confidence_drop.mean()),
        "necessity_prediction_flip_rate": float(necessity_prediction_flip.mean()),
        "sufficiency_prediction_preservation": float(sufficiency_prediction_preservation.mean()),
        "sufficiency_confidence_retention": float(sufficiency_confidence_retention.mean()),
    }


def _compute_local_instance_heatmap(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
) -> Tensor:
    grid = _ensure_writable_tensor(feature_grid)
    feature_vector = grid.reshape(-1).detach().cpu().numpy().astype(np.float32)
    path = tree.decision_path(feature_vector)
    if not path:
        return torch.zeros(grid.shape[-2:], dtype=torch.float32)

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


def _topk_region_overlap(heatmap: Tensor, gt_mask: Tensor) -> float:
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


def _normalize_heatmap(heatmap: Tensor | np.ndarray) -> Tensor:
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    tensor = torch.clamp(tensor, min=0.0)
    total = tensor.sum().item()
    if total <= 0.0:
        return torch.zeros_like(tensor)
    return tensor / total


def _heatmap_entropy(heatmap: Tensor) -> float:
    flattened = heatmap.reshape(-1)
    positive = flattened[flattened > 0]
    if positive.numel() == 0:
        return 0.0
    entropy = -torch.sum(positive * torch.log(positive))
    return float(entropy.item() / np.log(max(flattened.numel(), 2)))


def _pointing_score(heatmap: Tensor, gt_mask: Tensor) -> float:
    peak_index = torch.argmax(heatmap.reshape(-1)).item()
    row_index = peak_index // heatmap.shape[1]
    col_index = peak_index % heatmap.shape[1]
    return float(gt_mask[row_index, col_index].item())


def _energy_in_region(heatmap: Tensor, gt_mask: Tensor) -> float:
    if gt_mask.sum().item() == 0:
        return 0.0
    return heatmap[gt_mask].sum().item()


def _feature_perturbation_stability(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    base_heatmap: Tensor,
    random_state: int,
    noise_scale: float = 0.02,
) -> float:
    generator = torch.Generator()
    generator.manual_seed(random_state)
    feature_grid = _ensure_writable_tensor(feature_grid).detach().cpu()
    feature_std = feature_grid.std(unbiased=False).item()
    if feature_std <= 0.0:
        return 1.0

    perturbation = torch.randn(
        feature_grid.shape,
        generator=generator,
        dtype=feature_grid.dtype,
    ) * (noise_scale * feature_std)
    perturbed_heatmap = _compute_local_instance_heatmap(
        tree, feature_grid + perturbation
    )

    base_vector = _normalize_heatmap(base_heatmap).reshape(-1)
    perturbed_vector = _normalize_heatmap(perturbed_heatmap).reshape(-1)
    if (
        base_vector.sum().item() == 0.0
        and perturbed_vector.sum().item() == 0.0
    ):
        return 1.0

    denominator = base_vector.norm().item() * perturbed_vector.norm().item()
    if denominator <= 0.0:
        return 0.0
    return torch.dot(base_vector, perturbed_vector).item() / denominator


def _spatial_result(
    count: int,
    overlap: float,
    pointing: float,
    energy: float,
    entropy: float,
    stability: float,
) -> dict[str, Any]:
    return {
        "evaluated_roi_count": count,
        "box_grounded_roi_overlap": overlap,
        "pointing_score": pointing,
        "energy_in_defect_ratio": energy,
        "heatmap_entropy": entropy,
        "stability_score": stability,
    }


def evaluate_symbolic_spatial_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grids: Tensor | np.ndarray,
    proposal_boxes: Tensor,
    matched_gt_boxes: Tensor | None,
    has_matched_gt: Tensor | None,
    gt_iou: Tensor | None,
    random_state: int = 42,
) -> dict[str, Any]:
    if matched_gt_boxes is None or has_matched_gt is None:
        return _spatial_result(
            0,
            _NAN,
            _NAN,
            _NAN,
            _NAN,
            _NAN,
        )

    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    energy_scores: list[float] = []
    entropy_scores: list[float] = []
    stability_scores: list[float] = []

    for index in range(feature_grids.shape[0]):
        if not bool(has_matched_gt[index]):
            continue
        if gt_iou is not None and float(gt_iou[index]) <= 0.0:
            continue

        heatmap = _compute_local_instance_heatmap(tree, feature_grids[index])
        normalized_heatmap = _normalize_heatmap(heatmap)
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=tuple(normalized_heatmap.shape),
        )
        if gt_mask.sum().item() == 0:
            continue

        overlap_scores.append(_topk_region_overlap(normalized_heatmap, gt_mask))
        pointing_scores.append(_pointing_score(normalized_heatmap, gt_mask))
        energy_scores.append(_energy_in_region(normalized_heatmap, gt_mask))
        entropy_scores.append(_heatmap_entropy(normalized_heatmap))
        stability_scores.append(
            _feature_perturbation_stability(
                tree,
                feature_grid=feature_grids[index],
                base_heatmap=normalized_heatmap,
                random_state=random_state + index,
            )
        )

    if not overlap_scores:
        return _spatial_result(
            0,
            _NAN,
            _NAN,
            _NAN,
            _NAN,
            _NAN,
        )

    return _spatial_result(
        len(overlap_scores),
        float(np.mean(overlap_scores)),
        float(np.mean(pointing_scores)),
        float(np.mean(energy_scores)),
        float(np.mean(entropy_scores)),
        float(np.mean(stability_scores)),
    )
