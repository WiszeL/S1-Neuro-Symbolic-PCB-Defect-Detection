from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .sodt import SparseObliqueDecisionTreeClassifier
from util.geometry import project_gt_box_to_roi_grid

_NAN = float("nan")

_SPATIAL_METRIC_NAMES = [
    "box_grounded_roi_overlap",
    "pointing_score",
    "energy_in_defect_ratio",
    "heatmap_entropy",
    "stability_score",
]


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


def path_feature_indices(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_vector: np.ndarray,
) -> np.ndarray:
    path = tree.decision_path(feature_vector)
    if not path:
        return np.zeros((0,), dtype=np.int64)

    active_indices = [
        np.flatnonzero(tree.node_weights[step.node_index]) for step in path
    ]
    non_empty = [indices for indices in active_indices if indices.size > 0]
    if not non_empty:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(non_empty, axis=0)).astype(np.int64)


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
    random_trials: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    features = np.asarray(feature_matrix, dtype=np.float32)
    labels = np.asarray(teacher_labels, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("evaluate_symbolic_model expects a 2D feature matrix.")

    N, D = features.shape
    R = max(int(random_trials), 1)
    rng = np.random.default_rng(random_state)

    # Pre-allocate output arrays for N samples
    predictions = np.empty(N, dtype=np.int64)
    subset_sizes = np.empty(N, dtype=np.int64)
    necessity_confidence_drop = np.empty(N, dtype=np.float64)
    necessity_prediction_flip = np.empty(N, dtype=np.float64)
    sufficiency_prediction_preservation = np.empty(N, dtype=np.float64)
    sufficiency_confidence_retention = np.empty(N, dtype=np.float64)
    rnd_nec_drop_per_row = np.empty(N, dtype=np.float64)
    rnd_nec_flip_per_row = np.empty(N, dtype=np.float64)
    rnd_suf_pres_per_row = np.empty(N, dtype=np.float64)
    rnd_suf_ret_per_row = np.empty(N, dtype=np.float64)

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
            sel = path_feature_indices(tree, batch_features[i])
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

        # --- random controls ---
        active_mask = batch_subset_sizes > 0
        inactive_mask = ~active_mask
        active_indices = np.where(active_mask)[0]

        b_rnd_nec_drop = np.zeros(B, dtype=np.float64)
        b_rnd_nec_flip = np.zeros(B, dtype=np.float64)
        b_rnd_suf_pres = np.copy(batch_suf_pres)
        b_rnd_suf_ret = np.copy(batch_suf_ret)

        if active_indices.size > 0:
            M = active_indices.size * R
            rnd_nec_features = np.empty((M, D), dtype=np.float32)
            rnd_suf_features = np.empty((M, D), dtype=np.float32)
            trial_row_map = np.empty(M, dtype=np.int64)

            k = 0
            for i in active_indices:
                ss = int(batch_subset_sizes[i])
                for _ in range(R):
                    random_idx = np.sort(
                        rng.choice(D, size=ss, replace=False).astype(np.int64)
                    )
                    rnd_nec_features[k] = batch_features[i]
                    rnd_nec_features[k, random_idx] = 0.0
                    rnd_suf_features[k] = 0.0
                    rnd_suf_features[k, random_idx] = batch_features[i, random_idx]
                    trial_row_map[k] = i
                    k += 1

            rnd_nec_probs = tree.predict_proba(rnd_nec_features)
            rnd_suf_probs = tree.predict_proba(rnd_suf_features)

            rnd_nec_labels = rnd_nec_probs.argmax(axis=1).astype(np.int64)
            rnd_nec_conf = rnd_nec_probs[
                np.arange(M, dtype=np.int64), batch_preds[trial_row_map]
            ]
            rnd_nec_drop = np.maximum(
                batch_conf[trial_row_map] - rnd_nec_conf, 0.0
            ).astype(np.float64)
            rnd_nec_flip = (rnd_nec_labels != batch_preds[trial_row_map]).astype(
                np.float64
            )

            rnd_suf_labels = rnd_suf_probs.argmax(axis=1).astype(np.int64)
            rnd_suf_conf = rnd_suf_probs[
                np.arange(M, dtype=np.int64), batch_preds[trial_row_map]
            ]
            rnd_suf_pres = (rnd_suf_labels == batch_preds[trial_row_map]).astype(
                np.float64
            )
            rnd_suf_ret = (
                rnd_suf_conf / np.maximum(batch_conf[trial_row_map], 1e-8)
            ).astype(np.float64)

            trial_counts = np.bincount(trial_row_map, minlength=B).astype(np.float64)
            safe_counts = np.maximum(trial_counts, 1.0)

            b_rnd_nec_drop = (
                np.bincount(trial_row_map, weights=rnd_nec_drop, minlength=B)
                / safe_counts
            )
            b_rnd_nec_flip = (
                np.bincount(trial_row_map, weights=rnd_nec_flip, minlength=B)
                / safe_counts
            )
            b_rnd_suf_pres = (
                np.bincount(trial_row_map, weights=rnd_suf_pres, minlength=B)
                / safe_counts
            )
            b_rnd_suf_ret = (
                np.bincount(trial_row_map, weights=rnd_suf_ret, minlength=B)
                / safe_counts
            )

            b_rnd_nec_drop[inactive_mask] = 0.0
            b_rnd_nec_flip[inactive_mask] = 0.0
            b_rnd_suf_pres[inactive_mask] = batch_suf_pres[inactive_mask]
            b_rnd_suf_ret[inactive_mask] = batch_suf_ret[inactive_mask]

        rnd_nec_drop_per_row[start_idx:end_idx] = b_rnd_nec_drop
        rnd_nec_flip_per_row[start_idx:end_idx] = b_rnd_nec_flip
        rnd_suf_pres_per_row[start_idx:end_idx] = b_rnd_suf_pres
        rnd_suf_ret_per_row[start_idx:end_idx] = b_rnd_suf_ret

    # --- aggregate ---
    mimic_accuracy = float((predictions == labels).mean())
    macro_f1 = _safe_macro_f1(labels, predictions, num_classes=len(class_names))

    m_nec_drop = float(necessity_confidence_drop.mean())
    m_nec_flip = float(necessity_prediction_flip.mean())
    m_suf_pres = float(sufficiency_prediction_preservation.mean())
    m_suf_ret = float(sufficiency_confidence_retention.mean())
    m_rnd_nec_drop = float(rnd_nec_drop_per_row.mean())
    m_rnd_nec_flip = float(rnd_nec_flip_per_row.mean())
    m_rnd_suf_pres = float(rnd_suf_pres_per_row.mean())
    m_rnd_suf_ret = float(rnd_suf_ret_per_row.mean())

    return {
        "mimic_accuracy": mimic_accuracy,
        "macro_f1_vs_teacher": macro_f1,
        "per_class_agreement_vs_teacher": _per_class_agreement(
            labels, predictions, class_names
        ),
        "mean_path_feature_count": float(np.mean(subset_sizes)),
        "median_path_feature_count": float(np.median(subset_sizes)),
        "necessity_confidence_drop": m_nec_drop,
        "necessity_prediction_flip_rate": m_nec_flip,
        "sufficiency_prediction_preservation": m_suf_pres,
        "sufficiency_confidence_retention": m_suf_ret,
        "random_control_necessity_confidence_drop": m_rnd_nec_drop,
        "random_control_necessity_prediction_flip_rate": m_rnd_nec_flip,
        "random_control_sufficiency_prediction_preservation": m_rnd_suf_pres,
        "random_control_sufficiency_confidence_retention": m_rnd_suf_ret,
        "necessity_advantage_over_random": m_nec_drop - m_rnd_nec_drop,
        "necessity_flip_advantage_over_random": m_nec_flip - m_rnd_nec_flip,
        "sufficiency_advantage_over_random": m_suf_pres - m_rnd_suf_pres,
        "sufficiency_retention_advantage_over_random": m_suf_ret - m_rnd_suf_ret,
        "evaluation_extension": {
            "faithfulness_tests": [
                "necessity",
                "sufficiency",
                "random_control",
            ],
            "note": (
                "These necessity/sufficiency/random-control metrics are thesis-specific evaluation "
                "extensions. They are not claimed as direct procedures from Hada or Kairgeldin."
            ),
        },
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
    target_cells = max(int(gt_mask.sum().item()), 1)
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
    feature_std = float(feature_grid.std(unbiased=False).item())
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

    base_vector = torch.clamp(base_heatmap.detach().cpu().float(), min=0.0)
    base_total = float(base_vector.sum().item())
    base_vector = (
        torch.zeros_like(base_vector) if base_total <= 0.0 else base_vector / base_total
    ).reshape(-1)
    perturbed_vector = torch.clamp(perturbed_heatmap.detach().cpu().float(), min=0.0)
    perturbed_total = float(perturbed_vector.sum().item())
    if base_total <= 0.0 and perturbed_total <= 0.0:
        return 1.0
    perturbed_vector = (
        torch.zeros_like(perturbed_vector)
        if perturbed_total <= 0.0
        else perturbed_vector / perturbed_total
    ).reshape(-1)

    denominator = float(base_vector.norm().item() * perturbed_vector.norm().item())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(base_vector, perturbed_vector).item() / denominator)


def _spatial_result(
    count: int,
    overlap: float,
    pointing: float,
    energy: float,
    entropy: float,
    stability: float,
    metric_names: list[str],
    note: str,
) -> dict[str, Any]:
    return {
        "evaluated_roi_count": count,
        "box_grounded_roi_overlap": overlap,
        "pointing_score": pointing,
        "energy_in_defect_ratio": energy,
        "heatmap_entropy": entropy,
        "stability_score": stability,
        "evaluation_extension": {
            "spatial_metrics": metric_names,
            "note": note,
        },
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
            [],
            "No matched GT boxes were available. Spatial explanation metrics could not be computed.",
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

        heatmap = torch.clamp(
            _compute_local_instance_heatmap(tree, feature_grids[index])
            .detach()
            .cpu()
            .float(),
            min=0.0,
        )
        total = float(heatmap.sum().item())
        heatmap = torch.zeros_like(heatmap) if total <= 0.0 else heatmap / total
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=tuple(int(value) for value in heatmap.shape),
        )
        if int(gt_mask.sum().item()) == 0:
            continue

        overlap_scores.append(_topk_region_overlap(heatmap, gt_mask))
        peak_index = int(torch.argmax(heatmap.reshape(-1)).item())
        row_index = peak_index // heatmap.shape[1]
        col_index = peak_index % heatmap.shape[1]
        pointing_scores.append(float(gt_mask[row_index, col_index].item()))
        energy_scores.append(float(heatmap[gt_mask].sum().item()))
        flattened = heatmap.reshape(-1)
        positive = flattened[flattened > 0]
        entropy_scores.append(
            0.0
            if positive.numel() == 0
            else float(
                (-torch.sum(positive * torch.log(positive))).item()
                / np.log(max(flattened.numel(), 2))
            )
        )
        stability_scores.append(
            _feature_perturbation_stability(
                tree,
                feature_grid=feature_grids[index],
                base_heatmap=heatmap,
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
            _SPATIAL_METRIC_NAMES,
            "Spatial metrics are box-grounded on the RoI grid because the current symbolic export contains GT "
            "boxes, not pixel masks. They should not be overclaimed as pixel-level faithfulness.",
        )

    return _spatial_result(
        int(len(overlap_scores)),
        float(np.mean(overlap_scores)),
        float(np.mean(pointing_scores)),
        float(np.mean(energy_scores)),
        float(np.mean(entropy_scores)),
        float(np.mean(stability_scores)),
        _SPATIAL_METRIC_NAMES,
        "These spatial metrics are thesis-specific evaluation extensions. They are box-grounded on the RoI "
        "grid when only GT boxes are available, and they should not be interpreted as pixel-level ground truth.",
    )
