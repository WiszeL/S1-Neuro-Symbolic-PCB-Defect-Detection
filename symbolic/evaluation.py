from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .sodt import SparseObliqueDecisionTreeClassifier


def _safe_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> float:
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for symbolic candidate evaluation metrics."
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


def path_feature_indices(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_vector: np.ndarray,
) -> np.ndarray:
    path = tree.decision_path(feature_vector)
    if not path:
        return np.zeros((0,), dtype=np.int64)

    active_indices = [
        np.flatnonzero(tree.node_weights[step.node_index])
        for step in path
    ]
    non_empty = [indices for indices in active_indices if indices.size > 0]
    if not non_empty:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(non_empty, axis=0)).astype(np.int64)


def _prediction_confidence(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    indices = np.arange(probabilities.shape[0], dtype=np.int64)
    return probabilities[indices, labels]


def _masked_feature_vector(
    feature_vector: np.ndarray,
    selected_indices: np.ndarray,
    keep_selected: bool,
) -> np.ndarray:
    masked = np.zeros_like(feature_vector, dtype=np.float32) if keep_selected else feature_vector.copy()
    if selected_indices.size == 0:
        return masked
    if keep_selected:
        masked[selected_indices] = feature_vector[selected_indices]
    else:
        masked[selected_indices] = 0.0
    return masked


def evaluate_symbolic_candidate(
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
        raise ValueError("evaluate_symbolic_candidate expects a 2D feature matrix.")

    probabilities = tree.predict_proba(features)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    predicted_confidences = _prediction_confidence(probabilities, predictions)
    generator = np.random.default_rng(random_state)

    necessity_confidence_drop: list[float] = []
    necessity_prediction_flip: list[float] = []
    sufficiency_prediction_preservation: list[float] = []
    sufficiency_confidence_retention: list[float] = []
    random_necessity_confidence_drop: list[float] = []
    random_necessity_prediction_flip: list[float] = []
    random_sufficiency_prediction_preservation: list[float] = []
    random_sufficiency_confidence_retention: list[float] = []
    path_feature_counts: list[int] = []

    for row_index, feature_vector in enumerate(features):
        selected_indices = path_feature_indices(tree, feature_vector)
        subset_size = int(selected_indices.size)
        path_feature_counts.append(subset_size)

        original_label = int(predictions[row_index])
        original_confidence = float(predicted_confidences[row_index])

        necessity_feature = _masked_feature_vector(feature_vector, selected_indices, keep_selected=False)
        necessity_probability = tree.predict_proba(necessity_feature[None, :])[0]
        necessity_label = int(necessity_probability.argmax())
        necessity_confidence = float(necessity_probability[original_label])
        necessity_confidence_drop.append(max(original_confidence - necessity_confidence, 0.0))
        necessity_prediction_flip.append(float(necessity_label != original_label))

        sufficiency_feature = _masked_feature_vector(feature_vector, selected_indices, keep_selected=True)
        sufficiency_probability = tree.predict_proba(sufficiency_feature[None, :])[0]
        sufficiency_label = int(sufficiency_probability.argmax())
        sufficiency_confidence = float(sufficiency_probability[original_label])
        sufficiency_prediction_preservation.append(float(sufficiency_label == original_label))
        sufficiency_confidence_retention.append(
            float(sufficiency_confidence / max(original_confidence, 1e-8))
        )

        if subset_size == 0:
            random_necessity_confidence_drop.append(0.0)
            random_necessity_prediction_flip.append(0.0)
            random_sufficiency_prediction_preservation.append(float(sufficiency_label == original_label))
            random_sufficiency_confidence_retention.append(
                float(sufficiency_confidence / max(original_confidence, 1e-8))
            )
            continue

        trial_necessity_drop: list[float] = []
        trial_necessity_flip: list[float] = []
        trial_sufficiency_preservation: list[float] = []
        trial_sufficiency_retention: list[float] = []
        for _ in range(max(int(random_trials), 1)):
            random_indices = np.sort(
                generator.choice(tree.input_dim, size=subset_size, replace=False).astype(np.int64)
            )

            random_necessity_feature = _masked_feature_vector(
                feature_vector,
                random_indices,
                keep_selected=False,
            )
            random_necessity_probability = tree.predict_proba(random_necessity_feature[None, :])[0]
            random_necessity_label = int(random_necessity_probability.argmax())
            trial_necessity_drop.append(
                max(original_confidence - float(random_necessity_probability[original_label]), 0.0)
            )
            trial_necessity_flip.append(float(random_necessity_label != original_label))

            random_sufficiency_feature = _masked_feature_vector(
                feature_vector,
                random_indices,
                keep_selected=True,
            )
            random_sufficiency_probability = tree.predict_proba(random_sufficiency_feature[None, :])[0]
            random_sufficiency_label = int(random_sufficiency_probability.argmax())
            trial_sufficiency_preservation.append(float(random_sufficiency_label == original_label))
            trial_sufficiency_retention.append(
                float(random_sufficiency_probability[original_label] / max(original_confidence, 1e-8))
            )

        random_necessity_confidence_drop.append(float(np.mean(trial_necessity_drop)))
        random_necessity_prediction_flip.append(float(np.mean(trial_necessity_flip)))
        random_sufficiency_prediction_preservation.append(float(np.mean(trial_sufficiency_preservation)))
        random_sufficiency_confidence_retention.append(float(np.mean(trial_sufficiency_retention)))

    mimic_accuracy = float((predictions == labels).mean())
    macro_f1 = _safe_macro_f1(
        labels,
        predictions,
        num_classes=len(class_names),
    )
    return {
        "mimic_accuracy": mimic_accuracy,
        "macro_f1_vs_teacher": macro_f1,
        "per_class_agreement_vs_teacher": _per_class_agreement(labels, predictions, class_names),
        "mean_path_feature_count": float(np.mean(path_feature_counts)) if path_feature_counts else 0.0,
        "median_path_feature_count": float(np.median(path_feature_counts)) if path_feature_counts else 0.0,
        "necessity_confidence_drop": float(np.mean(necessity_confidence_drop)) if necessity_confidence_drop else 0.0,
        "necessity_prediction_flip_rate": float(np.mean(necessity_prediction_flip)) if necessity_prediction_flip else 0.0,
        "sufficiency_prediction_preservation": (
            float(np.mean(sufficiency_prediction_preservation)) if sufficiency_prediction_preservation else 0.0
        ),
        "sufficiency_confidence_retention": (
            float(np.mean(sufficiency_confidence_retention)) if sufficiency_confidence_retention else 0.0
        ),
        "random_control_necessity_confidence_drop": (
            float(np.mean(random_necessity_confidence_drop)) if random_necessity_confidence_drop else 0.0
        ),
        "random_control_necessity_prediction_flip_rate": (
            float(np.mean(random_necessity_prediction_flip)) if random_necessity_prediction_flip else 0.0
        ),
        "random_control_sufficiency_prediction_preservation": (
            float(np.mean(random_sufficiency_prediction_preservation))
            if random_sufficiency_prediction_preservation
            else 0.0
        ),
        "random_control_sufficiency_confidence_retention": (
            float(np.mean(random_sufficiency_confidence_retention))
            if random_sufficiency_confidence_retention
            else 0.0
        ),
        "necessity_advantage_over_random": (
            float(np.mean(necessity_confidence_drop) - np.mean(random_necessity_confidence_drop))
            if necessity_confidence_drop
            else 0.0
        ),
        "necessity_flip_advantage_over_random": (
            float(np.mean(necessity_prediction_flip) - np.mean(random_necessity_prediction_flip))
            if necessity_prediction_flip
            else 0.0
        ),
        "sufficiency_advantage_over_random": (
            float(np.mean(sufficiency_prediction_preservation) - np.mean(random_sufficiency_prediction_preservation))
            if sufficiency_prediction_preservation
            else 0.0
        ),
        "sufficiency_retention_advantage_over_random": (
            float(np.mean(sufficiency_confidence_retention) - np.mean(random_sufficiency_confidence_retention))
            if sufficiency_confidence_retention
            else 0.0
        ),
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


def project_gt_box_to_roi_grid(
    proposal_box: Tensor | np.ndarray,
    matched_gt_box: Tensor | np.ndarray,
    grid_shape: tuple[int, int],
) -> Tensor:
    proposal = torch.as_tensor(proposal_box, dtype=torch.float32)
    gt_box = torch.as_tensor(matched_gt_box, dtype=torch.float32)
    grid_height, grid_width = int(grid_shape[0]), int(grid_shape[1])

    proposal_width = max(float(proposal[2] - proposal[0]), 1e-6)
    proposal_height = max(float(proposal[3] - proposal[1]), 1e-6)

    projected_x1 = ((gt_box[0] - proposal[0]) / proposal_width) * grid_width
    projected_y1 = ((gt_box[1] - proposal[1]) / proposal_height) * grid_height
    projected_x2 = ((gt_box[2] - proposal[0]) / proposal_width) * grid_width
    projected_y2 = ((gt_box[3] - proposal[1]) / proposal_height) * grid_height

    projected_x1 = float(torch.clamp(projected_x1, min=0.0, max=float(grid_width)))
    projected_y1 = float(torch.clamp(projected_y1, min=0.0, max=float(grid_height)))
    projected_x2 = float(torch.clamp(projected_x2, min=0.0, max=float(grid_width)))
    projected_y2 = float(torch.clamp(projected_y2, min=0.0, max=float(grid_height)))

    mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
    if projected_x2 <= projected_x1 or projected_y2 <= projected_y1:
        return mask

    for row_index in range(grid_height):
        for col_index in range(grid_width):
            cell_x1 = float(col_index)
            cell_y1 = float(row_index)
            cell_x2 = float(col_index + 1)
            cell_y2 = float(row_index + 1)
            intersection_width = min(cell_x2, projected_x2) - max(cell_x1, projected_x1)
            intersection_height = min(cell_y2, projected_y2) - max(cell_y1, projected_y1)
            if intersection_width > 0.0 and intersection_height > 0.0:
                mask[row_index, col_index] = True

    return mask


def _compute_local_instance_heatmap(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
) -> Tensor:
    grid = torch.as_tensor(feature_grid, dtype=torch.float32)
    feature_vector = grid.reshape(-1).detach().cpu().numpy().astype(np.float32)
    path = tree.decision_path(feature_vector)
    if not path:
        return torch.zeros(grid.shape[-2:], dtype=torch.float32)

    weight_grids = np.stack([tree.node_weight_grid(step.node_index) for step in path], axis=0)
    path_directions = np.asarray(
        [1.0 if step.went_left else -1.0 for step in path],
        dtype=np.float32,
    ).reshape(-1, 1, 1, 1)
    local_signed = path_directions * weight_grids * grid.detach().cpu().numpy()[None, ...]
    positive_map = np.sum(np.maximum(local_signed, 0.0), axis=(0, 1)).astype(np.float32)
    return torch.from_numpy(positive_map)


def _normalize_heatmap(heatmap: Tensor) -> Tensor:
    tensor = torch.clamp(heatmap.detach().cpu().float(), min=0.0)
    total = float(tensor.sum().item())
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


def _pointing_score(heatmap: Tensor, gt_mask: Tensor) -> float:
    peak_index = int(torch.argmax(heatmap.reshape(-1)).item())
    row_index = peak_index // heatmap.shape[1]
    col_index = peak_index % heatmap.shape[1]
    return float(gt_mask[row_index, col_index].item())


def _energy_in_region(heatmap: Tensor, gt_mask: Tensor) -> float:
    if int(gt_mask.sum().item()) == 0:
        return 0.0
    return float(heatmap[gt_mask].sum().item())


def _feature_perturbation_stability(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor,
    base_heatmap: Tensor,
    random_state: int,
    noise_scale: float = 0.02,
) -> float:
    generator = torch.Generator()
    generator.manual_seed(random_state)
    feature_grid = feature_grid.detach().cpu().float()
    feature_std = float(feature_grid.std(unbiased=False).item())
    if feature_std <= 0.0:
        return 1.0

    perturbation = torch.randn(
        feature_grid.shape,
        generator=generator,
        dtype=feature_grid.dtype,
    ) * (noise_scale * feature_std)
    perturbed_heatmap = _compute_local_instance_heatmap(tree, feature_grid + perturbation)

    base_vector = _normalize_heatmap(base_heatmap).reshape(-1)
    perturbed_vector = _normalize_heatmap(perturbed_heatmap).reshape(-1)
    if float(base_vector.sum().item()) == 0.0 and float(perturbed_vector.sum().item()) == 0.0:
        return 1.0

    denominator = float(base_vector.norm().item() * perturbed_vector.norm().item())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(base_vector, perturbed_vector).item() / denominator)


def evaluate_symbolic_spatial_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grids: Tensor,
    proposal_boxes: Tensor,
    matched_gt_boxes: Tensor | None,
    has_matched_gt: Tensor | None,
    gt_iou: Tensor | None,
    random_state: int = 42,
) -> dict[str, Any]:
    if matched_gt_boxes is None or has_matched_gt is None:
        return {
            "evaluated_roi_count": 0,
            "box_grounded_roi_overlap": float("nan"),
            "pointing_score": float("nan"),
            "energy_in_defect_ratio": float("nan"),
            "heatmap_entropy": float("nan"),
            "stability_score": float("nan"),
            "evaluation_extension": {
                "spatial_metrics": [],
                "note": (
                    "No matched GT boxes were available. Spatial explanation metrics could not be computed."
                ),
            },
        }

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

        heatmap = _normalize_heatmap(_compute_local_instance_heatmap(tree, feature_grids[index]))
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=tuple(int(value) for value in heatmap.shape),
        )
        if int(gt_mask.sum().item()) == 0:
            continue

        overlap_scores.append(_topk_region_overlap(heatmap, gt_mask))
        pointing_scores.append(_pointing_score(heatmap, gt_mask))
        energy_scores.append(_energy_in_region(heatmap, gt_mask))
        entropy_scores.append(_heatmap_entropy(heatmap))
        stability_scores.append(
            _feature_perturbation_stability(
                tree,
                feature_grid=feature_grids[index],
                base_heatmap=heatmap,
                random_state=random_state + index,
            )
        )

    if not overlap_scores:
        return {
            "evaluated_roi_count": 0,
            "box_grounded_roi_overlap": float("nan"),
            "pointing_score": float("nan"),
            "energy_in_defect_ratio": float("nan"),
            "heatmap_entropy": float("nan"),
            "stability_score": float("nan"),
            "evaluation_extension": {
                "spatial_metrics": [
                    "box_grounded_roi_overlap",
                    "pointing_score",
                    "energy_in_defect_ratio",
                    "heatmap_entropy",
                    "stability_score",
                ],
                "note": (
                    "Spatial metrics are box-grounded on the RoI grid because the current symbolic export contains GT "
                    "boxes, not pixel masks. They should not be overclaimed as pixel-level faithfulness."
                ),
            },
        }

    return {
        "evaluated_roi_count": int(len(overlap_scores)),
        "box_grounded_roi_overlap": float(np.mean(overlap_scores)),
        "pointing_score": float(np.mean(pointing_scores)),
        "energy_in_defect_ratio": float(np.mean(energy_scores)),
        "heatmap_entropy": float(np.mean(entropy_scores)),
        "stability_score": float(np.mean(stability_scores)),
        "evaluation_extension": {
            "spatial_metrics": [
                "box_grounded_roi_overlap",
                "pointing_score",
                "energy_in_defect_ratio",
                "heatmap_entropy",
                "stability_score",
            ],
            "note": (
                "These spatial metrics are thesis-specific evaluation extensions. They are box-grounded on the RoI "
                "grid when only GT boxes are available, and they should not be interpreted as pixel-level ground truth."
            ),
        },
    }
