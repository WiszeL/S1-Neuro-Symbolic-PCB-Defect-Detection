from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neuro.inference import load_checkpoint_model
from symbolic.train import load_symbolic_tree

from .heatmap import (
    aggregate_projected_heatmaps,
    compute_symbolic_heatmap,
    project_heatmap_to_image,
    resize_heatmap_to_box,
)
from .hybrid import NeuroSymbolicDetector


def load_neurosymbolic_detector(
    detector_checkpoint_path: str | Path,
    model_config_path: str | Path,
    symbolic_checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[NeuroSymbolicDetector, dict[str, Any]]:
    teacher_model, teacher_checkpoint = load_checkpoint_model(
        detector_checkpoint_path,
        model_config_path,
        device=device,
    )
    symbolic_tree = load_symbolic_tree(symbolic_checkpoint_path)
    hybrid_model = NeuroSymbolicDetector(
        teacher_model=teacher_model,
        symbolic_tree=symbolic_tree,
        device=device,
    )
    return hybrid_model, teacher_checkpoint


@torch.inference_mode()
def run_neurosymbolic_inference(
    model: NeuroSymbolicDetector,
    images: list[Tensor],
) -> list[dict[str, Tensor]]:
    return model(images)


def explain_hybrid_detection(
    model: NeuroSymbolicDetector,
    detection: dict[str, Tensor],
    detection_index: int,
    image_shape: tuple[int, int],
    mode: str = "local_instance_evidence_map",
) -> dict[str, Any]:
    feature_grid = detection["pooled_features"][detection_index]
    proposal_box = detection["proposal_boxes"][detection_index]
    detection_box = detection["boxes"][detection_index]
    explanation = compute_symbolic_heatmap(
        model.symbolic_tree,
        feature_grid=feature_grid,
        mode=mode,
    )

    heatmap_keys = [
        "global_or_structural_density_map",
        "local_instance_evidence_map",
        "positive_local_evidence_map",
        "negative_local_evidence_map",
        "signed_local_evidence_map",
        "combined_local_evidence_map",
    ]
    projected_maps = {
        key: project_heatmap_to_image(
            explanation[key],
            box=detection_box,
            image_shape=image_shape,
        )
        for key in heatmap_keys
    }
    box_maps = {
        key: resize_heatmap_to_box(
            explanation[key],
            box=detection_box,
        )
        for key in heatmap_keys
    }

    explanation["projected_heatmap"] = projected_maps[mode] if mode in projected_maps else projected_maps["local_instance_evidence_map"]
    explanation["detection_box_heatmap"] = box_maps[mode] if mode in box_maps else box_maps["local_instance_evidence_map"]
    explanation["projected_maps"] = projected_maps
    explanation["detection_box_maps"] = box_maps
    explanation["proposal_box"] = proposal_box.detach().cpu()
    explanation["detection_box"] = detection_box.detach().cpu()
    explanation["label"] = int(detection["labels"][detection_index])
    explanation["score"] = float(detection["scores"][detection_index])
    explanation["detection_index"] = int(detection_index)
    explanation["symbolic_leaf_index"] = int(detection["symbolic_leaf_indices"][detection_index])
    explanation["symbolic_probabilities"] = detection["symbolic_probabilities"][detection_index]
    explanation["symbolic_label_confidence"] = float(
        detection["symbolic_probabilities"][detection_index][explanation["label"]]
    )
    explanation["explanation_mode"] = mode
    explanation["explanation_modes"] = {
        "local_instance_evidence_map": "Per-instance symbolic evidence map for this RoI.",
        "global_or_structural_density_map": "Path-level structural density map from symbolic weights.",
    }
    explanation["intrinsic_reasoning_summary"] = {
        "leaf_index": int(explanation["leaf_index"]),
        "path_length": int(explanation["reasoning_summary"]["path_length"]),
        "active_path_original_feature_count": int(
            explanation["reasoning_summary"]["active_path_original_feature_count"]
        ),
        "mean_active_original_features_per_node": float(
            explanation["reasoning_summary"]["mean_active_original_features_per_node"]
        ),
        "proposal_box": explanation["proposal_box"].tolist(),
        "detection_box": explanation["detection_box"].tolist(),
    }
    return explanation


def select_detection_indices(
    detection: dict[str, Tensor],
    score_threshold: float = 0.3,
    max_detections: int | None = None,
) -> list[int]:
    selected_indices = [
        index
        for index, score in enumerate(detection["scores"])
        if float(score) >= score_threshold
    ]
    if max_detections is not None:
        selected_indices = selected_indices[: max(int(max_detections), 0)]
    return selected_indices


def subset_detection(
    detection: dict[str, Tensor],
    detection_indices: list[int],
) -> dict[str, Tensor]:
    if not detection_indices:
        empty_detection: dict[str, Tensor] = {}
        for key, value in detection.items():
            if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == detection["boxes"].shape[0]:
                empty_detection[key] = value[:0]
            else:
                empty_detection[key] = value
        return empty_detection

    subset: dict[str, Tensor] = {}
    for key, value in detection.items():
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == detection["boxes"].shape[0]:
            index_tensor = torch.as_tensor(
                detection_indices,
                dtype=torch.long,
                device=value.device,
            )
            subset[key] = value.index_select(0, index_tensor)
        else:
            subset[key] = value
    return subset


def explain_hybrid_detections(
    model: NeuroSymbolicDetector,
    detection: dict[str, Tensor],
    image_shape: tuple[int, int],
    detection_indices: list[int] | None = None,
    score_threshold: float = 0.3,
    max_detections: int | None = None,
    mode: str = "local_instance_evidence_map",
) -> list[dict[str, Any]]:
    selected_indices = detection_indices
    if selected_indices is None:
        selected_indices = select_detection_indices(
            detection,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )

    return [
        explain_hybrid_detection(
            model,
            detection,
            detection_index=index,
            image_shape=image_shape,
            mode=mode,
        )
        for index in selected_indices
    ]


def aggregate_detection_heatmaps(
    explanations: list[dict[str, Any]],
    map_key: str = "projected_heatmap",
    reduction: str = "max",
    score_weighted: bool = True,
    normalize: bool = True,
) -> Tensor:
    if not explanations:
        raise ValueError("At least one explanation is required for heatmap aggregation.")

    scores = [float(explanation["score"]) for explanation in explanations] if score_weighted else None
    return aggregate_projected_heatmaps(
        [explanation[map_key] for explanation in explanations],
        scores=scores,
        reduction=reduction,
        normalize=normalize,
    )
