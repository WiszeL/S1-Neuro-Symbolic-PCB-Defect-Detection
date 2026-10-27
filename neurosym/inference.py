from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neuro.faster_rcnn import NeuroFasterRCNN
from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from util.device import select_device

from .heatmap import (
    compute_symbolic_heatmap,
    compute_node_local_evidence_maps,
    fpn_node_maps,
    fpn_node_maps_many,
)
from .hybrid import NeuroSymbolicDetector


def load_neurosymbolic_detector(
    detector_checkpoint_path: str | Path,
    neuro_config: dict[str, Any],
    train_config: dict[str, Any],
    symbolic_checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[NeuroSymbolicDetector, dict[str, Any]]:
    resolved_device = select_device(device)
    detector_neuro_config = deepcopy(neuro_config)
    # Weights come from the checkpoint, not a fresh ImageNet download.
    detector_neuro_config["net"]["backbone_pretrained"] = False
    detector_checkpoint = torch.load(
        detector_checkpoint_path,
        map_location=resolved_device,
        weights_only=True,
    )
    detector = NeuroFasterRCNN(
        neuro_config=detector_neuro_config,  # type: ignore[arg-type]
        train_config=train_config,  # type: ignore[arg-type]
    )
    detector.load_state_dict(detector_checkpoint["model_state_dict"])
    symbolic_checkpoint = torch.load(
        symbolic_checkpoint_path, map_location="cpu", weights_only=True
    )
    symbolic_tree = SparseObliqueDecisionTreeClassifier.from_state_dict(
        symbolic_checkpoint["tree_state"]
    )
    hybrid_model = NeuroSymbolicDetector(
        detector=detector,
        symbolic_tree=symbolic_tree,
        device=str(resolved_device),
    )
    return hybrid_model, detector_checkpoint


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
    image: Tensor,
    mode: str = "local_instance_evidence_map",
    pixel_attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decision path plus exact pixel maps for one detection; `image` is the original [3, H, W] image.

    `pixel_attribution` lets a caller that already has the maps (one pass for many detections) pass them in.
    """
    feature_grid = detection["pooled_features"][detection_index]
    explanation = compute_symbolic_heatmap(
        model.symbolic_tree,
        feature_grid=feature_grid,
        mode=mode,
    )

    path = model.symbolic_tree.decision_path(feature_grid.detach().cpu().numpy().reshape(-1))
    explanation["node_explanations"] = compute_node_local_evidence_maps(
        model.symbolic_tree, feature_grid
    )
    explanation["proposal_box"] = detection["proposal_boxes"][detection_index].detach().cpu()
    explanation["detection_box"] = detection["boxes"][detection_index].detach().cpu()
    explanation["label"] = int(detection["labels"][detection_index])
    explanation["score"] = float(detection["scores"][detection_index])
    explanation["detection_index"] = int(detection_index)
    explanation["symbolic_leaf_index"] = int(
        detection["symbolic_leaf_indices"][detection_index]
    )
    explanation["symbolic_probabilities"] = detection["symbolic_probabilities"][
        detection_index
    ]
    explanation["explanation_mode"] = mode
    # Same path as the node entries above, so the maps explain the decision on screen.
    explanation["pixel_attribution"] = pixel_attribution or fpn_node_maps(
        model, image, detection["proposal_boxes_processed"][detection_index], path=path
    )
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
        selected_indices = selected_indices[: max(max_detections, 0)]
    return selected_indices


def explain_hybrid_detections(
    model: NeuroSymbolicDetector,
    detection: dict[str, Tensor],
    image: Tensor,
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

    if not selected_indices:
        return []
    paths = [
        model.symbolic_tree.decision_path(
            detection["pooled_features"][index].detach().cpu().numpy().reshape(-1)
        )
        for index in selected_indices
    ]
    attributions = fpn_node_maps_many(
        model, image, detection["proposal_boxes_processed"][selected_indices], paths=paths, cache=detection
    )
    return [
        explain_hybrid_detection(
            model, detection, index, image, mode=mode, pixel_attribution=attribution
        )
        for index, attribution in zip(selected_indices, attributions)
    ]
