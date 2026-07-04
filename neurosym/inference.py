from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neuro.faster_rcnn import NeuroFasterRCNN
from symbolic.train import load_symbolic_tree
from util.device import select_device

from .heatmap import (
    compute_symbolic_heatmap,
    compute_node_local_evidence_maps,
    project_heatmap_to_image,
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
    # The checkpoint supplies trained backbone weights; avoid an ImageNet weight download during inference init.
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
    symbolic_tree = load_symbolic_tree(symbolic_checkpoint_path)
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
    node_explanations = []
    for node in compute_node_local_evidence_maps(model.symbolic_tree, feature_grid):
        node_heatmap = node["node_heatmap"]
        node_explanations.append(
            {
                **node,
                "projected_node_heatmap_on_proposal_box": project_heatmap_to_image(
                    node_heatmap,
                    box=proposal_box,
                    image_shape=image_shape,
                ),
            }
        )

    explanation["node_explanations"] = node_explanations
    explanation["proposal_box"] = proposal_box.detach().cpu()
    explanation["detection_box"] = detection_box.detach().cpu()
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
