from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import boxes as box_ops

from neuro.faster_rcnn import NeuroFasterRCNN
from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def calibrate_temperature(
    tree: SparseObliqueDecisionTreeClassifier,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_candidates: int = 50,
    t_min: float = 0.1,
    t_max: float = 5.0,
) -> float:
    """Find the optimal temperature T for the SODT's leaf distributions.

    Minimises the negative log-likelihood (NLL) of the true labels under
    the temperature-calibrated leaf probabilities:

        calibrated = softmax(log(leaf_dist) / T)
        NLL = -mean(log(calibrated[i, label_i]))

    This is a simple 1-D grid search.  The argmax class decision is
    T-invariant, so temperature only affects score magnitudes used for
    detection postprocessing (NMS), not classification decisions.

    Args:
        tree: A trained SODT with populated leaf distributions.
        val_features: 2-D feature matrix for the validation set.
        val_labels: 1-D integer labels for the validation set.
        num_candidates: Number of T values to evaluate in [t_min, t_max].
        t_min: Lower bound for temperature search.
        t_max: Upper bound for temperature search.

    Returns:
        The temperature T that minimises NLL on the validation set.
    """
    features = (
        val_features
        if val_features.dtype == np.float32
        else np.asarray(val_features, dtype=np.float32)
    )
    labels = np.asarray(val_labels, dtype=np.int64)

    # Pre-compute raw leaf distributions for all validation samples
    raw_probs = tree.predict_proba(features)
    log_probs = np.log(np.clip(raw_probs, 1e-12, 1.0))

    candidates = np.linspace(t_min, t_max, num_candidates)
    best_t = 1.0
    best_nll = float("inf")

    for T in candidates:
        scaled_logits = log_probs / float(T)
        # Stable softmax
        shifted = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        calibrated = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

        # NLL: -mean(log(p(correct_class)))
        nll = -float(
            np.mean(
                np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0))
            )
        )
        if nll < best_nll:
            best_nll = nll
            best_t = float(T)

    return best_t


def postprocess_symbolic_detections(
    detector: NeuroFasterRCNN,
    box_regression: Tensor,
    proposals: list[Tensor],
    image_shapes: list[tuple[int, int]],
    pooled_features: Tensor,
    proposal_probabilities: Tensor,
    proposal_leaf_indices: Tensor,
) -> list[dict[str, Tensor]]:
    device = proposal_probabilities.device
    num_classes = proposal_probabilities.shape[-1]
    boxes_per_image = [boxes.shape[0] for boxes in proposals]

    pred_boxes = detector.box_coder.decode(box_regression, proposals)
    pred_scores = proposal_probabilities

    pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
    pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
    pooled_features_list = pooled_features.split(boxes_per_image, dim=0)
    probabilities_list = proposal_probabilities.split(boxes_per_image, dim=0)
    leaf_indices_list = proposal_leaf_indices.split(boxes_per_image, dim=0)

    results: list[dict[str, Tensor]] = []
    for (
        boxes,
        scores,
        proposals_per_image,
        pooled_per_image,
        probabilities_per_image,
        leaf_indices_per_image,
        image_shape,
    ) in zip(
        pred_boxes_list,
        pred_scores_list,
        proposals,
        pooled_features_list,
        probabilities_list,
        leaf_indices_list,
        image_shapes,
    ):
        boxes = boxes.reshape(-1, num_classes, 4)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        boxes = boxes[:, 1:, :].reshape(-1, 4)
        scores = scores[:, 1:].reshape(-1)

        labels = (
            torch.arange(1, num_classes, device=device)
            .view(1, -1)
            .expand(proposals_per_image.shape[0], -1)
            .reshape(-1)
        )
        proposal_boxes = (
            proposals_per_image[:, None, :]
            .expand(-1, num_classes - 1, -1)
            .reshape(-1, 4)
        )
        pooled_grids = (
            pooled_per_image[:, None, :, :, :]
            .expand(-1, num_classes - 1, -1, -1, -1)
            .reshape(-1, *pooled_per_image.shape[1:])
        )
        symbolic_probabilities = (
            probabilities_per_image[:, None, :]
            .expand(-1, num_classes - 1, -1)
            .reshape(-1, probabilities_per_image.shape[-1])
        )
        symbolic_leaf_indices = (
            leaf_indices_per_image[:, None].expand(-1, num_classes - 1).reshape(-1)
        )

        keep = torch.where(scores > detector.BOX_SCORE_THRESH)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        proposal_boxes = proposal_boxes[keep]
        pooled_grids = pooled_grids[keep]
        symbolic_probabilities = symbolic_probabilities[keep]
        symbolic_leaf_indices = symbolic_leaf_indices[keep]

        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        proposal_boxes = proposal_boxes[keep]
        pooled_grids = pooled_grids[keep]
        symbolic_probabilities = symbolic_probabilities[keep]
        symbolic_leaf_indices = symbolic_leaf_indices[keep]

        if detector.soft_nms_enabled:
            keep, updated_scores = detector.batched_soft_nms(
                boxes,
                scores,
                labels,
            )
            boxes = boxes[keep]
            scores = updated_scores
            labels = labels[keep]
            proposal_boxes = proposal_boxes[keep]
            pooled_grids = pooled_grids[keep]
            symbolic_probabilities = symbolic_probabilities[keep]
            symbolic_leaf_indices = symbolic_leaf_indices[keep]
        else:
            keep = box_ops.batched_nms(
                boxes,
                scores,
                labels,
                detector.soft_nms_iou_thresh,
            )[: detector.DETECTIONS_PER_IMG]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            proposal_boxes = proposal_boxes[keep]
            pooled_grids = pooled_grids[keep]
            symbolic_probabilities = symbolic_probabilities[keep]
            symbolic_leaf_indices = symbolic_leaf_indices[keep]

        results.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "proposal_boxes": proposal_boxes,
                "pooled_features": pooled_grids,
                "symbolic_probabilities": symbolic_probabilities,
                "symbolic_leaf_indices": symbolic_leaf_indices,
            }
        )

    return results
