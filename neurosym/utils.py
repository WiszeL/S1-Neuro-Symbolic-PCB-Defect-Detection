from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import boxes as box_ops

from neuro.faster_rcnn import NeuroFasterRCNN


def postprocess_symbolic_detections(
    detector: NeuroFasterRCNN,
    box_regression: Tensor,
    proposals: list[Tensor],
    image_shapes: list[tuple[int, int]],
    pooled_features: Tensor,
    proposal_probabilities: Tensor,
    proposal_leaf_indices: Tensor,
    proposal_level_indices: Tensor | None = None,
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
    if proposal_level_indices is not None:
        level_indices_list = proposal_level_indices.split(boxes_per_image, dim=0)
    else:
        level_indices_list = [None] * len(boxes_per_image)

    results: list[dict[str, Tensor]] = []
    for (
        boxes,
        scores,
        proposals_per_image,
        pooled_per_image,
        probabilities_per_image,
        leaf_indices_per_image,
        level_indices_per_image,
        image_shape,
    ) in zip(
        pred_boxes_list,
        pred_scores_list,
        proposals,
        pooled_features_list,
        probabilities_list,
        leaf_indices_list,
        level_indices_list,
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
        if level_indices_per_image is not None:
            symbolic_level_indices = (
                level_indices_per_image[:, None].expand(-1, num_classes - 1).reshape(-1)
            )
        else:
            symbolic_level_indices = None

        keep = torch.where(scores > detector.BOX_SCORE_THRESH)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        proposal_boxes = proposal_boxes[keep]
        pooled_grids = pooled_grids[keep]
        symbolic_probabilities = symbolic_probabilities[keep]
        symbolic_leaf_indices = symbolic_leaf_indices[keep]
        if symbolic_level_indices is not None:
            symbolic_level_indices = symbolic_level_indices[keep]

        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        proposal_boxes = proposal_boxes[keep]
        pooled_grids = pooled_grids[keep]
        symbolic_probabilities = symbolic_probabilities[keep]
        symbolic_leaf_indices = symbolic_leaf_indices[keep]
        if symbolic_level_indices is not None:
            symbolic_level_indices = symbolic_level_indices[keep]

        use_soft_nms = detector.soft_nms_enabled

        if use_soft_nms:
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
            if symbolic_level_indices is not None:
                symbolic_level_indices = symbolic_level_indices[keep]
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
            if symbolic_level_indices is not None:
                symbolic_level_indices = symbolic_level_indices[keep]

        result = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "proposal_boxes": proposal_boxes,
            "pooled_features": pooled_grids,
            "symbolic_probabilities": symbolic_probabilities,
            "symbolic_leaf_indices": symbolic_leaf_indices,
        }
        if symbolic_level_indices is not None:
            result["symbolic_level_indices"] = symbolic_level_indices
        results.append(result)

    return results
