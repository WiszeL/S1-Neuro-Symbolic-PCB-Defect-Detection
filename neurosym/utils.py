from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import boxes as box_ops

from neuro.detector import batched_soft_nms


def postprocess_symbolic_detections(
    roi_heads: Any,
    class_logits: Tensor,
    box_regression: Tensor,
    proposals: list[Tensor],
    image_shapes: list[tuple[int, int]],
    pooled_features: Tensor,
    proposal_probabilities: Tensor,
    proposal_leaf_indices: Tensor,
) -> list[dict[str, Tensor]]:
    device = class_logits.device
    num_classes = class_logits.shape[-1]
    boxes_per_image = [boxes.shape[0] for boxes in proposals]

    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = F.softmax(class_logits, dim=-1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
    pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
    pooled_features_list = pooled_features.split(boxes_per_image, dim=0)
    probabilities_list = proposal_probabilities.split(boxes_per_image, dim=0)
    leaf_indices_list = proposal_leaf_indices.split(boxes_per_image, dim=0)

    results: list[dict[str, Tensor]] = []
    for boxes, scores, proposals_per_image, pooled_per_image, probabilities_per_image, leaf_indices_per_image, image_shape in zip(
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
            leaf_indices_per_image[:, None]
            .expand(-1, num_classes - 1)
            .reshape(-1)
        )

        keep = torch.where(scores > roi_heads.score_thresh)[0]
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

        if roi_heads.use_soft_nms:
            keep, updated_scores = batched_soft_nms(
                boxes,
                scores,
                labels,
                iou_thresh=roi_heads.nms_thresh,
                score_thresh=roi_heads.soft_nms_score_thresh,
                sigma=roi_heads.soft_nms_sigma,
                detections_per_img=roi_heads.detections_per_img,
                method=roi_heads.soft_nms_method,
            )
            boxes = boxes[keep]
            scores = updated_scores
            labels = labels[keep]
            proposal_boxes = proposal_boxes[keep]
            pooled_grids = pooled_grids[keep]
            symbolic_probabilities = symbolic_probabilities[keep]
            symbolic_leaf_indices = symbolic_leaf_indices[keep]
        else:
            keep = box_ops.batched_nms(boxes, scores, labels, roi_heads.nms_thresh)[: roi_heads.detections_per_img]
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
