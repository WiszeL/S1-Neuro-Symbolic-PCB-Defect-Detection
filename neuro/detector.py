from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.ops import boxes as box_ops

from .model_components import SFPSPyramid
from .preprocess_dataset import RCNNPreprocessing


def fastrcnn_loss_l1(
    class_logits: Tensor,
    box_regression: Tensor,
    labels: list[Tensor],
    regression_targets: list[Tensor],
) -> tuple[Tensor, Tensor]:
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    num_classes = class_logits.shape[1]
    box_regression = box_regression.reshape(class_logits.shape[0], num_classes, 4)

    if sampled_pos_inds_subset.numel() == 0:
        box_loss = box_regression.sum() * 0.0
    else:
        box_loss = F.l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def soft_nms_single_class(
    boxes: Tensor,
    scores: Tensor,
    iou_thresh: float,
    score_thresh: float,
    sigma: float,
    method: str,
) -> tuple[Tensor, Tensor]:
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=boxes.device)
        empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
        return empty, empty_scores

    remaining_boxes = boxes.clone()
    remaining_scores = scores.clone()
    remaining_indices = torch.arange(boxes.shape[0], device=boxes.device)

    kept_indices: list[Tensor] = []
    kept_scores: list[Tensor] = []

    while remaining_indices.numel() > 0:
        max_pos = torch.argmax(remaining_scores)
        kept_indices.append(remaining_indices[max_pos])
        kept_scores.append(remaining_scores[max_pos])

        if remaining_indices.numel() == 1:
            break

        reference_box = remaining_boxes[max_pos].unsqueeze(0)
        mask = torch.ones_like(remaining_scores, dtype=torch.bool)
        mask[max_pos] = False

        candidate_boxes = remaining_boxes[mask]
        candidate_scores = remaining_scores[mask]
        candidate_indices = remaining_indices[mask]

        ious = box_iou(reference_box, candidate_boxes).squeeze(0)
        if method == "gaussian":
            decay = torch.exp(-(ious * ious) / sigma)
        else:
            decay = torch.ones_like(ious)
            suppressed = ious > iou_thresh
            decay[suppressed] = 1.0 - ious[suppressed]

        candidate_scores = candidate_scores * decay
        valid = candidate_scores >= score_thresh

        remaining_boxes = candidate_boxes[valid]
        remaining_scores = candidate_scores[valid]
        remaining_indices = candidate_indices[valid]

    return torch.stack(kept_indices), torch.stack(kept_scores)


def batched_soft_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_thresh: float,
    score_thresh: float,
    sigma: float,
    detections_per_img: int,
    method: str,
) -> tuple[Tensor, Tensor]:
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=boxes.device)
        empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
        return empty, empty_scores

    kept_indices: list[Tensor] = []
    kept_scores: list[Tensor] = []

    for label in labels.unique(sorted=True):
        class_indices = torch.where(labels == label)[0]
        class_keep, class_scores = soft_nms_single_class(
            boxes[class_indices],
            scores[class_indices],
            iou_thresh=iou_thresh,
            score_thresh=score_thresh,
            sigma=sigma,
            method=method,
        )
        if class_keep.numel() == 0:
            continue

        kept_indices.append(class_indices[class_keep])
        kept_scores.append(class_scores)

    if not kept_indices:
        empty = torch.empty((0,), dtype=torch.int64, device=boxes.device)
        empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
        return empty, empty_scores

    kept_indices_tensor = torch.cat(kept_indices, dim=0)
    kept_scores_tensor = torch.cat(kept_scores, dim=0)
    order = torch.argsort(kept_scores_tensor, descending=True)[:detections_per_img]
    return kept_indices_tensor[order], kept_scores_tensor[order]


class L1RegionProposalNetwork(RegionProposalNetwork):
    def compute_loss(
        self,
        objectness: Tensor,
        pred_bbox_deltas: Tensor,
        labels: list[Tensor],
        regression_targets: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()
        labels_tensor = torch.cat(labels, dim=0)
        regression_targets_tensor = torch.cat(regression_targets, dim=0)

        if sampled_pos_inds.numel() == 0:
            box_loss = pred_bbox_deltas.sum() * 0.0
        else:
            box_loss = F.l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets_tensor[sampled_pos_inds],
                reduction="sum",
            )
            box_loss = box_loss / sampled_inds.numel()

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels_tensor[sampled_inds]
        )
        return objectness_loss, box_loss


class InspectableTwoMLPHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class InspectableRoIHeads(RoIHeads):
    def __init__(
        self,
        *args: Any,
        use_soft_nms: bool = True,
        soft_nms_sigma: float = 0.5,
        soft_nms_method: str = "linear",
        soft_nms_score_thresh: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_soft_nms = use_soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_method = soft_nms_method
        self.soft_nms_score_thresh = soft_nms_score_thresh

    def extract_box_embeddings(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[Tensor]:
        teacher_outputs = self.extract_teacher_box_outputs(features, proposals, image_shapes)
        return [output["roi_embeddings"] for output in teacher_outputs]

    def extract_teacher_box_outputs(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        roi_embeddings = self.box_head(pooled_features)
        class_logits, box_regression = self.box_predictor(roi_embeddings)

        counts = [proposal.shape[0] for proposal in proposals]
        pooled_splits = pooled_features.split(counts, dim=0)
        embedding_splits = roi_embeddings.split(counts, dim=0)
        logits_splits = class_logits.split(counts, dim=0)
        box_regression_splits = box_regression.split(counts, dim=0)

        outputs: list[dict[str, Tensor]] = []
        for pooled_per_image, embeddings_per_image, logits_per_image, box_regression_per_image in zip(
            pooled_splits,
            embedding_splits,
            logits_splits,
            box_regression_splits,
        ):
            outputs.append(
                {
                    "pooled_features": pooled_per_image,
                    "roi_embeddings": embeddings_per_image,
                    "class_logits": logits_per_image,
                    "box_regression": box_regression_per_image,
                }
            )
        return outputs

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
        box_features: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes.shape[0] for boxes in proposals]

        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
        box_features_list = box_features.split(boxes_per_image, dim=0)

        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_features: list[Tensor] = []

        for boxes, scores, roi_features, image_shape in zip(
            pred_boxes_list, pred_scores_list, box_features_list, image_shapes
        ):
            boxes = boxes.reshape(-1, num_classes, 4)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            boxes = boxes[:, 1:, :].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = (
                torch.arange(1, num_classes, device=device)
                .view(1, -1)
                .expand(scores.numel() // (num_classes - 1), -1)
                .reshape(-1)
            )
            roi_features = (
                roi_features[:, None, :]
                .expand(-1, num_classes - 1, -1)
                .reshape(-1, roi_features.shape[-1])
            )

            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, roi_features = (
                boxes[inds],
                scores[inds],
                labels[inds],
                roi_features[inds],
            )

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, roi_features = (
                boxes[keep],
                scores[keep],
                labels[keep],
                roi_features[keep],
            )

            if self.use_soft_nms:
                keep, updated_scores = batched_soft_nms(
                    boxes,
                    scores,
                    labels,
                    iou_thresh=self.nms_thresh,
                    score_thresh=self.soft_nms_score_thresh,
                    sigma=self.soft_nms_sigma,
                    detections_per_img=self.detections_per_img,
                    method=self.soft_nms_method,
                )
                boxes, scores, labels, roi_features = (
                    boxes[keep],
                    updated_scores,
                    labels[keep],
                    roi_features[keep],
                )
            else:
                keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[
                    : self.detections_per_img
                ]
                boxes, scores, labels, roi_features = (
                    boxes[keep],
                    scores[keep],
                    labels[keep],
                    roi_features[keep],
                )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_features.append(roi_features)

        return all_boxes, all_scores, all_labels, all_features

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        if targets is not None:
            for target in targets:
                if target["boxes"].dtype not in (
                    torch.float32,
                    torch.float64,
                    torch.float16,
                ):
                    raise TypeError("Target boxes must be floating point tensors.")
                if target["labels"].dtype != torch.int64:
                    raise TypeError("Target labels must be int64 tensors.")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = (
                self.select_training_samples(proposals, targets)
            )
        else:
            labels = None
            regression_targets = None

        # ---------------------------------------------------------------------
        # Run RoI Align and keep the shared RoI embedding explicit
        # ---------------------------------------------------------------------
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, Tensor]] = []
        losses: dict[str, Tensor] = {}

        # ---------------------------------------------------------------------
        # Use L1 box regression during training and Soft-NMS during inference
        # ---------------------------------------------------------------------
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError(
                    "Training targets are required when the RoI heads are in training mode."
                )
            loss_classifier, loss_box_reg = fastrcnn_loss_l1(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels, roi_features = self.postprocess_detections(
                class_logits,
                box_regression,
                proposals,
                image_shapes,
                box_features,
            )
            for (
                boxes_per_image,
                scores_per_image,
                labels_per_image,
                features_per_image,
            ) in zip(boxes, scores, labels, roi_features):
                result.append(
                    {
                        "boxes": boxes_per_image,
                        "scores": scores_per_image,
                        "labels": labels_per_image,
                        "roi_features": features_per_image,
                    }
                )

        return result, losses


def replace_batch_norm_with_frozen_batch_norm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight)
                frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            replace_batch_norm_with_frozen_batch_norm(child)


class SFPSPyramidBackbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        attention_reduction: int = 16,
        freeze_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "ImageNet backbone pretraining is enabled, but the ResNet-50 weights could not be loaded. "
                    "Cache the torchvision weights locally or set model.backbone.pretrained=false."
                ) from exc
            raise
        if freeze_batch_norm:
            replace_batch_norm_with_frozen_batch_norm(backbone)
        self.body = IntermediateLayerGetter(
            backbone,
            return_layers={
                "layer1": "c2",
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )
        self.neck = SFPSPyramid(
            out_channels=256, attention_reduction=attention_reduction
        )
        self.out_channels = self.neck.out_channels

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        features = self.body(x)
        return self.neck(features)


class SFPSPyramidDetector(FasterRCNN):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        train_min_sizes: tuple[int, ...],
        eval_min_size: int,
        max_size: int,
        image_mean: list[float],
        image_std: list[float],
        rpn_anchor_generator: AnchorGenerator,
        box_roi_pool: MultiScaleRoIAlign,
        box_head: nn.Module,
        box_predictor: nn.Module,
        rpn_config: dict[str, Any],
        box_config: dict[str, Any],
        soft_nms_config: dict[str, Any],
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=None,
            min_size=eval_min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            rpn_pre_nms_top_n_train=rpn_config["pre_nms_top_n_train"],
            rpn_pre_nms_top_n_test=rpn_config["pre_nms_top_n_test"],
            rpn_post_nms_top_n_train=rpn_config["post_nms_top_n_train"],
            rpn_post_nms_top_n_test=rpn_config["post_nms_top_n_test"],
            rpn_nms_thresh=rpn_config["nms_thresh"],
            rpn_fg_iou_thresh=rpn_config["fg_iou_thresh"],
            rpn_bg_iou_thresh=rpn_config["bg_iou_thresh"],
            rpn_batch_size_per_image=rpn_config["batch_size_per_image"],
            rpn_positive_fraction=rpn_config["positive_fraction"],
            box_fg_iou_thresh=box_config["fg_iou_thresh"],
            box_bg_iou_thresh=box_config["bg_iou_thresh"],
            box_batch_size_per_image=box_config["batch_size_per_image"],
            box_positive_fraction=box_config["positive_fraction"],
            bbox_reg_weights=box_config["bbox_reg_weights"],
            box_score_thresh=box_config["score_thresh"],
            box_nms_thresh=soft_nms_config["iou_thresh"],
            box_detections_per_img=box_config["detections_per_img"],
        )

        self.transform = RCNNPreprocessing(
            train_min_sizes=train_min_sizes,
            eval_min_size=eval_min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
        )

        self.rpn = L1RegionProposalNetwork(
            anchor_generator=self.rpn.anchor_generator,
            head=self.rpn.head,
            fg_iou_thresh=self.rpn.proposal_matcher.high_threshold,
            bg_iou_thresh=self.rpn.proposal_matcher.low_threshold,
            batch_size_per_image=self.rpn.fg_bg_sampler.batch_size_per_image,
            positive_fraction=self.rpn.fg_bg_sampler.positive_fraction,
            pre_nms_top_n=self.rpn._pre_nms_top_n,
            post_nms_top_n=self.rpn._post_nms_top_n,
            nms_thresh=self.rpn.nms_thresh,
            score_thresh=self.rpn.score_thresh,
        )

        self.roi_heads = InspectableRoIHeads(
            self.roi_heads.box_roi_pool,
            self.roi_heads.box_head,
            self.roi_heads.box_predictor,
            self.roi_heads.proposal_matcher.high_threshold,
            self.roi_heads.proposal_matcher.low_threshold,
            self.roi_heads.fg_bg_sampler.batch_size_per_image,
            self.roi_heads.fg_bg_sampler.positive_fraction,
            self.roi_heads.box_coder.weights,
            self.roi_heads.score_thresh,
            self.roi_heads.nms_thresh,
            self.roi_heads.detections_per_img,
            use_soft_nms=soft_nms_config["enabled"],
            soft_nms_sigma=soft_nms_config["sigma"],
            soft_nms_method=soft_nms_config["method"],
            soft_nms_score_thresh=soft_nms_config["score_thresh"],
        )

    @torch.inference_mode()
    def extract_teacher_roi_samples(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> list[dict[str, Tensor]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]
        images_list, transformed_targets = self.transform(images, targets)

        features = self.backbone(images_list.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict({"p2": features})

        proposals, _ = self.rpn(images_list, features, transformed_targets)
        teacher_outputs = self.roi_heads.extract_teacher_box_outputs(
            features,
            proposals,
            images_list.image_sizes,
        )

        postprocessed_boxes = self.transform.postprocess(
            [{"boxes": boxes.clone()} for boxes in proposals],
            images_list.image_sizes,
            original_image_sizes,
        )

        results: list[dict[str, Tensor]] = []
        for image_index, (teacher_output, boxes_dict) in enumerate(zip(teacher_outputs, postprocessed_boxes)):
            class_scores = F.softmax(teacher_output["class_logits"], dim=-1)
            result = {
                "proposal_boxes": boxes_dict["boxes"],
                "pooled_features": teacher_output["pooled_features"],
                "feature_vectors": teacher_output["pooled_features"].flatten(start_dim=1),
                "roi_embeddings": teacher_output["roi_embeddings"],
                "teacher_logits": teacher_output["class_logits"],
                "teacher_scores": class_scores,
                "teacher_labels": class_scores.argmax(dim=-1),
                "box_regression": teacher_output["box_regression"],
                "image_size": torch.tensor(original_image_sizes[image_index], dtype=torch.int64),
                "transformed_image_size": torch.tensor(images_list.image_sizes[image_index], dtype=torch.int64),
            }

            if targets is not None:
                proposal_boxes = boxes_dict["boxes"]
                target_boxes = targets[image_index]["boxes"]
                target_labels = targets[image_index]["labels"]
                if proposal_boxes.numel() == 0 or target_boxes.numel() == 0:
                    matched_iou = torch.zeros((proposal_boxes.shape[0],), dtype=torch.float32)
                    matched_labels = torch.zeros((proposal_boxes.shape[0],), dtype=torch.int64)
                else:
                    overlaps = box_iou(proposal_boxes, target_boxes)
                    matched_iou, matched_indices = overlaps.max(dim=1)
                    matched_labels = target_labels[matched_indices]
                    matched_labels = torch.where(
                        matched_iou > 0,
                        matched_labels,
                        torch.zeros_like(matched_labels),
                    )
                result["gt_labels"] = matched_labels
                result["gt_iou"] = matched_iou

            results.append(result)

        return results

    def extract_target_roi_features(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]
        images_list, transformed_targets = self.transform(images, targets)

        features = self.backbone(images_list.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict({"p2": features})

        proposals = [target["boxes"] for target in transformed_targets]
        teacher_outputs = self.roi_heads.extract_teacher_box_outputs(
            features, proposals, images_list.image_sizes
        )

        postprocessed_boxes = self.transform.postprocess(
            [{"boxes": boxes.clone()} for boxes in proposals],
            images_list.image_sizes,
            original_image_sizes,
        )

        results: list[dict[str, Tensor]] = []
        for transformed_target, boxes_dict, teacher_output in zip(
            transformed_targets, postprocessed_boxes, teacher_outputs
        ):
            results.append(
                {
                    "boxes": boxes_dict["boxes"],
                    "labels": transformed_target["labels"],
                    "pooled_features": teacher_output["pooled_features"],
                    "roi_features": teacher_output["roi_embeddings"],
                }
            )
        return results


def build_detector(config: dict[str, Any]) -> SFPSPyramidDetector:
    model_config = config["model"] if "model" in config else config
    class_names = tuple(model_config["class_names"])
    num_classes_without_background = int(model_config["num_classes"])

    if num_classes_without_background != len(class_names):
        raise ValueError(
            "model.num_classes must match the length of model.class_names."
        )

    train_min_sizes = tuple(
        int(size) for size in model_config["image"]["train_min_sizes"]
    )
    eval_min_size = int(model_config["image"]["test_min_size"])
    max_size = int(model_config["image"]["max_size"])

    anchor_sizes = tuple(
        tuple(level_sizes) for level_sizes in model_config["anchors"]["sizes"]
    )
    aspect_ratios = tuple(
        tuple(level_ratios) for level_ratios in model_config["anchors"]["aspect_ratios"]
    )

    backbone = SFPSPyramidBackbone(
        pretrained=bool(model_config["backbone"]["pretrained"]),
        attention_reduction=int(model_config["neck"]["attention_reduction"]),
        freeze_batch_norm=bool(model_config["backbone"].get("freeze_batch_norm", False)),
    )

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pool = MultiScaleRoIAlign(
        featmap_names=model_config["roi"]["featmap_names"],
        output_size=int(model_config["roi"]["pool_size"]),
        sampling_ratio=int(model_config["roi"]["sampling_ratio"]),
    )

    resolution = int(model_config["roi"]["pool_size"])
    representation_size = int(model_config["roi"]["representation_size"])
    box_head = InspectableTwoMLPHead(
        backbone.out_channels * resolution * resolution, representation_size
    )
    box_predictor = FastRCNNPredictor(
        representation_size, num_classes_without_background + 1
    )

    return SFPSPyramidDetector(
        backbone=backbone,
        num_classes=num_classes_without_background + 1,
        train_min_sizes=train_min_sizes,
        eval_min_size=eval_min_size,
        max_size=max_size,
        image_mean=list(model_config["image"]["mean"]),
        image_std=list(model_config["image"]["std"]),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        rpn_config=model_config["rpn"],
        box_config=model_config["box"],
        soft_nms_config=model_config["soft_nms"],
    )
