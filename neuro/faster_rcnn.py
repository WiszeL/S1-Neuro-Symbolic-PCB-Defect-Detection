from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.ops import boxes as box_ops
from torchvision.ops.misc import FrozenBatchNorm2d

from .config import NeuroConfig, NeuroTrainConfig
from .prepare_dataset import PCBTarget
from .preprocess_dataset import RCNNPreprocessing


FeatureMap = OrderedDict[str, Tensor]
SoftNMSMethod = Literal["linear", "gaussian", "hard"]


class ResNet50Extractor(nn.Module):
    """Four ResNet stages (c2-c5) feeding the neck."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        if freeze_batch_norm:
            self.replace_batch_norm_with_frozen_batch_norm(backbone)

        self.body = IntermediateLayerGetter(
            backbone,
            return_layers={
                "layer1": "c2",
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )

    def forward(self, images: Tensor) -> FeatureMap:
        """Return c2-c5 feature maps for a batch of images."""

        return self.body(images)

    @staticmethod
    def replace_batch_norm_with_frozen_batch_norm(module: nn.Module) -> None:
        """Frozen norms stay stable on tiny detection batches."""

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
                ResNet50Extractor.replace_batch_norm_with_frozen_batch_norm(child)


class SFPSPyramid(nn.Module):
    """Neck: deep meaning rebuilt at high resolution for tiny defects."""

    class MConv(nn.Module):
        """One stage squeezed into the shared channel width."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            )
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
            self.activation = nn.ReLU(inplace=True)

        def forward(self, feature: Tensor) -> Tensor:
            feature = self.activation(self.depthwise(feature))
            feature = self.activation(self.pointwise(feature))
            return feature

    class CPBlock(nn.Module):
        """Learn channels first, then shuffle them into 2x resolution."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.expand = nn.Conv2d(
                in_channels,
                out_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.activation = nn.ReLU(inplace=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        def forward(self, feature: Tensor) -> Tensor:
            feature = self.activation(self.expand(feature))
            return self.pixel_shuffle(feature)

    class SFAttention(nn.Module):
        """Blend deep meaning into detailed maps, per channel."""

        # Paper never sizes this bottleneck, so fixed at 32.
        HIDDEN_CHANNELS = 32

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.compress = nn.Linear(channels, self.HIDDEN_CHANNELS)
            self.expand = nn.Linear(self.HIDDEN_CHANNELS, channels * 2)
            self.activation = nn.ReLU(inplace=True)

        def forward(self, semantic_feature: Tensor, spatial_feature: Tensor) -> Tensor:
            if semantic_feature.shape[-2:] != spatial_feature.shape[-2:]:
                semantic_feature = F.interpolate(
                    semantic_feature,
                    size=spatial_feature.shape[-2:],
                    mode="nearest",
                )

            fused_descriptor = semantic_feature + spatial_feature
            fused_descriptor = F.adaptive_avg_pool2d(
                fused_descriptor,
                output_size=1,
            ).flatten(1)
            fused_descriptor = self.activation(self.compress(fused_descriptor))

            weights = self.expand(fused_descriptor)
            weights = weights.view(weights.shape[0], 2, -1, 1, 1)
            weights = torch.softmax(weights, dim=1)

            return weights[:, 0] * semantic_feature + weights[:, 1] * spatial_feature

    # Keep in step with BoxHead.POOLED_CHANNELS.
    OUT_CHANNELS = 64

    def __init__(self) -> None:
        super().__init__()
        self.out_channels = self.OUT_CHANNELS

        self.c5_to_p5 = self.MConv(2048, self.OUT_CHANNELS)
        self.c2_to_p1 = self.MConv(256, self.OUT_CHANNELS)

        self.c5_to_p4 = self.CPBlock(2048, self.OUT_CHANNELS)
        self.c4_to_p3 = self.CPBlock(1024, self.OUT_CHANNELS)
        self.c3_to_p2 = self.CPBlock(512, self.OUT_CHANNELS)

        self.p3_attention = self.SFAttention(self.OUT_CHANNELS)
        self.p2_attention = self.SFAttention(self.OUT_CHANNELS)

    def forward(self, stages: dict[str, Tensor]) -> FeatureMap:
        """Build p2-p6 exactly from the c2-c5 ResNet stage maps."""

        c2 = stages["c2"]
        c3 = stages["c3"]
        c4 = stages["c4"]
        c5 = stages["c5"]

        # Context
        p5 = self.c5_to_p5(c5)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2)

        # Detail
        p4 = self.c5_to_p4(c5)
        p3 = self.c4_to_p3(c4)
        p2 = self.c3_to_p2(c3)
        p1 = self.c2_to_p1(c2)

        # Fuse
        p3_prime = self.p3_attention(p4, p3)
        p2_prime = self.p2_attention(p3_prime, p1 + p2)

        return OrderedDict(
            {
                "p2": p2_prime,
                "p3": p3_prime,
                "p4": p4,
                "p5": p5,
                "p6": p6,
            }
        )


class BackboneWithNeck(nn.Module):
    """Backbone plus neck, as one torchvision-ready module."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = ResNet50Extractor(
            pretrained=pretrained,
            freeze_batch_norm=freeze_batch_norm,
        )
        self.neck = SFPSPyramid()
        self.out_channels = self.neck.out_channels

    def forward(self, images: Tensor) -> FeatureMap:
        """Return p2-p6 pyramid features for RPN and RoI Align."""

        stages = self.extractor(images)

        return self.neck(stages)


class L1RegionProposalNetwork(RegionProposalNetwork):
    """RPN with plain L1 box loss — tighter boxes on PCB."""

    # YAML covers anchors and thresholds; these extras stay hardcoded.
    BATCH_SIZE_PER_IMAGE = 256
    POSITIVE_FRACTION = 0.5
    PRE_NMS_TOP_N_TRAIN = 2000
    PRE_NMS_TOP_N_TEST = 1000
    POST_NMS_TOP_N_TRAIN = 2000
    POST_NMS_TOP_N_TEST = 1000
    NMS_THRESH = 0.7
    SCORE_THRESH = 0.0

    @staticmethod
    def build_anchor_generator(neuro_config: NeuroConfig) -> AnchorGenerator:
        """Create anchors for the five SF-PSPyramid levels: p2-p6."""

        return AnchorGenerator(
            sizes=neuro_config["anchors"]["sizes"],
            aspect_ratios=neuro_config["anchors"]["aspect_ratios"],
        )

    @staticmethod
    def build(
        backbone_out_channels: int,
        neuro_config: NeuroConfig,
    ) -> "L1RegionProposalNetwork":
        """Build the complete RPN module for the SF-PSPyramid features."""

        anchor_generator = L1RegionProposalNetwork.build_anchor_generator(neuro_config)
        rpn_head = RPNHead(
            in_channels=backbone_out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        return L1RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=neuro_config["proposal"]["positive_iou"],
            bg_iou_thresh=neuro_config["proposal"]["negative_iou"],
            batch_size_per_image=L1RegionProposalNetwork.BATCH_SIZE_PER_IMAGE,
            positive_fraction=L1RegionProposalNetwork.POSITIVE_FRACTION,
            pre_nms_top_n={
                "training": L1RegionProposalNetwork.PRE_NMS_TOP_N_TRAIN,
                "testing": L1RegionProposalNetwork.PRE_NMS_TOP_N_TEST,
            },
            post_nms_top_n={
                "training": L1RegionProposalNetwork.POST_NMS_TOP_N_TRAIN,
                "testing": L1RegionProposalNetwork.POST_NMS_TOP_N_TEST,
            },
            nms_thresh=L1RegionProposalNetwork.NMS_THRESH,
            score_thresh=L1RegionProposalNetwork.SCORE_THRESH,
        )

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

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds],
            labels_tensor[sampled_inds],
        )

        if sampled_pos_inds.numel() == 0:
            box_loss = pred_bbox_deltas.sum() * 0.0
        else:
            box_loss = F.l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets_tensor[sampled_pos_inds],
                reduction="sum",
            )
            box_loss = box_loss / sampled_inds.numel()

        return objectness_loss, box_loss


class RoIAlign(nn.Module):
    """The feature cut: pooled grids the tree trains on."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5", "p6"],
            output_size=7,
            sampling_ratio=2,
        )

    def forward(
        self,
        features: dict[str, Tensor],
        proposal_boxes: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> Tensor:
        """Return true RoI Align pooled features: [total_rois, 64, 7, 7]."""

        return self.pool(features, proposal_boxes, image_shapes)

    def extract(
        self,
        features: dict[str, Tensor],
        proposal_boxes: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Return per-image proposal boxes and pooled features for SODT export."""

        pooled_features = self(features, proposal_boxes, image_shapes)
        proposal_counts = [boxes.shape[0] for boxes in proposal_boxes]
        pooled_features_per_image = pooled_features.split(proposal_counts, dim=0)

        records: list[dict[str, Tensor]] = []
        for boxes, pooled in zip(proposal_boxes, pooled_features_per_image):
            records.append(
                {
                    "proposal_boxes": boxes,
                    "pooled_features": pooled,
                }
            )

        return records


class BoxHead(nn.Module):
    """Two FC layers turning each grid into one vector."""

    # Keep in step with SFPSPyramid.OUT_CHANNELS.
    POOLED_CHANNELS = 64
    POOLED_SIZE = 7
    REPRESENTATION_SIZE = 1024

    def __init__(self) -> None:
        super().__init__()
        input_size = self.POOLED_CHANNELS * self.POOLED_SIZE * self.POOLED_SIZE
        self.fc6 = nn.Linear(input_size, self.REPRESENTATION_SIZE)
        self.fc7 = nn.Linear(self.REPRESENTATION_SIZE, self.REPRESENTATION_SIZE)
        self.representation_size = self.REPRESENTATION_SIZE

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Return shared RoI representations: [total_rois, 1024]."""

        hidden = pooled_features.flatten(start_dim=1)
        hidden = F.relu(self.fc6(hidden))
        hidden = F.relu(self.fc7(hidden))
        return hidden


class BoxPredictor(nn.Module):
    """Final scores and box fixes, one set per class."""

    def __init__(self, representation_size: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(representation_size, num_classes)
        self.box_regressor = nn.Linear(representation_size, num_classes * 4)

    def forward(self, roi_representations: Tensor) -> tuple[Tensor, Tensor]:
        """Return class logits and class-specific box deltas."""

        class_logits = self.classifier(roi_representations)
        box_regression = self.box_regressor(roi_representations)
        return class_logits, box_regression


def fast_rcnn_l1_loss(
    class_logits: Tensor,
    box_regression: Tensor,
    labels: list[Tensor],
    regression_targets: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Same as torchvision, but plain L1 for tighter boxes."""

    labels_tensor = torch.cat(labels, dim=0)
    regression_targets_tensor = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels_tensor)

    positive_indices = torch.where(labels_tensor > 0)[0]
    if positive_indices.numel() == 0:
        box_loss = box_regression.sum() * 0.0
        return classification_loss, box_loss

    num_classes = class_logits.shape[1]
    box_regression = box_regression.reshape(class_logits.shape[0], num_classes, 4)
    positive_labels = labels_tensor[positive_indices]

    box_loss = F.l1_loss(
        box_regression[positive_indices, positive_labels],
        regression_targets_tensor[positive_indices],
        reduction="sum",
    )
    box_loss = box_loss / labels_tensor.numel()

    return classification_loss, box_loss


class NeuroFasterRCNN(nn.Module):
    """The full detector, wired from the blocks above."""

    # The rest stay hardcoded; YAML is the only config surface.
    BOX_FG_IOU_THRESH = 0.5
    BOX_BG_IOU_THRESH = 0.5
    BOX_BATCH_SIZE_PER_IMAGE = 512
    BOX_POSITIVE_FRACTION = 0.25
    BOX_SCORE_THRESH = 0.001
    DETECTIONS_PER_IMG = 100

    def __init__(
        self,
        neuro_config: NeuroConfig,
        train_config: NeuroTrainConfig,
    ) -> None:
        super().__init__()
        self.num_classes = len(train_config["dataset"]["class_names"]) + 1

        self.soft_nms_enabled = neuro_config["soft_nms"]["enabled"]
        self.soft_nms_method = neuro_config["soft_nms"]["method"]
        self.soft_nms_iou_thresh = neuro_config["soft_nms"]["iou_thresh"]
        self.soft_nms_sigma = neuro_config["soft_nms"]["sigma"]
        self.soft_nms_score_thresh = neuro_config["soft_nms"]["score_thresh"]

        self.transform = RCNNPreprocessing(train_config)
        self.backbone = BackboneWithNeck(
            pretrained=neuro_config["net"]["backbone_pretrained"],
            freeze_batch_norm=neuro_config["net"]["backbone_freeze_batch_norm"],
        )
        self.rpn = L1RegionProposalNetwork.build(
            self.backbone.out_channels,
            neuro_config=neuro_config,
        )
        self.roi_align = RoIAlign()
        self.box_head = BoxHead()
        self.box_predictor = BoxPredictor(
            representation_size=self.box_head.representation_size,
            num_classes=self.num_classes,
        )
        self.proposal_matcher = det_utils.Matcher(
            self.BOX_FG_IOU_THRESH,
            self.BOX_BG_IOU_THRESH,
            allow_low_quality_matches=False,
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            self.BOX_BATCH_SIZE_PER_IMAGE,
            self.BOX_POSITIVE_FRACTION,
        )
        self.box_coder = det_utils.BoxCoder((10.0, 10.0, 5.0, 5.0))

    @staticmethod
    def soft_nms_single_class(
        boxes: Tensor,
        scores: Tensor,
        iou_thresh: float,
        score_thresh: float,
        sigma: float,
        method: SoftNMSMethod,
    ) -> tuple[Tensor, Tensor]:
        """Soft-NMS for one class of candidate boxes."""

        if boxes.numel() == 0:
            empty_indices = torch.empty((0,), dtype=torch.int64, device=boxes.device)
            empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
            return empty_indices, empty_scores

        remaining_boxes = boxes.clone()
        remaining_scores = scores.clone()
        remaining_indices = torch.arange(boxes.shape[0], device=boxes.device)

        kept_indices: list[Tensor] = []
        kept_scores: list[Tensor] = []

        while remaining_indices.numel() > 0:
            best_position = torch.argmax(remaining_scores)
            kept_indices.append(remaining_indices[best_position])
            kept_scores.append(remaining_scores[best_position])

            if remaining_indices.numel() == 1:
                break

            reference_box = remaining_boxes[best_position].unsqueeze(0)
            candidate_mask = torch.ones_like(remaining_scores, dtype=torch.bool)
            candidate_mask[best_position] = False

            candidate_boxes = remaining_boxes[candidate_mask]
            candidate_scores = remaining_scores[candidate_mask]
            candidate_indices = remaining_indices[candidate_mask]

            overlaps = box_iou(reference_box, candidate_boxes).squeeze(0)
            if method == "gaussian":
                decay = torch.exp(-(overlaps * overlaps) / sigma)
            elif method == "hard":
                decay = torch.where(overlaps > iou_thresh, 0.0, 1.0)
            else:
                decay = torch.ones_like(overlaps)
                high_overlap = overlaps > iou_thresh
                decay[high_overlap] = 1.0 - overlaps[high_overlap]

            candidate_scores = candidate_scores * decay
            valid = candidate_scores >= score_thresh

            remaining_boxes = candidate_boxes[valid]
            remaining_scores = candidate_scores[valid]
            remaining_indices = candidate_indices[valid]

        return torch.stack(kept_indices), torch.stack(kept_scores)

    def batched_soft_nms(
        self,
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply Soft-NMS independently for each predicted defect class."""

        if boxes.numel() == 0:
            empty_indices = torch.empty((0,), dtype=torch.int64, device=boxes.device)
            empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
            return empty_indices, empty_scores

        kept_indices: list[Tensor] = []
        kept_scores: list[Tensor] = []

        for label in labels.unique(sorted=True):
            class_indices = torch.where(labels == label)[0]
            class_keep, class_scores = NeuroFasterRCNN.soft_nms_single_class(
                boxes=boxes[class_indices],
                scores=scores[class_indices],
                iou_thresh=self.soft_nms_iou_thresh,
                score_thresh=self.soft_nms_score_thresh,
                sigma=self.soft_nms_sigma,
                method=self.soft_nms_method,
            )
            if class_keep.numel() > 0:
                kept_indices.append(class_indices[class_keep])
                kept_scores.append(class_scores)

        if not kept_indices:
            empty_indices = torch.empty((0,), dtype=torch.int64, device=boxes.device)
            empty_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
            return empty_indices, empty_scores

        kept_indices_tensor = torch.cat(kept_indices, dim=0)
        kept_scores_tensor = torch.cat(kept_scores, dim=0)
        order = torch.argsort(kept_scores_tensor, descending=True)
        order = order[: self.DETECTIONS_PER_IMG]

        return kept_indices_tensor[order], kept_scores_tensor[order]

    def assign_targets_to_proposals(
        self,
        proposals: list[Tensor],
        gt_boxes: list[Tensor],
        gt_labels: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Match each proposal to a ground-truth box for Fast R-CNN training."""

        matched_idxs: list[Tensor] = []
        labels: list[Tensor] = []
        for proposals_per_image, gt_boxes_per_image, gt_labels_per_image in zip(
            proposals,
            gt_boxes,
            gt_labels,
        ):
            if gt_boxes_per_image.numel() == 0:
                device = proposals_per_image.device
                matched_idxs.append(
                    torch.zeros(
                        (proposals_per_image.shape[0],),
                        dtype=torch.int64,
                        device=device,
                    )
                )
                labels.append(
                    torch.zeros(
                        (proposals_per_image.shape[0],),
                        dtype=torch.int64,
                        device=device,
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(
                gt_boxes_per_image, proposals_per_image
            )
            matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
            clamped_matched_idxs = matched_idxs_per_image.clamp(min=0)

            labels_per_image = gt_labels_per_image[clamped_matched_idxs].to(torch.int64)
            labels_per_image[
                matched_idxs_per_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            ] = 0
            labels_per_image[
                matched_idxs_per_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            ] = -1

            matched_idxs.append(clamped_matched_idxs)
            labels.append(labels_per_image)

        return matched_idxs, labels

    def select_training_samples(
        self,
        proposals: list[Tensor],
        targets: list[PCBTarget],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Sample positive/negative RoIs and create box regression targets."""

        gt_boxes = [target["boxes"] for target in targets]
        gt_labels = [target["labels"] for target in targets]

        proposals = [
            torch.cat([proposals_per_image, gt_boxes_per_image], dim=0)
            for proposals_per_image, gt_boxes_per_image in zip(proposals, gt_boxes)
        ]
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals,
            gt_boxes,
            gt_labels,
        )
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        matched_gt_boxes: list[Tensor] = []
        sampled_proposals: list[Tensor] = []
        sampled_labels: list[Tensor] = []
        for image_index, (proposals_per_image, gt_boxes_per_image) in enumerate(
            zip(proposals, gt_boxes)
        ):
            sampled_inds = torch.where(
                sampled_pos_inds[image_index] | sampled_neg_inds[image_index]
            )[0]

            proposals_per_image = proposals_per_image[sampled_inds]
            labels_per_image = labels[image_index][sampled_inds]
            matched_idxs_per_image = matched_idxs[image_index][sampled_inds]

            if gt_boxes_per_image.numel() == 0:
                gt_boxes_per_image = torch.zeros(
                    (1, 4),
                    dtype=proposals_per_image.dtype,
                    device=proposals_per_image.device,
                )

            sampled_proposals.append(proposals_per_image)
            sampled_labels.append(labels_per_image)
            matched_gt_boxes.append(gt_boxes_per_image[matched_idxs_per_image])

        regression_targets = self.box_coder.encode(matched_gt_boxes, sampled_proposals)

        return sampled_proposals, sampled_labels, regression_targets

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Decode predictions and apply class-wise Soft-NMS."""

        device = class_logits.device
        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)

        detections: list[dict[str, Tensor]] = []
        for boxes, scores, image_shape, proposals_per_image in zip(
            pred_boxes_list,
            pred_scores_list,
            image_shapes,
            proposals,
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(self.num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # The proposal each candidate was classified from — its class
            # logit came from pooling there, not at the regressed box.
            proposal_index = torch.arange(scores.shape[0], device=device)
            proposal_index = proposal_index.view(-1, 1).expand_as(scores)

            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)
            proposal_index = proposal_index[:, 1:].reshape(-1)

            keep = torch.where(scores > self.BOX_SCORE_THRESH)[0]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            proposal_index = proposal_index[keep]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            proposal_index = proposal_index[keep]

            if self.soft_nms_enabled:
                keep, updated_scores = self.batched_soft_nms(
                    boxes,
                    scores,
                    labels,
                )
                boxes = boxes[keep]
                scores = updated_scores
                labels = labels[keep]
                proposal_index = proposal_index[keep]
            else:
                # No separate hard-NMS knob — reuses the Soft-NMS threshold.
                keep = box_ops.batched_nms(
                    boxes,
                    scores,
                    labels,
                    self.soft_nms_iou_thresh,
                )
                keep = keep[: self.DETECTIONS_PER_IMG]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                proposal_index = proposal_index[keep]

            detections.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    # Processed-space, like hybrid.py; postprocess leaves it alone.
                    "proposal_boxes_processed": proposals_per_image[proposal_index],
                }
            )

        return detections

    def forward(
        self,
        images: list[Tensor],
        targets: list[PCBTarget] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Losses when training, detections when evaluating."""

        if self.training and targets is None:
            raise ValueError(
                "targets must be provided when NeuroFasterRCNN is training."
            )

        original_image_sizes = [tuple(image.shape[-2:]) for image in images]

        transformed_images, transformed_targets = self.transform(images, targets)

        features = self.backbone(transformed_images.tensors)
        proposals, proposal_losses = self.rpn(
            transformed_images,
            features,
            transformed_targets,
        )

        if self.training:
            if transformed_targets is None:
                raise ValueError("transformed targets are required for training.")

            sampled_proposals, labels, regression_targets = (
                self.select_training_samples(
                    proposals,
                    transformed_targets,
                )
            )
            pooled_features = self.roi_align(
                features,
                sampled_proposals,
                transformed_images.image_sizes,
            )
            roi_representations = self.box_head(pooled_features)
            class_logits, box_regression = self.box_predictor(roi_representations)
            loss_classifier, loss_box_reg = fast_rcnn_l1_loss(
                class_logits,
                box_regression,
                labels,
                regression_targets,
            )

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
            losses.update(proposal_losses)
            return losses

        pooled_features = self.roi_align(
            features,
            proposals,
            transformed_images.image_sizes,
        )
        roi_representations = self.box_head(pooled_features)
        class_logits, box_regression = self.box_predictor(roi_representations)
        detections = self.postprocess_detections(
            class_logits,
            box_regression,
            proposals,
            transformed_images.image_sizes,
        )
        return self.transform.postprocess(
            detections,
            transformed_images.image_sizes,
            original_image_sizes,
        )

    @torch.inference_mode()
    def _extract_proposal_feature_records(
        self,
        images: list[Tensor],
        targets: list[PCBTarget] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Proposals plus pooled grids, before the box head sees them."""

        self.eval()
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]
        transformed_images, transformed_targets = self.transform(images, targets)
        features = self.backbone(transformed_images.tensors)
        proposals, _ = self.rpn(transformed_images, features, transformed_targets)
        records = self.roi_align.extract(
            features,
            proposals,
            transformed_images.image_sizes,
        )
        postprocessed_boxes = self.transform.postprocess(
            [{"boxes": record["proposal_boxes"]} for record in records],
            transformed_images.image_sizes,
            original_image_sizes,
        )

        for index, (record, boxes) in enumerate(zip(records, postprocessed_boxes)):
            record["transformed_proposal_boxes"] = record["proposal_boxes"]
            record["proposal_boxes"] = boxes["boxes"]
            record["image_size"] = torch.tensor(
                original_image_sizes[index],
                device=record["proposal_boxes"].device,
                dtype=torch.int64,
            )
            record["transformed_image_size"] = torch.tensor(
                transformed_images.image_sizes[index],
                device=record["proposal_boxes"].device,
                dtype=torch.int64,
            )

        return records

    @torch.inference_mode()
    def extract_teacher_roi_samples(
        self,
        images: list[Tensor],
        targets: list[PCBTarget] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Teacher labels for the same proposals the tree trains on."""

        records = self._extract_proposal_feature_records(images, targets)
        pooled_features = torch.cat(
            [record["pooled_features"] for record in records], dim=0
        )
        proposal_counts = [record["proposal_boxes"].shape[0] for record in records]

        roi_representations = self.box_head(pooled_features)
        teacher_logits = self.box_predictor.classifier(roi_representations)
        teacher_scores = F.softmax(teacher_logits, dim=-1)
        teacher_labels = teacher_scores.argmax(dim=1)

        logits_per_image = teacher_logits.split(proposal_counts, dim=0)
        scores_per_image = teacher_scores.split(proposal_counts, dim=0)
        labels_per_image = teacher_labels.split(proposal_counts, dim=0)

        for record, logits, scores, labels in zip(
            records,
            logits_per_image,
            scores_per_image,
            labels_per_image,
        ):
            record["teacher_logits"] = logits
            record["teacher_scores"] = scores
            record["teacher_labels"] = labels

        return records
