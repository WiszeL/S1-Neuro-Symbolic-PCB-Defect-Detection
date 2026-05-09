from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from neuro.faster_rcnn import NeuroFasterRCNN
from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from util.device import select_device

from .utils import postprocess_symbolic_detections


class NeuroSymbolicDetector(nn.Module):
    def __init__(
        self,
        detector: NeuroFasterRCNN,
        symbolic_tree: SparseObliqueDecisionTreeClassifier,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.symbolic_tree = symbolic_tree
        self.device = select_device(device)
        if self.symbolic_tree.num_classes != self.detector.num_classes:
            raise ValueError(
                "The symbolic tree class count must match the detector class count."
            )
        self.detector.to(self.device)
        self.detector.eval()

    def _symbolic_predict(self, pooled_features: Tensor) -> tuple[Tensor, Tensor]:
        feature_vectors = (
            pooled_features.flatten(start_dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        probabilities = self.symbolic_tree.predict_proba(feature_vectors)
        leaf_indices = self.symbolic_tree.predict_leaf_indices(feature_vectors)
        probability_tensor = torch.from_numpy(probabilities).to(
            self.device,
            dtype=pooled_features.dtype,
        )
        leaf_tensor = torch.from_numpy(leaf_indices).to(self.device, dtype=torch.int64)
        return probability_tensor, leaf_tensor

    @torch.inference_mode()
    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]
        images_list, _ = self.detector.transform(
            [image.to(self.device) for image in images], None
        )

        features = self.detector.backbone(images_list.tensors)
        proposals, _ = self.detector.rpn(images_list, features, None)
        pooled_features = self.detector.roi_align(
            features,
            proposals,
            images_list.image_sizes,
        )
        roi_representations = self.detector.box_head(pooled_features)
        box_regression = self.detector.box_predictor.box_regressor(roi_representations)
        symbolic_probabilities, symbolic_leaf_indices = self._symbolic_predict(
            pooled_features
        )

        results = postprocess_symbolic_detections(
            detector=self.detector,
            box_regression=box_regression,
            proposals=proposals,
            image_shapes=images_list.image_sizes,
            pooled_features=pooled_features,
            proposal_probabilities=symbolic_probabilities,
            proposal_leaf_indices=symbolic_leaf_indices,
        )

        postprocessed_boxes = self.detector.transform.postprocess(
            [{"boxes": result["boxes"].clone()} for result in results],
            images_list.image_sizes,
            original_image_sizes,
        )
        postprocessed_proposals = self.detector.transform.postprocess(
            [{"boxes": result["proposal_boxes"].clone()} for result in results],
            images_list.image_sizes,
            original_image_sizes,
        )

        outputs: list[dict[str, Tensor]] = []
        for result, boxes_dict, proposals_dict in zip(
            results, postprocessed_boxes, postprocessed_proposals
        ):
            outputs.append(
                {
                    "boxes": boxes_dict["boxes"].detach().cpu(),
                    "scores": result["scores"].detach().cpu(),
                    "labels": result["labels"].detach().cpu(),
                    "proposal_boxes": proposals_dict["boxes"].detach().cpu(),
                    "pooled_features": result["pooled_features"].detach().cpu(),
                    "symbolic_probabilities": result["symbolic_probabilities"]
                    .detach()
                    .cpu(),
                    "symbolic_leaf_indices": result["symbolic_leaf_indices"]
                    .detach()
                    .cpu(),
                }
            )

        return outputs
