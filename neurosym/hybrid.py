from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from util.device import select_device

from .utils import postprocess_symbolic_detections


class NeuroSymbolicDetector(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        symbolic_tree: SparseObliqueDecisionTreeClassifier,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.teacher_model = teacher_model
        self.symbolic_tree = symbolic_tree
        self.device = select_device(device)
        expected_num_classes = int(self.teacher_model.roi_heads.box_predictor.cls_score.out_features)
        if self.symbolic_tree.num_classes != expected_num_classes:
            raise ValueError(
                "The symbolic tree class count must match the detector RoI classifier output dimension."
            )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    @torch.inference_mode()
    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        original_image_sizes = [tuple(image.shape[-2:]) for image in images]
        images_list, _ = self.teacher_model.transform([image.to(self.device) for image in images], None)

        features = self.teacher_model.backbone(images_list.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict({"p2": features})

        proposals, _ = self.teacher_model.rpn(images_list, features, None)
        teacher_outputs = self.teacher_model.roi_heads.extract_teacher_box_outputs(
            features,
            proposals,
            images_list.image_sizes,
        )

        symbolic_logits: list[Tensor] = []
        pooled_features: list[Tensor] = []
        box_regression: list[Tensor] = []
        symbolic_probabilities: list[Tensor] = []
        symbolic_leaf_indices: list[Tensor] = []

        for teacher_output in teacher_outputs:
            pooled_per_image = teacher_output["pooled_features"]
            pooled_features.append(pooled_per_image)
            box_regression.append(teacher_output["box_regression"])

            if pooled_per_image.shape[0] == 0:
                symbolic_logits.append(teacher_output["class_logits"])
                symbolic_probabilities.append(
                    torch.zeros(
                        (0, self.symbolic_tree.num_classes),
                        dtype=teacher_output["class_logits"].dtype,
                        device=self.device,
                    )
                )
                symbolic_leaf_indices.append(
                    torch.zeros((0,), dtype=torch.int64, device=self.device)
                )
                continue

            feature_vectors = pooled_per_image.flatten(start_dim=1).detach().cpu().numpy().astype(np.float32)
            probabilities = self.symbolic_tree.predict_proba(feature_vectors)
            log_probabilities = np.log(np.clip(probabilities, 1e-8, 1.0))
            leaf_indices = self.symbolic_tree.predict_leaf_indices(feature_vectors)

            symbolic_logits.append(
                torch.from_numpy(log_probabilities).to(
                    self.device,
                    dtype=teacher_output["class_logits"].dtype,
                )
            )
            symbolic_probabilities.append(
                torch.from_numpy(probabilities).to(
                    self.device,
                    dtype=teacher_output["class_logits"].dtype,
                )
            )
            symbolic_leaf_indices.append(
                torch.from_numpy(leaf_indices).to(self.device, dtype=torch.int64)
            )

        results = postprocess_symbolic_detections(
            roi_heads=self.teacher_model.roi_heads,
            class_logits=torch.cat(symbolic_logits, dim=0),
            box_regression=torch.cat(box_regression, dim=0),
            proposals=proposals,
            image_shapes=images_list.image_sizes,
            pooled_features=torch.cat(pooled_features, dim=0),
            proposal_probabilities=torch.cat(symbolic_probabilities, dim=0),
            proposal_leaf_indices=torch.cat(symbolic_leaf_indices, dim=0),
        )

        postprocessed_boxes = self.teacher_model.transform.postprocess(
            [{"boxes": result["boxes"].clone()} for result in results],
            images_list.image_sizes,
            original_image_sizes,
        )
        postprocessed_proposals = self.teacher_model.transform.postprocess(
            [{"boxes": result["proposal_boxes"].clone()} for result in results],
            images_list.image_sizes,
            original_image_sizes,
        )

        outputs: list[dict[str, Tensor]] = []
        for result, boxes_dict, proposals_dict in zip(results, postprocessed_boxes, postprocessed_proposals):
            outputs.append(
                {
                    "boxes": boxes_dict["boxes"],
                    "scores": result["scores"].detach().cpu(),
                    "labels": result["labels"].detach().cpu(),
                    "proposal_boxes": proposals_dict["boxes"],
                    "pooled_features": result["pooled_features"].detach().cpu(),
                    "symbolic_probabilities": result["symbolic_probabilities"].detach().cpu(),
                    "symbolic_leaf_indices": result["symbolic_leaf_indices"].detach().cpu(),
                }
            )

        return outputs
