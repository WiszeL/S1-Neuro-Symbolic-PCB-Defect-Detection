"""Grad-CAM on the detector — the reference the tree's heatmaps compare against."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from neuro.faster_rcnn import NeuroFasterRCNN
from neurosym.heatmap import _fpn_box_bounds, _normalize_heatmap_array


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on the FPN level each RoI is pooled from.

    Grad-CAM targets the last conv layer before the head. Here the box head
    reads only the FPN level RoI-Align picks for that RoI, never ResNet's
    layer4 — the same target pytorch-grad-cam uses for Faster R-CNN
    (`model.backbone`, i.e. the FPN outputs). Class scores come from pooling
    at the proposal, since that pooling is what produced the class decision.
    """

    def __init__(
        self,
        model: NeuroFasterRCNN,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def fpn_maps(
        self,
        image: Tensor,
        proposal_boxes_processed: Tensor,
        class_indices: Tensor,
    ) -> dict[str, Any]:
        """Full-level CAM per RoI (Eq. 2.21-2.22), plus the FPN context it lives in."""
        detector = self.model
        images_list, _ = detector.transform([image.to(self.device)], None)
        boxes = proposal_boxes_processed.detach().to(self.device, dtype=torch.float32)
        featmap_names = list(detector.roi_align.pool.featmap_names)

        # Gradients must survive whatever mode the caller runs in.
        with torch.inference_mode(False), torch.enable_grad():
            features = detector.backbone(images_list.tensors.clone())
            pooled = detector.roi_align(features, [boxes], images_list.image_sizes)
            logits = detector.box_predictor.classifier(detector.box_head(pooled))
            level_indices = detector.roi_align.pool.map_levels([boxes])

            cams: list[Tensor] = []
            level_names: list[str] = []
            for row in range(boxes.shape[0]):
                level_name = featmap_names[int(level_indices[row])]
                activation = features[level_name]
                (grad,) = torch.autograd.grad(
                    logits[row, int(class_indices[row])],
                    activation,
                    retain_graph=True,
                )
                weights = grad[0].mean(dim=(1, 2), keepdim=True)
                cams.append(F.relu((weights * activation[0]).sum(0)).detach())
                level_names.append(level_name)

        return {
            "cams": cams,
            "level_names": level_names,
            "boxes_processed": boxes.detach(),
            "pooled_features": pooled.detach(),
            "fpn_features": {name: level.detach() for name, level in features.items()},
            "padded_size": tuple(images_list.tensors.shape[-2:]),
            "processed_size": tuple(images_list.image_sizes[0]),
        }

    def generate(
        self,
        image: Tensor,
        proposal_boxes_processed: Tensor,
        class_indices: Tensor,
        output_size: tuple[int, int] = (7, 7),
    ) -> list[Tensor]:
        """CAM cropped to each proposal, max-normalized like the tree's map, resized."""
        if proposal_boxes_processed.shape[0] == 0:
            return []

        maps = self.fpn_maps(image, proposal_boxes_processed, class_indices)
        heatmaps: list[Tensor] = []
        for cam, box in zip(maps["cams"], maps["boxes_processed"]):
            fx1, fy1, fx2, fy2 = _fpn_box_bounds(box, maps["padded_size"], tuple(cam.shape))
            if fx2 <= fx1 or fy2 <= fy1:
                heatmaps.append(torch.zeros(output_size))
                continue
            crop = _normalize_heatmap_array(cam[fy1:fy2, fx1:fx2].cpu().numpy())
            heatmaps.append(
                F.interpolate(
                    torch.from_numpy(np.ascontiguousarray(crop))[None, None],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            )
        return heatmaps
