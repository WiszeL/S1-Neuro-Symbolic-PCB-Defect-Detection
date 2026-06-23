"""Grad-CAM implementation for the project's Faster R-CNN backbone.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
from Selvaraju et al. (2017) as a baseline explainability method. It hooks into
the ResNet-50 backbone (specifically `layer4` / `c5`) to produce spatial
saliency maps that highlight which image regions drive the neural classifier's
decision for each detected defect.

The resulting heatmaps serve as a qualitative comparison against the SODT's
local evidence heatmaps — both methods should highlight similar defect-relevant
regions, validating that the SODT's symbolic explanations are faithful to the
underlying neural features.

Reference:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
    Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. ICCV 2017.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuro.faster_rcnn import NeuroFasterRCNN


class GradCAM:
    """Grad-CAM extractor for the Faster R-CNN detector.

    This class hooks into the ResNet-50 backbone's last convolutional layer
    (``layer4`` / ``c5``) and computes class-discriminative saliency maps for
    each RoI proposal.

    Usage::

        model = ...  # NeuroFasterRCNN, loaded and on device
        gradcam = GradCAM(model, device="cuda")

        # For a single image tensor [3, H, W]:
        heatmaps = gradcam.generate(image, boxes, class_indices)
        # heatmaps: list of [H_roi, W_roi] tensors, one per detection
    """

    def __init__(
        self,
        model: NeuroFasterRCNN,
        device: str | torch.device = "cpu",
        target_layer: Optional[str] = None,
    ) -> None:
        """Initialize Grad-CAM for the given Faster R-CNN model.

        Parameters
        ----------
        model:
            The trained NeuroFasterRCNN detector. Must be in eval mode.
        device:
            Device to run computations on.
        target_layer:
            Which ResNet layer to hook for activation maps. Defaults to
            ``"layer4"`` (the ``c5`` stage), which has the richest semantic
            information before the neck.
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Resolve target layer — default to the last conv block of ResNet-50.
        if target_layer is None:
            target_layer = "layer4"
        self._target_layer = self._resolve_layer(target_layer)

        # Storage for hook outputs.
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None

        # Register hooks.
        self._forward_hook = self._target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = self._target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _resolve_layer(self, layer_name: str) -> nn.Module:
        """Resolve a dot-separated layer name on the ResNet backbone body."""
        module = self.model.backbone.extractor.body
        for attr in layer_name.split("."):
            module = getattr(module, attr)
        return module

    def _save_activation(
        self, _module: nn.Module, _input: tuple, output: Tensor
    ) -> None:
        """Forward hook: cache the target layer's output activations."""
        self._activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_input: tuple, grad_output: tuple
    ) -> None:
        """Backward hook: cache gradients flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image: Tensor,
        boxes: Tensor,
        class_indices: Tensor,
        output_size: tuple[int, int] = (7, 7),
    ) -> list[Tensor]:
        """Generate Grad-CAM heatmaps for the given detections.

        Parameters
        ----------
        image:
            Input image tensor ``[3, H, W]``, normalized to ``[0, 1]``.
        boxes:
            Detection boxes ``[N, 4]`` in ``(x1, y1, x2, y2)`` format,
            in the *original* image coordinate space.
        class_indices:
            Predicted class index for each box ``[N]`` (1-indexed defect
            labels, matching the Faster R-CNN convention).
        output_size:
            Spatial size to resize each per-RoI heatmap to.

        Returns
        -------
        list[Tensor]
            One heatmap ``[H_out, W_out]`` per detection, values in ``[0, 1]``.
        """
        if boxes.shape[0] == 0:
            return []

        image = image.to(self.device).unsqueeze(0)  # [1, 3, H, W]

        # --- Forward pass with gradients enabled --------------------------
        # We need gradients through the backbone, so we cannot use
        # @torch.inference_mode or @torch.no_grad here.
        was_training = self.model.training
        self.model.eval()

        # Run the RCNN preprocessing (resize + normalize).
        images_list, _ = self.model.transform(image, None)

        # We must disable inference mode and enable gradients for the entire
        # forward & backward Grad-CAM extraction loop to be robust to callers
        # running in torch.inference_mode() or torch.no_grad().
        with torch.inference_mode(False), torch.enable_grad():
            # Clone input tensor inside normal autograd mode to ensure it's not an inference tensor.
            input_tensor = images_list.tensors.clone()
            features = self.model.backbone(input_tensor)

            # Scale detection boxes to the preprocessed image coordinate space
            # so RoI Align produces the correct crops.
            original_h, original_w = image.shape[-2:]
            processed_h, processed_w = images_list.image_sizes[0]
            scale_y = processed_h / original_h
            scale_x = processed_w / original_w
            scaled_boxes = boxes.clone().float().to(self.device)
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y

            # Pool features per detection box.
            pooled = self.model.roi_align(features, [scaled_boxes], images_list.image_sizes)
            roi_representations = self.model.box_head(pooled)
            class_logits = self.model.box_predictor.classifier(roi_representations)

            # --- Generate one heatmap per detection ---------------------------
            heatmaps: list[Tensor] = []
            for idx in range(boxes.shape[0]):
                self.model.zero_grad()

                # Target class score for this detection.
                target_class = int(class_indices[idx])
                score = class_logits[idx, target_class]

                # Backward to get gradients at the target layer.
                score.backward(retain_graph=True)

                if self._gradients is None or self._activations is None:
                    # Fallback: uniform heatmap if hooks did not fire.
                    heatmaps.append(torch.ones(output_size, device=self.device))
                    continue

                # Grad-CAM: GAP of gradients → channel weights → weighted sum.
                # activations: [1, C, H_feat, W_feat]
                # gradients:   [1, C, H_feat, W_feat]
                weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
                cam = F.relu(cam)  # Only positive contributions.

                # Upsample CAM to padded image size to reverse the backbone stride exactly.
                padded_h, padded_w = images_list.tensors.shape[-2:]
                cam_upsampled = F.interpolate(
                    cam, size=(padded_h, padded_w), mode="bilinear", align_corners=False
                )

                # Crop away the padding to perfectly align with the scaled (processed) image.
                cam_unpadded = cam_upsampled[:, :, :processed_h, :processed_w]

                x1 = max(0, int(scaled_boxes[idx, 0]))
                y1 = max(0, int(scaled_boxes[idx, 1]))
                x2 = min(processed_w, int(scaled_boxes[idx, 2]) + 1)
                y2 = min(processed_h, int(scaled_boxes[idx, 3]) + 1)

                if x2 <= x1 or y2 <= y1:
                    heatmaps.append(torch.ones(output_size, device=self.device))
                    continue

                roi_cam = cam_unpadded[:, :, y1:y2, x1:x2]

                # Resize to the standard RoI output size.
                roi_cam = (
                    F.interpolate(
                        roi_cam,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                # Normalize to [0, 1].
                cam_min = roi_cam.min()
                cam_max = roi_cam.max()
                if cam_max - cam_min > 1e-8:
                    roi_cam = (roi_cam - cam_min) / (cam_max - cam_min)
                else:
                    roi_cam = torch.ones_like(roi_cam)

                heatmaps.append(roi_cam.detach().cpu())

        # Restore training state.
        self.model.train(was_training)
        return heatmaps

    def release(self) -> None:
        """Remove hooks to free resources."""
        self._forward_hook.remove()
        self._backward_hook.remove()
        self._activations = None
        self._gradients = None
