"""Grad-CAM on the detector backbone — the reference the tree's heatmaps compare against."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuro.faster_rcnn import NeuroFasterRCNN


class GradCAM:
    """Saliency maps off the backbone's last conv block, one per detection."""

    def __init__(
        self,
        model: NeuroFasterRCNN,
        device: str | torch.device = "cpu",
    ) -> None:
        """Point at a trained detector; hooks the last conv block."""
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Last conv block holds the richest meaning before the neck.
        self._target_layer = self._resolve_layer("layer4")

        # Setup
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None

        # Hooks
        self._forward_hook = self._target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = self._target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _resolve_layer(self, layer_name: str) -> nn.Module:
        """Find a backbone layer by dotted name."""
        module = self.model.backbone.extractor.body
        for attr in layer_name.split("."):
            module = getattr(module, attr)
        return module

    def _save_activation(
        self, _module: nn.Module, _input: tuple, output: Tensor
    ) -> None:
        """Cache the layer's output."""
        self._activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_input: tuple, grad_output: tuple
    ) -> None:
        """Cache the gradients flowing in."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image: Tensor,
        boxes: Tensor,
        class_indices: Tensor,
        output_size: tuple[int, int] = (7, 7),
    ) -> list[Tensor]:
        """Heatmaps for the given boxes, in original-image coordinates."""
        if boxes.shape[0] == 0:
            return []

        image = image.to(self.device).unsqueeze(0)  # [1, 3, H, W]

        # Forward (gradients on — inference-mode would kill them).
        was_training = self.model.training
        self.model.eval()

        # Preprocess
        images_list, _ = self.model.transform(image, None)

        # Gradients must survive whatever mode the caller runs in.
        with torch.inference_mode(False), torch.enable_grad():
            # Fresh tensor — inference tensors can't carry gradients.
            input_tensor = images_list.tensors.clone()
            features = self.model.backbone(input_tensor)

            # Boxes must follow the image into preprocessed space.
            original_h, original_w = image.shape[-2:]
            processed_h, processed_w = images_list.image_sizes[0]
            scale_y = processed_h / original_h
            scale_x = processed_w / original_w
            scaled_boxes = boxes.clone().float().to(self.device)
            scaled_boxes[:, [0, 2]] *= scale_x
            scaled_boxes[:, [1, 3]] *= scale_y

            # Pool
            pooled = self.model.roi_align(
                features, [scaled_boxes], images_list.image_sizes
            )
            roi_representations = self.model.box_head(pooled)
            class_logits = self.model.box_predictor.classifier(roi_representations)

            # Heatmaps
            heatmaps: list[Tensor] = []
            for idx in range(boxes.shape[0]):
                self.model.zero_grad()

                # Target score
                target_class = int(class_indices[idx])
                score = class_logits[idx, target_class]

                # Backward
                score.backward(retain_graph=True)

                if self._gradients is None or self._activations is None:
                    # Blank map when hooks miss.
                    heatmaps.append(torch.ones(output_size, device=self.device))
                    continue

                # Weight channels by their gradients, keep positive parts.
                weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                cam = (weights * self._activations).sum(
                    dim=1, keepdim=True
                )  # [1, 1, H, W]
                cam = F.relu(cam)  # Only positive contributions.

                # Undo the backbone stride exactly.
                padded_h, padded_w = images_list.tensors.shape[-2:]
                cam_upsampled = F.interpolate(
                    cam, size=(padded_h, padded_w), mode="bilinear", align_corners=False
                )

                # Cut padding so the map lines up with the image.
                cam_unpadded = cam_upsampled[:, :, :processed_h, :processed_w]

                x1 = max(0, int(scaled_boxes[idx, 0]))
                y1 = max(0, int(scaled_boxes[idx, 1]))
                x2 = min(processed_w, int(scaled_boxes[idx, 2]) + 1)
                y2 = min(processed_h, int(scaled_boxes[idx, 3]) + 1)

                if x2 <= x1 or y2 <= y1:
                    heatmaps.append(torch.ones(output_size, device=self.device))
                    continue

                roi_cam = cam_unpadded[:, :, y1:y2, x1:x2]

                # Crop
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

                # Scale
                cam_min = roi_cam.min()
                cam_max = roi_cam.max()
                if cam_max - cam_min > 1e-8:
                    roi_cam = (roi_cam - cam_min) / (cam_max - cam_min)
                else:
                    roi_cam = torch.ones_like(roi_cam)

                heatmaps.append(roi_cam.detach().cpu())

        # Restore mode
        self.model.train(was_training)
        return heatmaps

    def release(self) -> None:
        """Unhook; call when done."""
        self._forward_hook.remove()
        self._backward_hook.remove()
        self._activations = None
        self._gradients = None
