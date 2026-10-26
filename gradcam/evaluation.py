"""Grad-CAM's report card, scored exactly like the tree's.

- Faithfulness: same image-pixel masking protocol (`pixel_masking_metrics`),
  each method on its own model's detections at the P/R operating point.
- Localization: one shared RoI list for SODT, Grad-CAM and random.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

from neuro.faster_rcnn import NeuroFasterRCNN
from neurosym.evaluation import (
    PATCH_SIZE,
    _mean,
    _tile_order,
    box_tile_mask,
    keep_mask_to_pixels,
    memoize_by_keep_mask,
    patch_scores,
    pixel_map_to_roi_grid,
    pixel_masking_metrics,
    random_order_box_first,
)
from neurosym.heatmap import (
    _normalization,
    fpn_node_maps,
    pooled_roi_from_normalized_image,
)
from neurosym.hybrid import NeuroSymbolicDetector
from symbolic.evaluation import _FAITHFULNESS_CELL_BUDGET_FRACTION
from util.geometry import project_gt_box_to_roi_grid
from util.heatmap_metrics import (
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)

from .gradcam import GradCAM

_ROI_GRID = (7, 7)


def _neural_labels(model: NeuroFasterRCNN, pooled: Tensor) -> np.ndarray:
    """The detector's own class call for pooled RoI grids."""
    with torch.no_grad():
        logits = model.box_predictor.classifier(model.box_head(pooled))
    return logits.argmax(dim=1).cpu().numpy()


def evaluate_gradcam_faithfulness(
    model: NeuroFasterRCNN,
    gradcam: GradCAM,
    images: list[Tensor],
    score_threshold: float = 0.5,
    budget_fraction: float = _FAITHFULNESS_CELL_BUDGET_FRACTION,
    random_state: int = 42,
    auc_steps: int = 5,
) -> dict[str, Any]:
    """Necessity/sufficiency of Grad-CAM on Faster R-CNN's detections (>= threshold).

    Same image-pixel protocol as the tree side (`pixel_masking_metrics`): the CAM is the full
    FPN-level map upsampled over the whole image and ranked on the same tiles; "random" ranks
    the tiles at random. The label is the detector's own class call at the proposal and the
    score is the softmax probability of that class.
    """
    rng = np.random.default_rng(random_state)
    scores: dict[str, dict[str, list[float]]] = {
        name: {"necessity": [], "sufficiency": [], "deletion_auc": [], "insertion_auc": []}
        for name in ("gradcam", "gradcam_box", "random")
    }

    for image in tqdm(images, desc="Grad-CAM pixel-masking faithfulness"):
        with torch.inference_mode():
            detection = model([image.to(gradcam.device)])[0]
        kept = torch.where(detection["scores"] >= score_threshold)[0]
        if kept.numel() == 0:
            continue

        image = image.to(gradcam.device)
        mean, std = _normalization(model, image)
        normalized = (image - mean) / std
        image_shape = tuple(image.shape[-2:])
        grid_shape = (-(-image_shape[0] // PATCH_SIZE), -(-image_shape[1] // PATCH_SIZE))
        proposals = detection["proposal_boxes_processed"][kept]
        cams = gradcam.image_maps(image, proposals, detection["labels"][kept])
        # Tiles live in original pixels, the proposals in processed ones.
        processed_h, processed_w = model.transform([image], None)[0].image_sizes[0]
        to_original = torch.tensor(
            [image_shape[1] / processed_w, image_shape[0] / processed_h] * 2, device=proposals.device
        )

        for local in range(kept.numel()):
            box = proposals[local].to(gradcam.device)
            box_mask = box_tile_mask(box * to_original, grid_shape)
            box_tiles = int(box_mask.sum())
            budget = max(int(box_tiles * budget_fraction), 1)

            @memoize_by_keep_mask
            def logits_for(keep: np.ndarray) -> Tensor:
                masked = normalized * keep_mask_to_pixels(keep, image_shape).to(gradcam.device)
                with torch.no_grad():
                    pooled = pooled_roi_from_normalized_image(model, masked, box)[None]
                    return model.box_predictor.classifier(model.box_head(pooled))[0]

            base_label = int(logits_for(np.ones(grid_shape, dtype=bool)).argmax())

            def evaluate(keep: np.ndarray) -> tuple[int, float]:
                logits = logits_for(keep)
                return int(logits.argmax()), float(F.softmax(logits, dim=0)[base_label])

            # Same CAM, but only the proposal's tiles may be spent: separates "the CAM ranks the
            # box badly" from "the CAM puts its mass outside the box".
            in_box = np.where(box_mask, patch_scores(cams[local]), -np.inf)
            orders = {
                "gradcam": _tile_order(cams[local]),
                "gradcam_box": np.argsort(in_box.reshape(-1))[::-1],
                "random": random_order_box_first(rng, box_mask),
            }
            for name, order in orders.items():
                metrics = pixel_masking_metrics(evaluate, order, grid_shape, budget, box_tiles, auc_steps)
                for key, value in metrics.items():
                    scores[name][key].append(value)

    return {
        name: {
            "necessity_prediction_flip_rate": _mean(values["necessity"]),
            "sufficiency_prediction_preservation": _mean(values["sufficiency"]),
            "deletion_auc": _mean(values["deletion_auc"]),
            "insertion_auc": _mean(values["insertion_auc"]),
            "evaluated_roi_count": len(values["necessity"]),
        }
        for name, values in scores.items()
    }


def evaluate_shared_localization(
    hybrid_model: NeuroSymbolicDetector,
    gradcam: GradCAM,
    images: list[Tensor],
    targets: list[dict[str, Tensor]],
    min_proposal_iou: float = 0.05,
    max_proposal_iou: float = 0.35,
    random_state: int = 42,
) -> dict[str, Any]:
    """Pointing / IoU for SODT, Grad-CAM and random on one identical RoI list.

    Both models share the frozen backbone + RPN, so the proposals are the
    same. A RoI counts when its IoU to the truth sits in the loose window
    (tight boxes make pointing trivial) and both models call it a defect;
    each method explains its own model's class. RoIs where either map is
    empty are dropped for all three, so the lists stay identical.

    SODT's map is its overview contribution map M, cropped to the
    proposal and pooled to the 7x7 grid (`pixel_map_to_roi_grid`).
    """
    detector = hybrid_model.detector
    tree = hybrid_model.symbolic_tree
    rng = np.random.default_rng(random_state)
    results: dict[str, dict[str, list[float]]] = {
        name: {"overlap": [], "pointing": [], "coverage": []}
        for name in ("sodt", "gradcam", "random")
    }

    for image, target in tqdm(
        list(zip(images, targets)), desc="Shared-RoI localization"
    ):
        gt_boxes = target["boxes"].cpu()
        if gt_boxes.numel() == 0:
            continue

        with torch.inference_mode():
            images_list, _ = detector.transform([image.to(hybrid_model.device)], None)
            features = detector.backbone(images_list.tensors)
            proposals = detector.rpn(images_list, features, None)[0][0]

        processed_h, processed_w = images_list.image_sizes[0]
        original_h, original_w = image.shape[-2:]
        proposals_original = proposals.detach().cpu().float().clone()
        proposals_original[:, [0, 2]] *= original_w / processed_w
        proposals_original[:, [1, 3]] *= original_h / processed_h

        ious, matched = box_iou(proposals_original, gt_boxes).max(dim=1)
        window = (ious >= min_proposal_iou) & (ious <= max_proposal_iou)
        rows = torch.where(window)[0]
        if rows.numel() == 0:
            continue

        with torch.inference_mode():
            pooled = detector.roi_align(features, [proposals[rows]], images_list.image_sizes)
        pooled_np = pooled.detach().cpu().numpy().astype(np.float32)
        neural_labels = _neural_labels(detector, pooled)
        tree_labels = tree.predict(pooled_np.reshape(pooled_np.shape[0], -1))
        defect = (neural_labels > 0) & (tree_labels > 0)
        if not defect.any():
            continue

        kept = rows[torch.from_numpy(defect)]
        kept_local = np.where(defect)[0]
        gradcam_maps = gradcam.generate(
            image, proposals[kept], torch.from_numpy(neural_labels[kept_local])
        )
        image_on_device = image.to(hybrid_model.device)

        for local, row in enumerate(kept.tolist()):
            gt_mask = project_gt_box_to_roi_grid(
                proposals_original[row], gt_boxes[int(matched[row])], _ROI_GRID
            )
            if gt_mask.sum().item() == 0:
                continue

            pooled_row = kept_local[local]
            overview = fpn_node_maps(
                hybrid_model,
                image_on_device,
                proposals[row],
                path=tree.decision_path(pooled_np[pooled_row].reshape(-1)),
            )["overview_map"]
            sodt_map = pixel_map_to_roi_grid(overview, proposals_original[row])
            gradcam_map = normalize_heatmap(gradcam_maps[local])
            if sodt_map.sum() == 0 or gradcam_map.sum() == 0:
                continue

            random_map = normalize_heatmap(torch.from_numpy(rng.random(_ROI_GRID).astype(np.float32)))
            coverage = float(gt_mask.float().mean().item())
            for name, heatmap in (
                ("sodt", sodt_map),
                ("gradcam", gradcam_map),
                ("random", random_map),
            ):
                results[name]["overlap"].append(topk_region_overlap(heatmap, gt_mask))
                results[name]["pointing"].append(pointing_score(heatmap, gt_mask))
                results[name]["coverage"].append(coverage)

    return {
        name: stratified_spatial_result(values["overlap"], values["pointing"], values["coverage"])
        for name, values in results.items()
    }
