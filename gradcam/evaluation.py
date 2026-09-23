"""Grad-CAM's report card, scored exactly like the tree's.

- Faithfulness: same FPN-masking protocol (`fpn_necessity_sufficiency`),
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
from neurosym.evaluation import _mean, fpn_deletion_insertion_auc, fpn_necessity_sufficiency
from neurosym.heatmap import (
    _fpn_box_bounds,
    compute_exact_attribution,
    compute_node_local_evidence_maps,
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
    margin: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Necessity/sufficiency of Grad-CAM on Faster R-CNN's detections (>= threshold).

    Same masking as the tree side; "random" ranks the same boxes at random.
    """
    rng = np.random.default_rng(random_state)
    scores: dict[str, dict[str, list[float]]] = {
        name: {"necessity": [], "sufficiency": [], "deletion_auc": [], "insertion_auc": []}
        for name in ("gradcam", "random")
    }

    def classify(grid: Tensor) -> np.ndarray:
        return _neural_labels(model, grid)

    def class_probability(grid: Tensor, label: int) -> float:
        # Grad-CAM's confidence: probability of the detected class.
        with torch.no_grad():
            logits = model.box_predictor.classifier(model.box_head(grid))
        return float(F.softmax(logits, dim=1)[0, label])

    for image in tqdm(images, desc="Grad-CAM FPN-masking faithfulness"):
        with torch.inference_mode():
            detection = model([image.to(gradcam.device)])[0]
        keep = detection["scores"] >= score_threshold
        proposals = detection["proposal_boxes_processed"][keep]
        labels = detection["labels"][keep]
        if proposals.shape[0] == 0:
            continue

        maps = gradcam.fpn_maps(image, proposals, labels)
        for row, (cam, level_name) in enumerate(zip(maps["cams"], maps["level_names"])):
            box = maps["boxes_processed"][row]
            bounds = _fpn_box_bounds(box, maps["padded_size"], tuple(cam.shape), margin=margin)
            fx1, fy1, fx2, fy2 = bounds
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            n_pos = (fy2 - fy1) * (fx2 - fx1)
            budget = max(int(n_pos * budget_fraction), 1)
            base_grid = maps["pooled_features"][row : row + 1]
            orders = {
                "gradcam": np.argsort(cam[fy1:fy2, fx1:fx2].cpu().numpy().reshape(-1))[::-1],
                "random": rng.permutation(n_pos),
            }
            for name, order in orders.items():
                flipped, preserved = fpn_necessity_sufficiency(
                    model.roi_align, maps["fpn_features"], level_name, box,
                    maps["processed_size"], bounds, order, budget, classify,
                    int(labels[row]),
                )
                scores[name]["necessity"].append(flipped)
                scores[name]["sufficiency"].append(preserved)
                del_auc, ins_auc = fpn_deletion_insertion_auc(
                    model.roi_align, maps["fpn_features"], level_name, box,
                    maps["processed_size"], bounds, order, base_grid,
                    lambda g: class_probability(g, int(labels[row])),
                )
                scores[name]["deletion_auc"].append(del_auc)
                scores[name]["insertion_auc"].append(ins_auc)

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


def _to_grid(heatmap: np.ndarray | Tensor) -> Tensor:
    """Resample a proposal-cropped map to the 7x7 grid the shared metrics expect."""
    tensor = torch.as_tensor(heatmap, dtype=torch.float32)
    return normalize_heatmap(
        F.interpolate(tensor[None, None], size=_ROI_GRID, mode="bilinear", align_corners=False)[0, 0]
    )


def evaluate_shared_localization(
    hybrid_model: NeuroSymbolicDetector,
    gradcam: GradCAM,
    images: list[Tensor],
    targets: list[dict[str, Tensor]],
    min_proposal_iou: float = 0.05,
    max_proposal_iou: float = 0.35,
    random_state: int = 42,
    use_fpn_heatmap: bool = True,
) -> dict[str, Any]:
    """Pointing / IoU for SODT, Grad-CAM and random on one identical RoI list.

    Both models share the frozen backbone + RPN, so the proposals are the
    same. A RoI counts when its IoU to the truth sits in the loose window
    (tight boxes make pointing trivial) and both models call it a defect;
    each method explains its own model's class. RoIs where either map is
    empty are dropped for all three, so the lists stay identical.

    `use_fpn_heatmap=False`: SODT's map is the displayed 7x7 grid heatmap
    (`heatmap.compute_node_local_evidence_maps`'s `raw_node_heatmap`, summed
    over the path) instead of the exact FPN attribution — the resolution
    ablation used elsewhere in `neurosym.evaluation`.
    """
    detector = hybrid_model.detector
    tree = hybrid_model.symbolic_tree
    featmap_names = list(detector.roi_align.pool.featmap_names)
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
            level_indices = detector.roi_align.pool.map_levels([proposals[rows]])
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
        unbatched_fpn = {name: level[0].detach() for name, level in features.items()}
        padded_size = tuple(images_list.tensors.shape[-2:])
        processed_size = tuple(images_list.image_sizes[0])

        for local, row in enumerate(kept.tolist()):
            gt_mask = project_gt_box_to_roi_grid(
                proposals_original[row], gt_boxes[int(matched[row])], _ROI_GRID
            )
            if gt_mask.sum().item() == 0:
                continue

            pooled_row = kept_local[local]
            if use_fpn_heatmap:
                sodt_map = _to_grid(
                    compute_exact_attribution(
                        tree,
                        pooled_np[pooled_row],
                        detector.roi_align,
                        unbatched_fpn,
                        featmap_names[int(level_indices[pooled_row])],
                        proposals[row],
                        processed_size,
                        padded_size,
                    )
                )
            else:
                # The 7x7 map shown when use_fpn_heatmap=False; already on the proposal grid.
                sodt_map = normalize_heatmap(
                    sum(
                        node["raw_node_heatmap"]
                        for node in compute_node_local_evidence_maps(tree, pooled_np[pooled_row])
                    )
                )
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
