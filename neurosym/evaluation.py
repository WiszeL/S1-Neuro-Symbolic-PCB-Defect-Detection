"""Spatial-grounding and faithfulness evaluation for the exact path attribution
and the leaf-only map (see neurosym/heatmap.py).

Both need a live forward pass: the FPN map isn't part of the precomputed export
that symbolic/evaluation.py works from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm

from symbolic.evaluation import _compute_local_instance_heatmap, _FAITHFULNESS_CELL_BUDGET_FRACTION
from util.geometry import project_gt_box_to_roi_grid
from util.heatmap_metrics import (
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)

from .heatmap import _fpn_box_bounds, exact_fpn_contribution, path_weight_grid
from .hybrid import NeuroSymbolicDetector
from .inference import explain_hybrid_detection


@torch.no_grad()  # not inference_mode: the exact map runs autograd inside
def evaluate_exact_attribution_spatial_metrics(
    model: NeuroSymbolicDetector,
    images: list[Tensor],
    targets: list[dict[str, Tensor]],
    min_proposal_iou: float = 0.0,
    max_proposal_iou: float = 1.0,
) -> dict[str, Any]:
    """Pointing/IoU spatial metrics for the exact path attribution, same schema
    and GT projection as evaluate_symbolic_spatial_metrics / evaluate_gradcam,
    so the three are directly comparable in one table.
    """
    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    gt_coverage: list[float] = []

    for image, target in tqdm(
        list(zip(images, targets)), desc="Exact attribution spatial grounding"
    ):
        gt_boxes = target["boxes"]
        if gt_boxes.numel() == 0:
            continue

        detection = model([image])[0]
        proposals = detection["proposal_boxes"]
        if proposals.shape[0] == 0:
            continue

        ious = box_iou(proposals, gt_boxes)
        matched_iou, matched_idx = ious.max(dim=1)

        for row in range(proposals.shape[0]):
            iou = float(matched_iou[row])
            if not (min_proposal_iou <= iou <= max_proposal_iou):
                continue

            gt_box = gt_boxes[int(matched_idx[row])]
            gt_mask = project_gt_box_to_roi_grid(proposals[row], gt_box, (7, 7))
            if gt_mask.sum().item() == 0:
                continue

            explanation = explain_hybrid_detection(
                model,
                detection,
                detection_index=row,
                image_shape=tuple(image.shape[-2:]),
            )
            if "path_exact_attribution" not in explanation:
                # detection wasn't produced by NeuroSymbolicDetector.forward
                # (no FPN context) — nothing to score.
                continue

            # The exact map is at FPN resolution; the GT mask is 7x7. Resample
            # so both are on the grid util/heatmap_metrics.py expects.
            attribution = torch.from_numpy(explanation["path_exact_attribution"]).float()
            heatmap = normalize_heatmap(
                F.interpolate(
                    attribution[None, None], size=(7, 7), mode="bilinear", align_corners=False
                )[0, 0]
            )
            if heatmap.sum() == 0:
                continue

            overlap_scores.append(topk_region_overlap(heatmap, gt_mask))
            pointing_scores.append(pointing_score(heatmap, gt_mask))
            gt_coverage.append(float(gt_mask.float().mean().item()))

    return stratified_spatial_result(overlap_scores, pointing_scores, gt_coverage)


def _rank_positions_fpn(
    ranking: str,
    tree,
    pooled_grid: np.ndarray,
    level_map: Tensor,
    box_bounds: tuple[int, int, int, int],
    rng: np.random.Generator,
    fpn_map: Tensor | None = None,
) -> np.ndarray:
    """Return flat position indices (within the box's FPN sub-grid), ranked
    most-important first, under the given ranking method.

    `fpn_map` is a ready `(Hf, Wf)` importance map (exact attribution, or a
    control) computed by the caller — this helper only crops and ranks it. The
    `leaf_only` and `random` rankings are self-contained.
    """
    fx1, fy1, fx2, fy2 = box_bounds
    bh, bw = fy2 - fy1, fx2 - fx1

    if fpn_map is not None:
        scores = fpn_map[fy1:fy2, fx1:fx2].detach().cpu().numpy()
    elif ranking == "leaf_only":
        leaf7 = _compute_local_instance_heatmap(tree, torch.from_numpy(pooled_grid), mode="leaf_only")
        scores = F.interpolate(
            leaf7[None, None].float(), size=(bh, bw), mode="bilinear", align_corners=False
        )[0, 0].numpy()
    elif ranking == "random":
        scores = rng.random((bh, bw)).astype(np.float32)
    else:
        raise ValueError(f"Unknown ranking {ranking!r}.")

    return np.argsort(scores.reshape(-1))[::-1]


@torch.no_grad()  # not inference_mode: the exact map runs autograd inside
def evaluate_faithfulness_fpn_masking(
    model: NeuroSymbolicDetector,
    images: list[Tensor],
    rankings: tuple[str, ...] = ("leaf_only", "exact", "random"),
    budget_fraction: float = _FAITHFULNESS_CELL_BUDGET_FRACTION,
    margin: int = 2,
    max_rois_per_image: int = 8,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Necessity test at the FPN's native resolution: mask the pixels a map
    ranks highest, re-pool through `model.detector.roi_align`, re-run the tree,
    and see how often the prediction flips.

    The 7x7-grid masking in symbolic/evaluation.py handicaps a finer-resolution
    map by shrinking it to 7x7 first; masking the FPN map directly does not.
    `shuffled_w` (permuted path weights) and `activation_only` (FPN magnitude,
    no tree) are opt-in controls — see WHAT-I-DID.md §8 for the numbers.
    Evaluates the top `max_rois_per_image` raw RPN proposals per image, no GT
    filter.
    """
    tree = model.symbolic_tree
    detector = model.detector
    rng = np.random.default_rng(random_state)
    results = {name: [] for name in rankings}
    featmap_names = list(detector.roi_align.pool.featmap_names)
    map_rankings = {"exact", "shuffled_w", "activation_only"}

    for image in tqdm(images, desc="FPN-masking faithfulness"):
        images_list, _ = detector.transform([image.to(model.device)], None)
        fpn_features = detector.backbone(images_list.tensors)
        proposals_per_image, _ = detector.rpn(images_list, fpn_features, None)
        boxes_processed = proposals_per_image[0][:max_rois_per_image]
        if boxes_processed.shape[0] == 0:
            continue

        pooled = detector.roi_align(fpn_features, [boxes_processed], images_list.image_sizes)
        level_indices = detector.roi_align.pool.map_levels([boxes_processed])
        padded_size = tuple(images_list.tensors.shape[-2:])
        processed_size = images_list.image_sizes[0]
        # exact_fpn_contribution expects unbatched [C, H, W] levels, matching
        # what NeuroSymbolicDetector.forward stores on a detection dict.
        unbatched_fpn = {name: level[0].detach() for name, level in fpn_features.items()}

        for row in range(boxes_processed.shape[0]):
            box_processed = boxes_processed[row]
            level_name = featmap_names[int(level_indices[row])]
            level_map = fpn_features[level_name][0]

            bounds = _fpn_box_bounds(
                box_processed, padded_size, tuple(level_map.shape[-2:]), margin=margin
            )
            fx1, fy1, fx2, fy2 = bounds
            if fx2 <= fx1 or fy2 <= fy1:
                continue

            pooled_grid = pooled[row].detach().cpu().numpy().astype(np.float32)
            base_pred = int(tree.predict_proba(pooled_grid.reshape(1, -1)).argmax())

            n_pos = (fy2 - fy1) * (fx2 - fx1)
            budget = max(int(n_pos * budget_fraction), 1)

            # Importance maps at FPN resolution — computed once per RoI.
            fpn_maps: dict[str, Tensor] = {}
            if map_rankings & set(rankings):
                base_weights = path_weight_grid(tree, pooled_grid)
                if "exact" in rankings:
                    fpn_maps["exact"] = exact_fpn_contribution(
                        tree, pooled_grid, detector.roi_align, unbatched_fpn,
                        level_name, box_processed, processed_size,
                    ).abs().sum(0)
                if "shuffled_w" in rankings:
                    shuffled = base_weights.reshape(-1).copy()
                    rng.shuffle(shuffled)
                    fpn_maps["shuffled_w"] = exact_fpn_contribution(
                        tree, pooled_grid, detector.roi_align, unbatched_fpn,
                        level_name, box_processed, processed_size,
                        weight_grid_override=shuffled.reshape(base_weights.shape),
                    ).abs().sum(0)
                if "activation_only" in rankings:
                    fpn_maps["activation_only"] = level_map.detach().abs().sum(0)

            for name in rankings:
                order = _rank_positions_fpn(
                    name, tree, pooled_grid, level_map, bounds, rng,
                    fpn_map=fpn_maps.get(name),
                )
                sel = np.zeros(n_pos, dtype=bool)
                sel[order[:budget]] = True
                sel_t = torch.from_numpy(sel.reshape(fy2 - fy1, fx2 - fx1)).to(model.device)

                nec_level = level_map.clone()
                nec_level[:, fy1:fy2, fx1:fx2][:, sel_t] = 0.0
                nec_fpn = dict(fpn_features)
                nec_fpn[level_name] = nec_level.unsqueeze(0)
                nec_grid = detector.roi_align(
                    nec_fpn, [box_processed.unsqueeze(0)], [processed_size]
                )[0]
                nec_pred = int(
                    tree.predict_proba(
                        nec_grid.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                    ).argmax()
                )
                results[name].append(float(nec_pred != base_pred))

    return {
        name: {
            "necessity_prediction_flip_rate": float(np.mean(flips)) if flips else float("nan"),
            "evaluated_roi_count": len(flips),
        }
        for name, flips in results.items()
    }
