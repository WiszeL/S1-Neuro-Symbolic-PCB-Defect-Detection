"""Heatmap scores for the FPN-resolution maps (see heatmap.py)."""

from __future__ import annotations

from collections import defaultdict
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

from .heatmap import _fpn_box_bounds, exact_fpn_contribution, path_fpn_map, path_weight_grid
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
    """Same pointing/IoU table as the other two methods, for direct comparison."""
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
                # No FPN context — nothing to score.
                continue

            # Resample to the 7x7 grid the shared metrics expect.
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
    """Rank box positions most-important first, under the given ranking."""
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


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _masked_level(
    level_map: Tensor,
    bounds: tuple[int, int, int, int],
    keep_mask: Tensor,
    start_from_zero: bool = False,
) -> Tensor:
    """`level_map` with the box region reduced to only `keep_mask` positions.

    `start_from_zero=False` (deletion): full map, then zero the box positions
    NOT in `keep_mask` — i.e. `keep_mask` marks what survives.
    `start_from_zero=True` (insertion): empty map, then copy in the box
    positions that ARE in `keep_mask`.
    """
    fx1, fy1, fx2, fy2 = bounds
    if start_from_zero:
        out = torch.zeros_like(level_map)
        out[:, fy1:fy2, fx1:fx2][:, keep_mask] = level_map[:, fy1:fy2, fx1:fx2][:, keep_mask]
    else:
        out = level_map.clone()
        out[:, fy1:fy2, fx1:fx2][:, ~keep_mask] = 0.0
    return out


def _node_score_after_masking(
    detector,
    fpn_features: dict[str, Tensor],
    level_name: str,
    masked_level: Tensor,
    box_processed: Tensor,
    processed_size: tuple[int, int],
    weight: np.ndarray,
    bias: float,
) -> float:
    """Re-pool with one FPN level replaced, then re-score this single node
    directly (`w . x + b`) — cheaper and more exact than re-running the whole
    tree, since the node's own weights/bias never change."""
    patched = dict(fpn_features)
    patched[level_name] = masked_level.unsqueeze(0)
    grid = detector.roi_align(patched, [box_processed.unsqueeze(0)], [processed_size])[0]
    features = grid.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return float((features * weight).sum() + bias)


@torch.no_grad()  # not inference_mode: the exact map runs autograd inside
def evaluate_faithfulness_fpn_masking(
    model: NeuroSymbolicDetector,
    images: list[Tensor],
    rankings: tuple[str, ...] = ("exact", "random"),
    budget_fraction: float = _FAITHFULNESS_CELL_BUDGET_FRACTION,
    margin: int = 2,
    max_rois_per_image: int = 8,
    random_state: int = 42,
    per_node: bool = True,
    auc_steps: int = 5,
) -> dict[str, Any]:
    """Necessity test at full FPN resolution: mask top pixels, re-pool, check flips.

    Masking the FPN map directly (not shrunk to 7x7 first) is the fair test
    for a finer-resolution map. Two things are measured, at the SAME
    `rankings`/`budget_fraction`:

    - **Path-level** (`return["path"]`): mask the path-level map (the
      per-node panels stacked, Σ|node map| — `heatmap.path_fpn_map`), re-pool,
      check whether the FINAL LABEL flips. "Do the regions the steps weigh,
      together, matter to the decision" — the head-to-head vs Grad-CAM.
    - **Per-node** (`return["node"]`, `per_node=True`, `name in {"exact",
      "random"}`): for each node on the path, mask that NODE'S OWN map,
      re-pool, check whether that NODE'S OWN routing sign
      (`w_i . x + b_i >= 0`) flips, plus a deletion/insertion AUC over that
      node's routing confidence `sigmoid(|w_i . x + b_i|)` — same quantity
      `NeuroSymbolicDetector` uses for routing-margin scoring (`hybrid.py`),
      tracked per masking budget instead of once. "Does the region THAT
      NODE weighs decide THAT NODE's question" — the per-node panels' claim.

    `return["node_map_similarity"]`: mean cosine similarity between
    consecutive nodes' exact maps along the path — evidence the six steps
    weigh different regions (high similarity would mean nothing to show).

    Cost note: per-node is ~2 orders of magnitude more re-pools than
    path-level alone (depth * rankings * masking steps, per RoI). Time
    `images[:5]` before picking a subsample size.
    """
    tree = model.symbolic_tree
    detector = model.detector
    rng = np.random.default_rng(random_state)
    path_results = {name: [] for name in rankings}
    node_rankings = tuple(name for name in rankings if name in ("exact", "random"))
    node_results: dict[str, dict[int, dict[str, list[float]]]] = {
        name: defaultdict(
            lambda: {"flip": [], "deletion_auc": [], "insertion_auc": [], "support_fraction": []}
        )
        for name in node_rankings
    }
    node_similarity: list[float] = []
    node_ceiling: dict[int, list[float]] = defaultdict(list)
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
        # Forward stores unbatched levels on each detection dict.
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
            base_pred = int(tree.predict(pooled_grid.reshape(1, -1))[0])

            n_pos = (fy2 - fy1) * (fx2 - fx1)
            budget = max(int(n_pos * budget_fraction), 1)

            # ── Path-level (unchanged): does the final label flip? ──
            fpn_maps: dict[str, Tensor] = {}
            if map_rankings & set(rankings):
                # Path map = stacked per-node panels (Σ|node map|), never a signed sum.
                path_steps = tree.decision_path(pooled_grid.reshape(-1))
                if "exact" in rankings:
                    fpn_maps["exact"] = path_fpn_map(
                        tree, pooled_grid, detector.roi_align, unbatched_fpn,
                        level_name, box_processed, processed_size, path=path_steps,
                    )
                if "shuffled_w" in rankings:
                    # Same stacking, each node's own weights shuffled in place.
                    shuffled_grids = []
                    for step in path_steps:
                        node_grid = path_weight_grid(tree, pooled_grid, path=[step])
                        flat = node_grid.reshape(-1).copy()
                        rng.shuffle(flat)
                        shuffled_grids.append(flat.reshape(node_grid.shape))
                    fpn_maps["shuffled_w"] = path_fpn_map(
                        tree, pooled_grid, detector.roi_align, unbatched_fpn,
                        level_name, box_processed, processed_size, path=path_steps,
                        node_weight_grids=shuffled_grids,
                    )
                if "activation_only" in rankings:
                    fpn_maps["activation_only"] = level_map.detach().abs().sum(0)

            for name in rankings:
                order = _rank_positions_fpn(
                    name, tree, pooled_grid, level_map, bounds, rng,
                    fpn_map=fpn_maps.get(name),
                )
                sel = np.zeros(n_pos, dtype=bool)
                sel[order[:budget]] = True
                keep = ~sel.reshape(fy2 - fy1, fx2 - fx1)
                keep_t = torch.from_numpy(keep).to(model.device)

                nec_level = _masked_level(level_map, bounds, keep_t)
                nec_fpn = dict(fpn_features)
                nec_fpn[level_name] = nec_level.unsqueeze(0)
                nec_grid = detector.roi_align(
                    nec_fpn, [box_processed.unsqueeze(0)], [processed_size]
                )[0]
                nec_pred = int(
                    tree.predict(
                        nec_grid.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                    )[0]
                )
                path_results[name].append(float(nec_pred != base_pred))

            # ── Per-node: does THIS node's own routing call flip? ──
            if per_node and node_rankings:
                path = tree.decision_path(pooled_grid.reshape(-1))
                exact_node_maps: list[np.ndarray] = []

                for depth, step in enumerate(path):
                    weight = tree.node_weights[step.node_index]
                    bias = float(tree.node_bias[step.node_index])
                    if not np.any(weight != 0.0):
                        # Pruned node: masking can't change anything, so it'd
                        # just pad the average with fake agreement. Skip it
                        # (same call as hybrid.py's routing-margin).
                        continue

                    # Ceiling for "exact" specifically: if its top-ranked
                    # pixels cover the node's whole nonzero support (sparse
                    # node, common under L1), masking collapses the score to
                    # bias alone, so flip needs sign(bias) != sign(score).
                    # NOT a bound for "random" — a random 50% subset can flip
                    # via unrelated cancellation even past this. Verified
                    # against a dense-weight synthetic tree, where random's
                    # flip rate did exceed this number.
                    bias_blocks_flip = (bias >= 0.0) == step.went_left
                    node_ceiling[depth].append(float(not bias_blocks_flip))

                    # Ranking-independent (same node, same box either way) —
                    # computed once, not once per ranking. Zero features in
                    # is exactly `bias` out (locked by
                    # test_faithfulness_protocol.py) — no re-pool needed.
                    base_conf = 1.0 / (1.0 + np.exp(-abs(step.score)))
                    empty_conf = 1.0 / (1.0 + np.exp(-abs(bias)))
                    box_shape = (fy2 - fy1, fx2 - fx1)

                    def conf_ratio(sel_flat: np.ndarray, *, insert: bool) -> float:
                        """Confidence after keeping/masking `sel_flat`, over unmasked.

                        Reuses `weight`/`bias` from the enclosing loop — a
                        node's own weights never change while masking it.
                        """
                        keep = sel_flat.reshape(box_shape) if insert else ~sel_flat.reshape(box_shape)
                        level = _masked_level(
                            level_map, bounds, torch.from_numpy(keep).to(model.device),
                            start_from_zero=insert,
                        )
                        score = _node_score_after_masking(
                            detector, fpn_features, level_name, level,
                            box_processed, processed_size, weight, bias,
                        )
                        return (1.0 / (1.0 + np.exp(-abs(score)))) / base_conf

                    for name in node_rankings:
                        if name == "exact":
                            node_map_full = exact_fpn_contribution(
                                tree, pooled_grid, detector.roi_align, unbatched_fpn,
                                level_name, box_processed, processed_size, path=[step],
                            ).abs().sum(0)
                            node_map_box = node_map_full[fy1:fy2, fx1:fx2].detach().cpu().numpy()
                            exact_node_maps.append(node_map_box)
                            order = np.argsort(node_map_box.reshape(-1))[::-1]
                            # Fraction of box positions with any contribution
                            # at all — the ceiling above is only tight for
                            # "exact" when this is <= budget_fraction (its
                            # top-ranked pixels then cover the whole nonzero
                            # support); above that, the ceiling is loose.
                            support_fraction = float((node_map_box > 0).mean())
                            node_results[name][depth]["support_fraction"].append(support_fraction)
                        else:  # "random"
                            order = rng.permutation(n_pos)

                        sel = np.zeros(n_pos, dtype=bool)
                        sel[order[:budget]] = True
                        nec_score = _node_score_after_masking(
                            detector, fpn_features, level_name,
                            _masked_level(
                                level_map, bounds,
                                torch.from_numpy(~sel.reshape(box_shape)).to(model.device),
                            ),
                            box_processed, processed_size, weight, bias,
                        )
                        flipped = (nec_score >= 0.0) != step.went_left
                        bucket = node_results[name][depth]
                        bucket["flip"].append(float(flipped))

                        # Endpoints (frac=0 and frac=1) are the same two
                        # identities as above — no re-pool for those either.
                        del_curve = [1.0]
                        ins_curve = [empty_conf / base_conf]
                        for s in range(1, auc_steps):
                            frac = s / auc_steps
                            k = min(max(int(round(n_pos * frac)), 0), n_pos)
                            sel_k = np.zeros(n_pos, dtype=bool)
                            sel_k[order[:k]] = True
                            del_curve.append(conf_ratio(sel_k, insert=False))
                            ins_curve.append(conf_ratio(sel_k, insert=True))
                        del_curve.append(empty_conf / base_conf)
                        ins_curve.append(1.0)

                        dx = 1.0 / auc_steps
                        bucket["deletion_auc"].append(float(np.trapz(del_curve, dx=dx)))
                        bucket["insertion_auc"].append(float(np.trapz(ins_curve, dx=dx)))

                if "exact" in node_rankings and len(exact_node_maps) >= 2:
                    for map_a, map_b in zip(exact_node_maps[:-1], exact_node_maps[1:]):
                        flat_a, flat_b = map_a.reshape(-1), map_b.reshape(-1)
                        denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
                        if denom > 1e-8:
                            node_similarity.append(float(np.dot(flat_a, flat_b) / denom))

    result: dict[str, Any] = {
        "path": {
            name: {
                "necessity_prediction_flip_rate": _mean(flips),
                "evaluated_roi_count": len(flips),
            }
            for name, flips in path_results.items()
        }
    }
    if per_node:
        result["node"] = {
            name: {
                depth: {
                    "necessity_prediction_flip_rate": _mean(bucket["flip"]),
                    # Curve is routing confidence, not class probability —
                    # don't put this next to Grad-CAM's AUC in one table.
                    "deletion_auc": _mean(bucket["deletion_auc"]),
                    "insertion_auc": _mean(bucket["insertion_auc"]),
                    # "random" has no map of its own — nan. For "exact",
                    # whether the ceiling above is tight: <= budget_fraction
                    # means its ranking covers the node's whole nonzero
                    # support, so the ceiling applies; above that, it's loose.
                    "support_fraction": _mean(bucket["support_fraction"]),
                    "evaluated_roi_count": len(bucket["flip"]),
                }
                for depth, bucket in sorted(node_results[name].items())
            }
            for name in node_rankings
        }
        # Ceiling for "exact" only (see comment above where it's built) — not
        # "random". Compare "exact" against this, not against "random".
        result["node_necessity_ceiling"] = {
            depth: _mean(flips) for depth, flips in sorted(node_ceiling.items())
        }
        result["node_map_similarity"] = {
            "mean_cosine_similarity_between_consecutive_nodes": _mean(node_similarity),
            "evaluated_pair_count": len(node_similarity),
        }
    return result
