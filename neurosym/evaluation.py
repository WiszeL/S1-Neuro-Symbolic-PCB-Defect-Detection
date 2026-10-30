"""Heatmap scores for the exact pixel maps (see heatmap.py)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm

from symbolic.evaluation import _FAITHFULNESS_CELL_BUDGET_FRACTION
from util.geometry import project_gt_box_to_roi_grid
from util.heatmap_metrics import (
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)

from .heatmap import (
    _normalization,
    path_scores,
    fpn_node_maps,
    pooled_roi_from_normalized_image,
)
from .hybrid import NeuroSymbolicDetector

_ROI_GRID = (7, 7)


def pixel_map_to_roi_grid(pixel_map: np.ndarray, box: Tensor) -> Tensor:
    """Contribution map cropped to a box (original pixels) and pooled to 7x7."""
    height, width = pixel_map.shape
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    crop = pixel_map[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)]
    if crop.size == 0:
        return torch.zeros(_ROI_GRID)
    pooled = F.adaptive_avg_pool2d(torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None, None], _ROI_GRID)
    return normalize_heatmap(pooled[0, 0])


@torch.no_grad()  # not inference_mode: the pixel map runs autograd inside
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
            gt_mask = project_gt_box_to_roi_grid(proposals[row], gt_box, _ROI_GRID)
            if gt_mask.sum().item() == 0:
                continue

            path = model.symbolic_tree.decision_path(
                detection["pooled_features"][row].numpy().reshape(-1)
            )
            overview = fpn_node_maps(
                model,
                image.to(model.device),
                detection["proposal_boxes_processed"][row],
                path=path,
            )["overview_map"]
            heatmap = pixel_map_to_roi_grid(overview, proposals[row])
            if heatmap.sum() == 0:
                continue

            overlap_scores.append(topk_region_overlap(heatmap, gt_mask))
            pointing_scores.append(pointing_score(heatmap, gt_mask))
            gt_coverage.append(float(gt_mask.float().mean().item()))

    return stratified_spatial_result(overlap_scores, pointing_scores, gt_coverage)


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# Image-pixel faithfulness. Candidates are PATCH_SIZE tiles over the whole image (the
# receptive field reaches outside the box); a tile is ranked by its summed map, which keeps the
# ranking exact for an additive map. Masking sets a tile to the mean colour (0 in x̃ units).
PATCH_SIZE = 16


def patch_scores(pixel_map: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    """Sum of a pixel map over every `patch` tile; edge tiles are zero-padded."""
    height, width = pixel_map.shape
    rows, cols = -(-height // patch), -(-width // patch)
    padded = np.zeros((rows * patch, cols * patch), dtype=np.float64)
    padded[:height, :width] = pixel_map
    return padded.reshape(rows, patch, cols, patch).sum(axis=(1, 3))


def keep_mask_to_pixels(keep: np.ndarray, image_shape: tuple[int, int]) -> Tensor:
    tiles = np.repeat(np.repeat(keep, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    return torch.from_numpy(tiles[: image_shape[0], : image_shape[1]])


def box_tile_mask(box: Tensor, grid_shape: tuple[int, int]) -> np.ndarray:
    """Tiles the box (original pixels) touches, as a [rows, cols] bool grid; at least one."""
    x1, y1, x2, y2 = (float(v) for v in box)
    first_col = min(max(int(x1 // PATCH_SIZE), 0), grid_shape[1] - 1)
    first_row = min(max(int(y1 // PATCH_SIZE), 0), grid_shape[0] - 1)
    last_col = max(min(int((x2 - 1e-6) // PATCH_SIZE), grid_shape[1] - 1), first_col)
    last_row = max(min(int((y2 - 1e-6) // PATCH_SIZE), grid_shape[0] - 1), first_row)
    mask = np.zeros(grid_shape, dtype=bool)
    mask[first_row : last_row + 1, first_col : last_col + 1] = True
    return mask


def random_order_box_first(rng: np.random.Generator, box_mask: np.ndarray) -> np.ndarray:
    """Random floor: the box's tiles in random order, then the rest. Whole-image random would
    almost never touch the box and tell nothing."""
    flat = box_mask.reshape(-1)
    return np.concatenate([rng.permutation(np.flatnonzero(flat)), rng.permutation(np.flatnonzero(~flat))])


def memoize_by_keep_mask(function: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
    """Endpoints (nothing / everything kept) repeat across rankings; one forward each is enough."""
    cache: dict[bytes, Any] = {}

    def wrapped(keep: np.ndarray) -> Any:
        key = keep.tobytes()
        if key not in cache:
            cache[key] = function(keep)
        return cache[key]

    return wrapped


def pixel_masking_metrics(
    evaluate: Callable[[np.ndarray], tuple[int, float]],
    order: np.ndarray,
    grid_shape: tuple[int, int],
    budget: int,
    box_tiles: int,
    auc_steps: int,
) -> dict[str, float]:
    """One detection, one ranking. `evaluate(keep)` -> (label, score) with only `keep` tiles unmasked.

    Necessity masks the top `budget` tiles and asks whether the label flips; sufficiency keeps
    only them and asks whether it survives. The AUCs follow `score / base score` as the top
    0 .. `box_tiles` tiles are masked (deletion) or added to an empty image (insertion); the
    schedule runs over the box's tile count, not the image's, and both ends are evaluated.
    Every method goes through this; only `order` (its own map) and `evaluate` (its own model)
    differ.
    """
    base_label, base_score = evaluate(np.ones(grid_shape, dtype=bool))
    base_score = max(base_score, 1e-12)

    def selection(count: int) -> np.ndarray:
        selected = np.zeros(grid_shape[0] * grid_shape[1], dtype=bool)
        selected[order[:count]] = True
        return selected.reshape(grid_shape)

    top = selection(budget)
    deletion, insertion = [], []
    for step in range(auc_steps + 1):
        selected = selection(round(box_tiles * step / auc_steps))
        deletion.append(evaluate(~selected)[1] / base_score)
        insertion.append(evaluate(selected)[1] / base_score)

    dx = 1.0 / auc_steps
    return {
        "necessity": float(evaluate(~top)[0] != base_label),
        "sufficiency": float(evaluate(top)[0] == base_label),
        "deletion_auc": float(np.trapz(deletion, dx=dx)),
        "insertion_auc": float(np.trapz(insertion, dx=dx)),
    }


def _tile_order(pixel_map: np.ndarray) -> np.ndarray:
    return np.argsort(patch_scores(pixel_map).reshape(-1))[::-1]


@torch.no_grad()  # not inference_mode: the pixel map runs autograd inside
def evaluate_pixel_faithfulness(
    model: NeuroSymbolicDetector,
    images: list[Tensor],
    rankings: tuple[str, ...] = ("exact", "random"),
    budget_fraction: float = _FAITHFULNESS_CELL_BUDGET_FRACTION,
    score_threshold: float = 0.5,
    random_state: int = 42,
    per_node: bool = False,
    auc_steps: int = 5,
) -> dict[str, Any]:
    """Necessity/sufficiency on image pixels: mask the top-ranked tiles, re-run the real model.

    Population: the hybrid's detections scoring >= `score_threshold`, each at the proposal it
    was classified from (the proposal stays fixed under masking). Masked images go through the
    real forward pass.

    - **Path-level** (`return["path"]`): rank tiles by the overview map M = Σ_i H_i ("exact", most
      contributing first) or at random; the score is the routing confidence of the final leaf and
      the label is its class.
    - **Per-node** (`return["node"]`, `per_node=True`): each node's own map against that node's own
      call: its routing sign flips (necessity), and σ(|w_i·x + b_i|) is tracked as tiles go.
      Pruned nodes are skipped. Off by default: it is most of the cost (~6 extra rankings per
      detection, each a full set of forward passes).
    """
    tree = model.symbolic_tree
    detector = model.detector
    rng = np.random.default_rng(random_state)
    path_results = {
        name: {"necessity": [], "sufficiency": [], "deletion_auc": [], "insertion_auc": []}
        for name in rankings
    }
    node_results: dict[str, dict[int, dict[str, list[float]]]] = {
        name: defaultdict(lambda: {"necessity": [], "sufficiency": [], "deletion_auc": [], "insertion_auc": []})
        for name in ("exact", "random")
    }
    node_similarity: list[float] = []

    for image in tqdm(images, desc="Pixel-masking faithfulness"):
        detection = model([image])[0]
        kept = torch.where(detection["scores"] >= score_threshold)[0].tolist()
        if not kept:
            continue

        image = image.to(model.device)
        mean, std = _normalization(detector, image)
        normalized = (image - mean) / std
        image_shape = tuple(image.shape[-2:])
        grid_shape = (-(-image_shape[0] // PATCH_SIZE), -(-image_shape[1] // PATCH_SIZE))

        for row in kept:
            box = detection["proposal_boxes_processed"][row].to(model.device)
            path = tree.decision_path(detection["pooled_features"][row].numpy().reshape(-1))
            attribution = fpn_node_maps(model, image, box, path=path)
            box_mask = box_tile_mask(detection["proposal_boxes"][row], grid_shape)
            box_tiles = int(box_mask.sum())
            budget = max(int(box_tiles * budget_fraction), 1)

            @memoize_by_keep_mask
            def pooled_for(keep: np.ndarray) -> Tensor:
                masked = normalized * keep_mask_to_pixels(keep, image_shape).to(model.device)
                return pooled_roi_from_normalized_image(detector, masked, box)

            def path_evaluate(keep: np.ndarray) -> tuple[int, float]:
                grid = pooled_for(keep).detach().cpu().numpy().astype(np.float32).reshape(1, -1)
                leaf, confidence = tree.route(grid, True)
                return int(tree.leaf_labels[leaf[0]]), float(confidence[0])

            orders = {
                "exact": _tile_order(attribution["overview_map"]),
                "random": random_order_box_first(rng, box_mask),
            }
            for name in rankings:
                metrics = pixel_masking_metrics(
                    path_evaluate, orders[name], grid_shape, budget, box_tiles, auc_steps
                )
                for key, value in metrics.items():
                    path_results[name][key].append(value)

            if not per_node:
                continue
            aligned_maps = []
            for position, node_index in enumerate(attribution["node_indices"]):
                if not np.any(tree.node_weights[node_index] != 0.0):
                    continue  # pruned: masking cannot change it

                def node_evaluate(keep: np.ndarray, node_index: int = node_index) -> tuple[int, float]:
                    score = float(path_scores(tree, pooled_for(keep), [node_index])[0])
                    return int(score >= 0.0), float(1.0 / (1.0 + np.exp(-abs(score))))

                aligned = attribution["node_maps"][position]
                aligned_maps.append(aligned)
                for name, order in (
                    ("exact", _tile_order(aligned)),
                    ("random", random_order_box_first(rng, box_mask)),
                ):
                    metrics = pixel_masking_metrics(
                        node_evaluate, order, grid_shape, budget, box_tiles, auc_steps
                    )
                    for key, value in metrics.items():
                        node_results[name][position][key].append(value)
            for map_a, map_b in zip(aligned_maps[:-1], aligned_maps[1:]):
                flat_a, flat_b = map_a.reshape(-1), map_b.reshape(-1)
                denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
                if denom > 1e-8:
                    node_similarity.append(float(np.dot(flat_a, flat_b) / denom))

    def summary(values: dict[str, list[float]]) -> dict[str, float]:
        return {
            "necessity_prediction_flip_rate": _mean(values["necessity"]),
            "sufficiency_prediction_preservation": _mean(values["sufficiency"]),
            "deletion_auc": _mean(values["deletion_auc"]),
            "insertion_auc": _mean(values["insertion_auc"]),
            "evaluated_roi_count": len(values["necessity"]),
        }

    result: dict[str, Any] = {"path": {name: summary(values) for name, values in path_results.items()}}
    if per_node:
        result["node"] = {
            name: {position: summary(bucket) for position, bucket in sorted(buckets.items())}
            for name, buckets in node_results.items()
        }
        result["node_map_similarity"] = {
            "mean_cosine_similarity_between_consecutive_nodes": _mean(node_similarity),
            "evaluated_pair_count": len(node_similarity),
        }
    return result


def _found(
    detection: dict[str, Tensor],
    gt_box: Tensor,
    gt_label: int,
    score_threshold: float,
    iou_threshold: float,
) -> bool:
    """A same-class detection at or above the score threshold overlaps the box enough."""
    keep = (detection["labels"] == gt_label) & (detection["scores"] >= score_threshold)
    if not bool(keep.any()):
        return False
    return bool((box_iou(detection["boxes"][keep], gt_box[None]) >= iou_threshold).any())


def breakdown_nesy_misses(
    frcnn_detections: list[dict[str, Tensor]],
    nesy_detections: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    tree: Any,
    score_threshold: float,
    iou_threshold: float,
    nesy_score_threshold: float | None = None,
) -> dict[str, Any]:
    """Ground truths FRCNN finds at the P/R operating point and NeSy does not, by cause.

    `nesy_score_threshold` is NeSy's own operating point (default: `score_threshold`).

    "score": NeSy has a same-class detection on it, scored under the threshold.
    "class": NeSy has a detection on it at or above the threshold, of another class.
    "none":  nothing on it (routed to background, or suppressed by Soft-NMS).
    Checked in that order. A per-box existence check, not the evaluator's one-to-one
    greedy match, so totals can differ from its FN by a few. For "score" misses the
    weakest node is the path node with the lowest log σ(|margin|) of the best-scored
    such detection; its depth is counted (0 = root). Detections need `pooled_features`, or
    precomputed `log_certainties` ([detections, nodes], as `path_log_certainties` returns).
    "score_misses" lists each such miss with its label and per-node log certainties.
    """
    nesy_threshold = score_threshold if nesy_score_threshold is None else nesy_score_threshold
    causes: dict[int, dict[str, int]] = defaultdict(lambda: {"score": 0, "class": 0, "none": 0})
    frcnn_hits: dict[int, int] = defaultdict(int)
    weakest_depth: dict[int, int] = defaultdict(int)
    score_misses: list[dict[str, Any]] = []

    for frcnn, nesy, target in zip(frcnn_detections, nesy_detections, targets):
        for gt_box, gt_label in zip(target["boxes"], target["labels"].tolist()):
            if not _found(frcnn, gt_box, gt_label, score_threshold, iou_threshold):
                continue
            frcnn_hits[gt_label] += 1
            if _found(nesy, gt_box, gt_label, nesy_threshold, iou_threshold):
                continue

            overlap = box_iou(nesy["boxes"], gt_box[None])[:, 0] >= iou_threshold
            weak = overlap & (nesy["labels"] == gt_label)
            if bool(weak.any()):
                causes[gt_label]["score"] += 1
                best = int(torch.where(weak, nesy["scores"], torch.full_like(nesy["scores"], -1.0)).argmax())
                if "log_certainties" in nesy:
                    log_certainties = nesy["log_certainties"][best].numpy()
                else:
                    _, path = tree.path_log_certainties(nesy["pooled_features"][best].numpy().reshape(1, -1))
                    log_certainties = path[0]
                weakest = int(np.argmin(log_certainties))
                weakest_depth[tree.depth_of(weakest)] += 1
                score_misses.append({"label": gt_label, "log_certainties": log_certainties})
            elif bool((overlap & (nesy["scores"] >= nesy_threshold)).any()):
                causes[gt_label]["class"] += 1
            else:
                causes[gt_label]["none"] += 1

    return {
        "causes": {label: dict(counts) for label, counts in causes.items()},
        "frcnn_hits": dict(frcnn_hits),
        "weakest_depth": dict(weakest_depth),
        "score_misses": score_misses,
    }


FALSE_POSITIVE_KINDS = ("duplicate", "wrong_class", "mislocalized", "background")


def breakdown_false_positives(
    detections: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    score_threshold: float,
    iou_threshold: float,
    reference: list[dict[str, Tensor]] | None = None,
    background_iou: float = 0.1,
) -> dict[str, Any]:
    """Every false positive at the P/R operating point, by kind.

    Same matching as the precision/recall counter (`neuro.train._greedy_match_within_label`):
    per class, best score first, each ground truth claimed once, a detection is a
    false positive when its best same-class ground truth is below the IoU threshold or
    already claimed. Kinds, checked in this order:
    "duplicate":    IoU >= threshold with a same-class ground truth (already claimed).
    "wrong_class":  IoU >= threshold with a ground truth of another class.
    "mislocalized": best IoU over all ground truths in [background_iou, threshold).
    "background":   best IoU under background_iou (or no ground truth in the image).
    With `reference` (e.g. FRCNN's detections), "shared" counts the false positives it
    also produces: a same-class reference detection at or above the score threshold
    with IoU >= threshold.
    """
    counts = {kind: 0 for kind in FALSE_POSITIVE_KINDS}
    shared = {kind: 0 for kind in FALSE_POSITIVE_KINDS}
    reference = reference if reference is not None else [None] * len(detections)

    for detection, target, other in zip(detections, targets, reference):
        keep = detection["scores"] >= score_threshold
        boxes, scores, labels = detection["boxes"][keep], detection["scores"][keep], detection["labels"][keep]
        gt_boxes, gt_labels = target["boxes"], target["labels"]
        if boxes.shape[0] == 0:
            continue
        ious = box_iou(boxes, gt_boxes) if gt_boxes.shape[0] else torch.zeros((boxes.shape[0], 0))

        for label in labels.unique().tolist():
            same_gt = gt_labels == label
            claimed: set[int] = set()
            indices = torch.where(labels == label)[0]
            for row in indices[torch.argsort(scores[indices], descending=True)].tolist():
                if bool(same_gt.any()):
                    best_iou, best_gt = ious[row][same_gt].max(dim=0)
                    best_gt = int(torch.where(same_gt)[0][best_gt])
                    if best_iou.item() >= iou_threshold and best_gt not in claimed:
                        claimed.add(best_gt)
                        continue
                overlap = ious[row].max().item() if ious.shape[1] else 0.0
                if bool((ious[row][same_gt] >= iou_threshold).any()):
                    kind = "duplicate"
                elif bool((ious[row][~same_gt] >= iou_threshold).any()):
                    kind = "wrong_class"
                elif overlap >= background_iou:
                    kind = "mislocalized"
                else:
                    kind = "background"
                counts[kind] += 1
                if other is not None and _found(other, boxes[row], label, score_threshold, iou_threshold):
                    shared[kind] += 1

    return {"counts": counts, "shared": shared, "total": sum(counts.values())}
