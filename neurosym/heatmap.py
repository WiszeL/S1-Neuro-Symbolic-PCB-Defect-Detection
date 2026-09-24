from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align as _roi_align_op
from torchvision.ops.poolers import _infer_scale

from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def _as_feature_grid(feature_grid: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(feature_grid, Tensor):
        return feature_grid.detach().cpu().numpy().astype(np.float32)
    return np.asarray(feature_grid, dtype=np.float32)


def _top_grid_cells(
    heatmap: np.ndarray,
    top_k: int = 10,
) -> list[dict[str, int | float]]:
    if heatmap.size == 0:
        return []

    flattened = heatmap.reshape(-1)
    nonzero_indices = np.flatnonzero(np.abs(flattened) > 0)
    if nonzero_indices.size == 0:
        return []

    ranked_indices = nonzero_indices[
        np.argsort(np.abs(flattened[nonzero_indices]))[::-1]
    ]
    results: list[dict[str, int | float]] = []
    width = heatmap.shape[1]
    for flat_index in ranked_indices[: max(top_k, 0)]:
        row_index = int(flat_index // width)
        col_index = int(flat_index % width)
        results.append(
            {
                "row": row_index,
                "col": col_index,
                "signed_contribution": float(flattened[flat_index]),
                "absolute_contribution": float(abs(flattened[flat_index])),
            }
        )
    return results


def _top_channels(
    channel_scores: np.ndarray,
    top_k: int = 8,
) -> list[dict[str, int | float]]:
    scores = np.asarray(channel_scores, dtype=np.float32).reshape(-1)
    nonzero_indices = np.flatnonzero(scores > 0.0)
    if nonzero_indices.size == 0:
        return []

    ranked_indices = nonzero_indices[np.argsort(scores[nonzero_indices])[::-1]]
    return [
        {
            "channel": int(channel_index),
            "score": float(scores[channel_index]),
        }
        for channel_index in ranked_indices[: max(top_k, 0)]
    ]


def _normalize_heatmap_array(heatmap: np.ndarray) -> np.ndarray:
    normalized = np.asarray(heatmap, dtype=np.float32)
    max_value = np.max(np.abs(normalized)) if normalized.size > 0 else 0.0
    if max_value <= 0.0:
        return np.zeros_like(normalized, dtype=np.float32)
    return (normalized / max_value).astype(np.float32)


def compute_node_local_evidence_maps(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
) -> list[dict[str, Any]]:
    grid = _as_feature_grid(feature_grid)
    feature_vector = grid.reshape(-1)
    path = tree.decision_path(feature_vector)

    node_explanations: list[dict[str, Any]] = []
    for depth, step in enumerate(path):
        direction = 1.0 if step.went_left else -1.0
        weight_grid = tree.node_weight_grid(step.node_index)
        signed_local = direction * weight_grid * grid
        # Magnitude, not just the positive half — matches the FPN path's
        # `.abs().sum(0)` (heatmap.path_fpn_map / exact_fpn_contribution), so
        # the FPN-vs-grid ablation differs only in resolution, not in whether
        # negative contributions count.
        evidence_magnitude = np.abs(signed_local).sum(axis=0).astype(np.float32)
        node_heatmap = _normalize_heatmap_array(evidence_magnitude)
        node_explanations.append(
            {
                "depth": int(depth),
                "node_index": int(step.node_index),
                "decision": "left" if step.went_left else "right",
                "score": float(step.score),
                "evidence_magnitude_sum": float(evidence_magnitude.sum()),
                "evidence_magnitude_cell_count": int(
                    np.count_nonzero(evidence_magnitude > 0.0)
                ),
                "active_original_feature_count": int(
                    tree.node_feature_indices(step.node_index).size
                ),
                "node_heatmap": node_heatmap,
                "raw_node_heatmap": evidence_magnitude,
                # Signed, so it sums back to the node's score (tested).
                "signed_evidence_map": signed_local.sum(axis=0).astype(np.float32),
                "top_local_cells": _top_grid_cells(node_heatmap, top_k=8),
                "top_positive_local_channels": _top_channels(
                    np.maximum(signed_local, 0.0).sum(axis=(1, 2)),
                    top_k=8,
                ),
            }
        )

    return node_explanations


def compute_symbolic_heatmap(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    mode: str = "local_instance_evidence_map",
) -> dict[str, Any]:
    grid = _as_feature_grid(feature_grid)
    feature_vector = grid.reshape(-1)
    # Path never empty — depth always >= 1 by construction.
    path = tree.decision_path(feature_vector)

    # Gather weights
    weight_grids = np.stack(
        [tree.node_weight_grid(step.node_index) for step in path], axis=0
    )
    path_directions = np.asarray(
        [1.0 if step.went_left else -1.0 for step in path],
        dtype=np.float32,
    ).reshape(-1, 1, 1, 1)

    # Build maps
    local_signed_contributions = path_directions * weight_grids * grid[None, ...]
    structural_density_map = np.sum(np.abs(weight_grids), axis=(0, 1)).astype(
        np.float32
    )
    positive_local_evidence_map = np.sum(
        np.maximum(local_signed_contributions, 0.0),
        axis=(0, 1),
    ).astype(np.float32)
    negative_local_evidence_map = np.sum(
        np.maximum(-local_signed_contributions, 0.0),
        axis=(0, 1),
    ).astype(np.float32)
    signed_local_evidence_map = np.sum(local_signed_contributions, axis=(0, 1)).astype(
        np.float32
    )
    combined_local_evidence_map = (
        positive_local_evidence_map + negative_local_evidence_map
    ).astype(np.float32)
    structural_channel_scores = np.sum(np.abs(weight_grids), axis=(0, 2, 3)).astype(
        np.float32
    )
    positive_local_channel_scores = np.sum(
        np.maximum(local_signed_contributions, 0.0),
        axis=(0, 2, 3),
    ).astype(np.float32)
    negative_local_channel_scores = np.sum(
        np.maximum(-local_signed_contributions, 0.0),
        axis=(0, 2, 3),
    ).astype(np.float32)

    maps = {
        "global_or_structural_density_map": structural_density_map,
        "local_instance_evidence_map": positive_local_evidence_map,
        "negative_local_evidence_map": negative_local_evidence_map,
        "signed_local_evidence_map": signed_local_evidence_map,
        "combined_local_evidence_map": combined_local_evidence_map,
    }
    if mode not in maps:
        raise ValueError(
            f"Unknown symbolic heatmap mode {mode!r}. Expected one of {sorted(maps)}."
        )
    path_summary = tree.summarize_path(feature_vector, path=path)
    path_trace = [
        {
            "node_index": int(step.node_index),
            "score": float(step.score),
            "went_left": bool(step.went_left),
            "active_original_feature_count": int(
                tree.node_feature_indices(step.node_index).size
            ),
        }
        for step in path
    ]
    top_local_cells = _top_grid_cells(positive_local_evidence_map, top_k=12)
    top_negative_local_cells = _top_grid_cells(negative_local_evidence_map, top_k=12)
    top_structural_cells = _top_grid_cells(structural_density_map, top_k=12)
    top_structural_channels = _top_channels(structural_channel_scores, top_k=8)
    top_positive_local_channels = _top_channels(positive_local_channel_scores, top_k=8)
    top_negative_local_channels = _top_channels(negative_local_channel_scores, top_k=8)
    return {
        "heatmap": maps[mode],
        "path": path,
        "path_trace": path_trace,
        "leaf_index": tree.leaf_index_for_feature(feature_vector),
        "reasoning_summary": {
            "path_length": int(path_summary["path_length"]),
            "active_path_original_feature_count": int(
                path_summary["active_path_original_feature_count"]
            ),
            "mean_active_original_features_per_node": float(
                path_summary["mean_active_original_features_per_node"]
            ),
            "map_types": {
                "global_or_structural_density_map": (
                    "Intrinsic structural view of what the active symbolic path weights use in general."
                ),
                "local_instance_evidence_map": (
                    "Intrinsic per-instance symbolic evidence for this RoI using the actual pooled feature values."
                ),
            },
        },
        "top_local_cells": top_local_cells,
        "top_negative_local_cells": top_negative_local_cells,
        "top_structural_cells": top_structural_cells,
        "top_structural_channels": top_structural_channels,
        "top_positive_local_channels": top_positive_local_channels,
        "top_negative_local_channels": top_negative_local_channels,
        **maps,
    }


def _fpn_box_bounds(
    box_processed: Tensor,
    padded_size: tuple[int, int],
    feature_hw: tuple[int, int],
    margin: int = 0,
) -> tuple[int, int, int, int]:
    """Box bounds on an FPN level map; margin covers RoI-Align's sampling spill."""
    feature_h, feature_w = feature_hw
    padded_h, padded_w = padded_size
    scale_x = feature_w / padded_w
    scale_y = feature_h / padded_h
    x1, y1, x2, y2 = box_processed.detach().cpu().tolist()
    fx1 = max(0, int(np.floor(x1 * scale_x)) - margin)
    fy1 = max(0, int(np.floor(y1 * scale_y)) - margin)
    fx2 = min(feature_w, int(np.ceil(x2 * scale_x)) + margin)
    fy2 = min(feature_h, int(np.ceil(y2 * scale_y)) + margin)
    return fx1, fy1, fx2, fy2


def path_weight_grid(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    path: list[Any] | None = None,
) -> np.ndarray:
    """Path weights summed with routing signs; None walks the whole path."""
    grid = _as_feature_grid(feature_grid)
    if path is None:
        path = tree.decision_path(grid.reshape(-1))
    total = np.zeros(tree.feature_shape, dtype=np.float32)
    for step in path:
        direction = 1.0 if step.went_left else -1.0
        total += direction * tree.node_weight_grid(step.node_index)
    return total


def exact_fpn_contribution(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    roi_align: Any,
    fpn_features: dict[str, Tensor],
    level_name: str,
    box_processed: Tensor,
    processed_image_size: tuple[int, int],
    path: list[Any] | None = None,
    weight_grid_override: np.ndarray | None = None,  # controls only (permuted/foreign weights)
) -> Tensor:
    """Path score split exactly across the FPN pixels it came from.

    RoI-Align is linear, so this sums back to the score with no
    approximation — autograd just reads RoI-Align's own coefficients.
    """
    weights = (
        weight_grid_override
        if weight_grid_override is not None
        else path_weight_grid(tree, feature_grid, path=path)
    )
    device = fpn_features[level_name].device  # stay where the features are (GPU stays GPU)
    with torch.inference_mode(False), torch.enable_grad():
        # Fresh copies so autograd works even on tensors made in inference mode.
        feats = {name: level.detach().clone().unsqueeze(0) for name, level in fpn_features.items()}
        feats[level_name].requires_grad_(True)
        pooled = roi_align(
            feats,
            [box_processed.detach().to(device).unsqueeze(0)],
            [tuple(processed_image_size)],
        )[0]
        score = (torch.as_tensor(weights, dtype=torch.float32, device=device) * pooled).sum()
        (grad,) = torch.autograd.grad(score, feats[level_name])
    return (grad[0] * feats[level_name][0]).detach()


def path_fpn_map(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    roi_align: Any,
    fpn_features: dict[str, Tensor],
    level_name: str,
    box_processed: Tensor,
    processed_image_size: tuple[int, int],
    path: list[Any] | None = None,
    node_weight_grids: list[np.ndarray] | None = None,  # controls only, one per step
) -> Tensor:
    """Stacked per-node panels: sum of each node's |map|; nodes never cancel.

    Exact per node, not per path — the tree has no single path score.
    """
    if path is None:
        path = tree.decision_path(_as_feature_grid(feature_grid).reshape(-1))
    total = None
    for index, step in enumerate(path):
        node_map = exact_fpn_contribution(
            tree,
            feature_grid,
            roi_align,
            fpn_features,
            level_name,
            box_processed,
            processed_image_size,
            path=[step],
            weight_grid_override=None
            if node_weight_grids is None
            else node_weight_grids[index],
        ).abs().sum(0)
        total = node_map if total is None else total + node_map
    return total


def compute_exact_attribution(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    roi_align: Any,
    fpn_features: dict[str, Tensor],
    level_name: str,
    box_processed: Tensor,
    processed_image_size: tuple[int, int],
    padded_size: tuple[int, int],
    path: list[Any] | None = None,
    margin: int = 0,
) -> np.ndarray:
    """Exact map as a 2-D picture, cropped to the box.

    One step = that node's panel; several = their panels stacked.
    Honest limits: channel-collapsing is a display choice, and FPN
    pixels are regions, not image pixels.
    """
    heatmap = (
        path_fpn_map(
            tree,
            feature_grid,
            roi_align,
            fpn_features,
            level_name,
            box_processed,
            processed_image_size,
            path=path,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    fx1, fy1, fx2, fy2 = _fpn_box_bounds(
        box_processed, padded_size, heatmap.shape, margin=margin
    )
    if fx2 <= fx1 or fy2 <= fy1:
        return np.zeros((1, 1), dtype=np.float32)
    return _normalize_heatmap_array(heatmap[fy1:fy2, fx1:fx2])


def node_fpn_maps(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor | np.ndarray,
    pool: Any,
    level_feature: Tensor,
    box_processed: Tensor,
    processed_image_size: tuple[int, int],
    padded_size: tuple[int, int],
    path: list[Any] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-node exact maps and the stacked path map M in one backward pass.

    Same numbers as `compute_exact_attribution` with `path=[step]` per node and
    `path=path` for M, but on the single FPN level the RoI came from, staying on
    that level's device, with every node read out of one batched autograd call.
    """
    if path is None:
        path = tree.decision_path(_as_feature_grid(feature_grid).reshape(-1))
    if not path:
        return [], np.zeros((1, 1), dtype=np.float32)

    device = level_feature.device
    weights = torch.as_tensor(
        np.stack([path_weight_grid(tree, feature_grid, path=[step]) for step in path]),
        dtype=torch.float32,
        device=device,
    )  # [depth, C, 7, 7]
    depth = weights.shape[0]
    x1, y1, x2, y2 = box_processed.detach().to(device, torch.float32).tolist()
    rois = torch.tensor(
        [[node, x1, y1, x2, y2] for node in range(depth)],
        dtype=torch.float32,
        device=device,
    )
    # Same scale MultiScaleRoIAlign infers; `pool.scales` is overwritten per call.
    scale = _infer_scale(level_feature, list(processed_image_size))

    with torch.inference_mode(False), torch.enable_grad():
        base = level_feature.detach().clone().unsqueeze(0)  # [1, C, H, W]
        # One copy of the level per node, so each node gets its own gradient.
        replicated = base.expand(depth, -1, -1, -1).contiguous().requires_grad_(True)
        pooled = _roi_align_op(
            replicated,
            rois,
            output_size=pool.output_size,
            spatial_scale=scale,
            sampling_ratio=pool.sampling_ratio,
            aligned=False,
        )
        (grad,) = torch.autograd.grad(pooled, replicated, grad_outputs=weights)
        raw = (grad * base).abs().sum(1).detach()  # [depth, H, W]

    fx1, fy1, fx2, fy2 = _fpn_box_bounds(
        box_processed, padded_size, tuple(raw.shape[-2:])
    )
    if fx2 <= fx1 or fy2 <= fy1:
        empty = np.zeros((1, 1), dtype=np.float32)
        return [empty.copy() for _ in path], empty

    # Only the box crop leaves the device, like Grad-CAM's maps.
    node_crops = raw[:, fy1:fy2, fx1:fx2].cpu().numpy().astype(np.float32)
    node_maps = [_normalize_heatmap_array(crop) for crop in node_crops]
    # M sums the raw node maps, then normalizes (not the normalized panels).
    path_map = _normalize_heatmap_array(node_crops.sum(axis=0))
    return node_maps, path_map


def resize_heatmap_to_box(
    heatmap: Tensor | np.ndarray,
    box: Tensor | np.ndarray,
) -> Tensor:
    if isinstance(heatmap, np.ndarray):
        heatmap_tensor = torch.from_numpy(heatmap).float()
    else:
        heatmap_tensor = heatmap.detach().cpu().float()

    if isinstance(box, np.ndarray):
        box_tensor = torch.from_numpy(box).float()
    else:
        box_tensor = box.detach().cpu().float()

    x1, y1, x2, y2 = box_tensor.round().int().tolist()
    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)

    return F.interpolate(
        heatmap_tensor[None, None, :, :],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def project_heatmap_to_image(
    heatmap: Tensor | np.ndarray,
    box: Tensor | np.ndarray,
    image_shape: tuple[int, int],
) -> Tensor:
    if isinstance(box, np.ndarray):
        box_tensor = torch.from_numpy(box).float()
    else:
        box_tensor = box.detach().cpu().float()

    x1, y1, x2, y2 = box_tensor.round().int().tolist()
    resized = resize_heatmap_to_box(heatmap, box)

    img_h, img_w = image_shape
    cx1 = max(0, min(x1, img_w))
    cx2 = max(0, min(x2, img_w))
    cy1 = max(0, min(y1, img_h))
    cy2 = max(0, min(y2, img_h))

    rx1 = max(0, cx1 - x1)
    rx2 = rx1 + (cx2 - cx1)
    ry1 = max(0, cy1 - y1)
    ry2 = ry1 + (cy2 - cy1)

    canvas = torch.zeros(image_shape, dtype=torch.float32)
    if (cx2 > cx1) and (cy2 > cy1):
        canvas[cy1:cy2, cx1:cx2] = resized[ry1:ry2, rx1:rx2]
    return canvas
