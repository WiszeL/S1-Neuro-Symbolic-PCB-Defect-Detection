from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

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
        positive_evidence = np.maximum(signed_local, 0.0).sum(axis=0).astype(np.float32)
        node_heatmap = _normalize_heatmap_array(positive_evidence)
        node_explanations.append(
            {
                "depth": int(depth),
                "node_index": int(step.node_index),
                "decision": "left" if step.went_left else "right",
                "score": float(step.score),
                "positive_evidence_sum": float(positive_evidence.sum()),
                "positive_evidence_cell_count": int(
                    np.count_nonzero(positive_evidence > 0.0)
                ),
                "active_original_feature_count": int(
                    tree.node_feature_indices(step.node_index).size
                ),
                "node_heatmap": node_heatmap,
                "raw_node_heatmap": positive_evidence,
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
    # tree.decision_path is never empty here: SparseObliqueDecisionTreeClassifier
    # rejects max_depth <= 0 at construction, so num_internal_nodes >= 1 always.
    path = tree.decision_path(feature_vector)

    # ---------------------------------------------------------------------
    # Gather the actual path weights in the original pooled-feature lattice
    # ---------------------------------------------------------------------
    weight_grids = np.stack(
        [tree.node_weight_grid(step.node_index) for step in path], axis=0
    )
    path_directions = np.asarray(
        [1.0 if step.went_left else -1.0 for step in path],
        dtype=np.float32,
    ).reshape(-1, 1, 1, 1)

    # ---------------------------------------------------------------------
    # Build structural and local evidence maps from the symbolic path itself
    # ---------------------------------------------------------------------
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
