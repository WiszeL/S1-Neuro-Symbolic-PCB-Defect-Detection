from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import roi_align

from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def describe_decision_path(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_vector: np.ndarray,
) -> list[dict[str, Any]]:
    """One record per node on the decision path: depth, node index, direction taken and raw score."""
    return [
        {
            "depth": depth,
            "node_index": step.node_index,
            "decision": "left" if step.went_left else "right",
            "score": step.score,
        }
        for depth, step in enumerate(tree.decision_path(feature_vector))
    ]


def _normalize_heatmap_array(heatmap: np.ndarray) -> np.ndarray:
    normalized = np.asarray(heatmap, dtype=np.float32)
    max_value = np.max(np.abs(normalized)) if normalized.size > 0 else 0.0
    if max_value <= 0.0:
        return np.zeros_like(normalized, dtype=np.float32)
    return (normalized / max_value).astype(np.float32)


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


def _normalization(detector: Any, like: Tensor) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(detector.transform.image_mean, dtype=like.dtype, device=like.device)
    std = torch.as_tensor(detector.transform.image_std, dtype=like.dtype, device=like.device)
    return mean[:, None, None], std[:, None, None]


def _features_from_normalized_image(detector: Any, normalized_image: Tensor) -> tuple[Any, Any]:
    """Backbone features and processed sizes for an image in (x - mean) / std units.

    Feeding `x̃·std + mean` into the transform makes it normalize straight back to x̃,
    so x̃ is the autograd leaf and gradient and input share units.
    """
    mean, std = _normalization(detector, normalized_image)
    images_list, _ = detector.transform([normalized_image * std + mean], None)
    return detector.backbone(images_list.tensors), images_list.image_sizes


def pooled_roi_from_normalized_image(
    detector: Any,
    normalized_image: Tensor,
    box_processed: Tensor,
) -> Tensor:
    """RoI grid [C, 7, 7] for a fixed box, from an image already in (x - mean) / std units."""
    features, image_sizes = _features_from_normalized_image(detector, normalized_image)
    return detector.roi_align(
        features, [box_processed.detach().to(normalized_image).unsqueeze(0)], image_sizes
    )[0]


def path_scores(
    tree: SparseObliqueDecisionTreeClassifier,
    pooled: Tensor,
    node_indices: list[int],
) -> Tensor:
    """Raw node values s_i = w_i·x + b_i for the given nodes, [len(node_indices)]."""
    weights = torch.as_tensor(tree.node_weights[node_indices]).to(pooled)
    biases = torch.as_tensor(tree.node_bias[node_indices]).to(pooled)
    return weights @ pooled.reshape(-1) + biases


def _place_on_image(
    cells: Tensor, padded_size: tuple[int, int], processed_size: tuple[int, int], image_size: tuple[int, int]
) -> np.ndarray:
    """FPN-cell values [K, h, w] spread evenly over the image pixels each cell covers.

    Every cell keeps its value split over its pixels, so sums over a region are cell sums
    (up to nearest-neighbour rounding when the transform resized the image). Only the block
    of cells that carry gradient is filled and copied off the GPU; the rest stays zero.
    """
    # Original pixels per cell, per axis: cell size in processed pixels times the resize factor.
    size = [
        padded / cells_along * image / processed
        for padded, cells_along, image, processed in zip(
            padded_size, cells.shape[-2:], image_size, processed_size
        )
    ]
    placed = np.zeros((cells.shape[0], *image_size), dtype=np.float32)
    rows, cols = (cells != 0).any(0).any(1).nonzero().flatten(), (cells != 0).any(0).any(0).nonzero().flatten()
    if rows.numel() == 0:
        return placed
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(r0 * size[0]), min(int(np.ceil(r1 * size[0])), image_size[0])
    x0, x1 = int(c0 * size[1]), min(int(np.ceil(c1 * size[1])), image_size[1])
    row_cell = (torch.arange(y0, y1, device=cells.device) / size[0]).long().clamp(r0, r1 - 1) - r0
    col_cell = (torch.arange(x0, x1, device=cells.device) / size[1]).long().clamp(c0, c1 - 1) - c0
    block = cells[:, r0:r1, c0:c1][:, row_cell][:, :, col_cell] / (size[0] * size[1])
    placed[:, y0:y1, x0:x1] = block.float().cpu().numpy()
    return placed


def fpn_node_maps_many(
    model: Any,
    image: Tensor,
    boxes_processed: Tensor,
    paths: list[list[Any]] | None = None,
    cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Which regions contribute to each decision-path node, per box, on the FPN level the RoI read.

    Gradient × Input per channel, magnitudes summed over channels (draft Eq 3.5 and 3.7):
    H_i(p) = Σ_c |(∂s_i/∂F)_c(p) · F_c(p)|. RoI Align is linear, so this is the model's own
    arithmetic, not an approximation. `overview_map` is M = Σ_i H_i (Eq 3.8).

    `cache` is the detection dict from `NeuroSymbolicDetector.forward`: its stored FPN features
    save a second backbone pass. Without it the backbone is re-run on `image`.

    `image` is the original [3, H, W] image, `boxes_processed` is [N, 4]. Maps come back
    spread over original image pixels (see `_place_on_image`), so they plug into the same
    figure and tile scoring as any pixel map; `level_stride` says how coarse the cells are.
    """
    if boxes_processed.shape[0] == 0:
        return []
    detector = model.detector
    tree = model.symbolic_tree
    parameter = next(detector.parameters())
    image_size = tuple(image.shape[-2:])

    with torch.no_grad():
        if cache is None:
            images_list, _ = detector.transform([image.detach().to(parameter)], None)
            features = detector.backbone(images_list.tensors)
            processed_sizes = images_list.image_sizes
            padded_size = tuple(images_list.tensors.shape[-2:])
        else:
            features = cache["fpn_features"]
            processed_sizes = [tuple(cache["processed_size"])]
            padded_size = cache["padded_size"]
        boxes = boxes_processed.detach().to(parameter)
        pooled = detector.roi_align(features, [boxes], processed_sizes)
        pool = detector.roi_align.pool
        level_names = [name for name in features if name in pool.featmap_names]
        box_levels = pool.map_levels([boxes])  # set by the pooling call above
    if paths is None:
        paths = [tree.decision_path(grid.cpu().numpy().reshape(-1)) for grid in pooled]

    results = []
    for k, path in enumerate(paths):
        node_indices = [step.node_index for step in path]
        count = len(node_indices)
        weights = torch.as_tensor(tree.node_weights[node_indices]).to(parameter)
        # One box reads one pyramid level, and only a few cells around it: work on that crop.
        level_index = int(box_levels[k])
        level = features[level_names[level_index]][0]
        scale = pool.scales[level_index]
        x1, y1, x2, y2 = boxes[k].tolist()
        fx0, fy0 = max(0, int(np.floor(x1 * scale)) - 2), max(0, int(np.floor(y1 * scale)) - 2)
        fx1, fy1 = min(level.shape[-1], int(np.ceil(x2 * scale)) + 3), min(level.shape[-2], int(np.ceil(y2 * scale)) + 3)
        crop = level[:, fy0:fy1, fx0:fx1]
        # The box moves with the crop, so RoI Align reads the same cells it read on the full level.
        rois = torch.tensor(
            [[node, x1 - fx0 / scale, y1 - fy0 / scale, x2 - fx0 / scale, y2 - fy0 / scale] for node in range(count)],
            dtype=crop.dtype,
            device=crop.device,
        )
        with torch.inference_mode(False), torch.enable_grad():
            # One copy of the crop per node, so one backward gives every node its own gradient.
            copies = crop.detach().clone()[None].expand(count, -1, -1, -1).contiguous().requires_grad_(True)
            grid = roi_align(
                copies, rois, output_size=pool.output_size, spatial_scale=scale,
                sampling_ratio=pool.sampling_ratio, aligned=False,
            )
            (gradient,) = torch.autograd.grad(grid, copies, grad_outputs=weights.reshape(grid.shape))
            node_crop = (gradient * crop.detach()[None]).abs().sum(1).detach()  # H_i on the crop
        node_cells = torch.zeros(count, *level.shape[-2:], dtype=node_crop.dtype, device=node_crop.device)
        node_cells[:, fy0:fy1, fx0:fx1] = node_crop
        cells = torch.cat([node_cells, node_cells.sum(0, keepdim=True)])  # M placed with them
        placed = _place_on_image(cells, padded_size, tuple(processed_sizes[0]), image_size)
        results.append(
            {
                "path": path,
                "node_indices": node_indices,
                "node_maps": placed[:-1],
                "overview_map": placed[-1],
                "level_stride": padded_size[0] / cells.shape[-2],
            }
        )
    return results


def fpn_node_maps(
    model: Any,
    image: Tensor,
    box_processed: Tensor,
    path: list[Any] | None = None,
) -> dict[str, Any]:
    """`fpn_node_maps_many` for one box."""
    return fpn_node_maps_many(
        model, image, box_processed.reshape(1, 4), paths=None if path is None else [path]
    )[0]
