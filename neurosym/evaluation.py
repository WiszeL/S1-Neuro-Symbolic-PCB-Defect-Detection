from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.evaluation import (
    _normalize_heatmap,
    _heatmap_entropy,
    _topk_region_overlap,
    _pointing_score,
    _energy_in_region,
)

from .heatmap import compute_symbolic_heatmap
from util.geometry import project_gt_box_to_roi_grid




def _feature_perturbation_stability(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grid: Tensor,
    base_heatmap: Tensor,
    mode: str,
    random_state: int,
    noise_scale: float = 0.02,
) -> float:
    generator = torch.Generator()
    generator.manual_seed(random_state)
    feature_grid = feature_grid.detach().cpu().float()
    feature_std = float(feature_grid.std(unbiased=False).item())
    if feature_std <= 0.0:
        return 1.0

    perturbation = torch.randn(
        feature_grid.shape,
        generator=generator,
        dtype=feature_grid.dtype,
    ) * (noise_scale * feature_std)
    perturbed_grid = feature_grid + perturbation
    perturbed_heatmap = torch.as_tensor(
        compute_symbolic_heatmap(tree, perturbed_grid, mode=mode)["heatmap"],
        dtype=torch.float32,
    )

    base_vector = _normalize_heatmap(base_heatmap).reshape(-1)
    perturbed_vector = _normalize_heatmap(perturbed_heatmap).reshape(-1)
    if (
        float(base_vector.sum().item()) == 0.0
        and float(perturbed_vector.sum().item()) == 0.0
    ):
        return 1.0

    denominator = float(base_vector.norm().item() * perturbed_vector.norm().item())
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(base_vector, perturbed_vector).item() / denominator)


def evaluate_symbolic_spatial_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    feature_grids: Tensor,
    proposal_boxes: Tensor,
    matched_gt_boxes: Tensor | None,
    has_matched_gt: Tensor | None,
    gt_iou: Tensor | None,
    mode: str = "local_instance_evidence_map",
    random_state: int = 42,
) -> dict[str, Any]:
    if matched_gt_boxes is None or has_matched_gt is None:
        return {
            "evaluated_roi_count": 0,
            "box_grounded_roi_overlap": float("nan"),
            "pointing_score": float("nan"),
            "energy_in_defect_ratio": float("nan"),
            "heatmap_entropy": float("nan"),
            "stability_score": float("nan"),
        }

    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    energy_scores: list[float] = []
    entropy_scores: list[float] = []
    stability_scores: list[float] = []

    for index in range(feature_grids.shape[0]):
        if not bool(has_matched_gt[index]):
            continue
        if gt_iou is not None and float(gt_iou[index]) <= 0.0:
            continue

        feature_grid = feature_grids[index]
        heatmap = torch.as_tensor(
            compute_symbolic_heatmap(tree, feature_grid, mode=mode)["heatmap"],
            dtype=torch.float32,
        )
        normalized_heatmap = _normalize_heatmap(heatmap)
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=proposal_boxes[index],
            matched_gt_box=matched_gt_boxes[index],
            grid_shape=tuple(int(value) for value in normalized_heatmap.shape),
        )
        if int(gt_mask.sum().item()) == 0:
            continue

        overlap_scores.append(_topk_region_overlap(normalized_heatmap, gt_mask))
        pointing_scores.append(_pointing_score(normalized_heatmap, gt_mask))
        energy_scores.append(_energy_in_region(normalized_heatmap, gt_mask))
        entropy_scores.append(_heatmap_entropy(normalized_heatmap))
        stability_scores.append(
            _feature_perturbation_stability(
                tree,
                feature_grid=feature_grid,
                base_heatmap=normalized_heatmap,
                mode=mode,
                random_state=random_state + index,
            )
        )

    if not overlap_scores:
        return {
            "evaluated_roi_count": 0,
            "box_grounded_roi_overlap": float("nan"),
            "pointing_score": float("nan"),
            "energy_in_defect_ratio": float("nan"),
            "heatmap_entropy": float("nan"),
            "stability_score": float("nan"),
        }

    return {
        "evaluated_roi_count": int(len(overlap_scores)),
        "box_grounded_roi_overlap": float(np.mean(overlap_scores)),
        "pointing_score": float(np.mean(pointing_scores)),
        "energy_in_defect_ratio": float(np.mean(energy_scores)),
        "heatmap_entropy": float(np.mean(entropy_scores)),
        "stability_score": float(np.mean(stability_scores)),
    }
