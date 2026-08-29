"""Exact path attribution: the SODT decision path's score decomposed onto the
pre-pooling FPN feature map (see neurosym/heatmap.py).

The load-bearing property is exactness — the per-pixel contribution sums back to
the node/path score with no approximation, because RoI-Align is linear. The
other checks confirm the presentation wrapper stays inside the box and follows
the tree's channel selection.
"""

import numpy as np
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.heatmap import (
    compute_exact_attribution,
    compute_node_local_evidence_maps,
    exact_fpn_contribution,
    path_weight_grid,
)
from neurosym.inference import explain_hybrid_detection
from symbolic.sodt import SparseObliqueDecisionTreeClassifier

PROCESSED_SIZE = (32, 32)
PADDED_SIZE = (32, 32)


def _single_leaf_tree(feature_shape: tuple[int, int, int]) -> SparseObliqueDecisionTreeClassifier:
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[0, 0] = 1.0  # channel 0, cell (0, 0)
    tree.leaf_labels[:] = [0, 1]
    return tree


def _roi_align(feature_shape: tuple[int, int, int]) -> MultiScaleRoIAlign:
    # One level => LevelMapper can only route to it, so the exactness identity
    # is deterministic in a synthetic test.
    return MultiScaleRoIAlign(
        featmap_names=["p2"], output_size=feature_shape[1:], sampling_ratio=2
    )


def _fpn(channels: int, size: int) -> dict[str, torch.Tensor]:
    return {"p2": torch.zeros(channels, size, size)}


def test_exact_fpn_contribution_sums_to_the_path_score():
    feature_shape = (3, 4, 4)
    tree = _single_leaf_tree(feature_shape)
    roi_align = _roi_align(feature_shape)

    fpn = _fpn(3, 16)
    fpn["p2"][:, 2:10, 2:10] = torch.rand(3, 8, 8) + 0.5
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])

    # Grid the tree splits on = what RoI-Align produces for this box.
    grid = roi_align(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )[0].numpy().astype(np.float32)

    contribution = exact_fpn_contribution(
        tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE
    )
    score = float((path_weight_grid(tree, grid) * grid).sum())
    assert abs(float(contribution.sum()) - score) < 1e-4


def test_map_follows_the_tree_selected_channel():
    feature_shape = (2, 4, 4)
    tree = _single_leaf_tree(feature_shape)  # weight only on channel 0
    roi_align = _roi_align(feature_shape)

    fpn = _fpn(2, 16)
    fpn["p2"][0, 0:4, 0:4] = 5.0    # channel 0 hot in the box's top-left
    fpn["p2"][1, 12:16, 12:16] = 5.0  # channel 1 hot outside / bottom-right
    box = torch.tensor([0.0, 0.0, 16.0, 16.0])

    grid = roi_align(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )[0].numpy().astype(np.float32)

    heatmap = compute_exact_attribution(
        tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE, PADDED_SIZE
    )
    peak_row, peak_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert peak_row < heatmap.shape[0] // 2 and peak_col < heatmap.shape[1] // 2


def test_map_crops_to_the_processed_box():
    feature_shape = (1, 4, 4)
    tree = _single_leaf_tree(feature_shape)
    roi_align = _roi_align(feature_shape)

    fpn = _fpn(1, 16)
    fpn["p2"][0, 12:16, 12:16] = 10.0  # hot region entirely outside the box below
    box = torch.tensor([0.0, 0.0, 8.0, 8.0])

    grid = roi_align(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )[0].numpy().astype(np.float32)

    heatmap = compute_exact_attribution(
        tree, grid, roi_align, fpn, "p2", box, PADDED_SIZE, PADDED_SIZE
    )
    assert float(np.abs(heatmap).sum()) == 0.0


def test_signed_evidence_map_sums_to_the_node_score_exactly():
    # The leaf-only pooled-grid path (unchanged by the exact-attribution work):
    # the signed evidence map's total is direction*(w.x) = direction*(score - bias).
    rng = np.random.default_rng(0)
    feature_shape = (5, 4, 4)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=3, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)

    grid = rng.normal(size=feature_shape).astype(np.float32)
    path = tree.decision_path(grid.reshape(-1))
    node_explanations = compute_node_local_evidence_maps(tree, grid)

    for step, node in zip(path, node_explanations):
        direction = 1.0 if step.went_left else -1.0
        bias = float(tree.node_bias[step.node_index])
        expected = direction * (step.score - bias)
        actual = float(node["signed_evidence_map"].sum())
        assert abs(actual - expected) < 1e-4


class _FakeModel:
    def __init__(self, tree: SparseObliqueDecisionTreeClassifier, roi_align) -> None:
        self.symbolic_tree = tree

        class _Detector:
            pass

        self.detector = _Detector()
        self.detector.roi_align = roi_align


def test_explain_hybrid_detection_projects_attribution_from_processed_box():
    feature_shape = (2, 4, 4)
    tree = _single_leaf_tree(feature_shape)
    roi_align = _roi_align(feature_shape)
    model = _FakeModel(tree, roi_align)

    fpn_map = torch.zeros(2, 32, 32)
    fpn_map[0, 10:20, 10:20] = 3.0  # hot inside the processed box
    # Original-image proposal box is 2x the processed-space box.
    proposal_box = torch.tensor([20.0, 20.0, 40.0, 40.0])
    proposal_box_processed = torch.tensor([10.0, 10.0, 20.0, 20.0])

    pooled = roi_align(
        {"p2": fpn_map.unsqueeze(0)}, [proposal_box_processed.unsqueeze(0)], [PROCESSED_SIZE]
    )

    detection = {
        "pooled_features": pooled,
        "proposal_boxes": proposal_box.unsqueeze(0),
        "proposal_boxes_processed": proposal_box_processed.unsqueeze(0),
        "boxes": proposal_box.unsqueeze(0),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
        "symbolic_leaf_indices": torch.tensor([1]),
        "symbolic_probabilities": torch.tensor([[0.1, 0.9]]),
        "symbolic_level_indices": torch.tensor([0]),
        "featmap_names": ["p2"],
        "fpn_features": {"p2": fpn_map},
        "padded_image_size": PADDED_SIZE,
        "processed_image_size": PROCESSED_SIZE,
    }

    explanation = explain_hybrid_detection(
        model, detection, detection_index=0, image_shape=(64, 64)
    )

    assert "projected_path_exact_attribution_on_proposal_box" in explanation
    canvas = explanation["projected_path_exact_attribution_on_proposal_box"]
    assert bool(canvas[20:40, 20:40].abs().sum() > 0)
    assert not bool(canvas[:20, :].abs().sum() > 0)
    assert not bool(canvas[40:, :].abs().sum() > 0)


if __name__ == "__main__":
    test_exact_fpn_contribution_sums_to_the_path_score()
    test_map_follows_the_tree_selected_channel()
    test_map_crops_to_the_processed_box()
    test_signed_evidence_map_sums_to_the_node_score_exactly()
    test_explain_hybrid_detection_projects_attribution_from_processed_box()
    print("OK")
