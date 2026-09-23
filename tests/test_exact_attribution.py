"""The map must sum back to the score exactly — linearity, not luck."""

import numpy as np
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.heatmap import (
    compute_exact_attribution,
    compute_node_local_evidence_maps,
    exact_fpn_contribution,
    path_fpn_map,
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
    # One level keeps the identity deterministic.
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

    # Setup
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
    # Leaf-only path, untouched by exact-attribution work.
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


def test_node_heatmap_uses_absolute_value_not_positive_only():
    """Mixed positive/negative contributions: `raw_node_heatmap` must be the
    magnitude of the signed contribution — matches the FPN path's
    `.abs().sum(0)` (heatmap.path_fpn_map / exact_fpn_contribution) — not
    just the positive half."""
    rng = np.random.default_rng(7)
    feature_shape = (3, 2, 2)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[0] = rng.normal(size=tree.node_weights.shape[1]).astype(np.float32)
    tree.node_bias[0] = 0.0
    tree.leaf_labels[:] = [0, 1]

    grid = rng.normal(size=feature_shape).astype(np.float32)
    path = tree.decision_path(grid.reshape(-1))
    node_explanations = compute_node_local_evidence_maps(tree, grid)

    step = path[0]
    node = node_explanations[0]
    direction = 1.0 if step.went_left else -1.0
    weight_grid = tree.node_weight_grid(step.node_index)
    signed_local = direction * weight_grid * grid

    expected = np.abs(signed_local).sum(axis=0).astype(np.float32)
    np.testing.assert_allclose(node["raw_node_heatmap"], expected, atol=1e-5)

    # Fixture must actually produce a negative signed contribution somewhere,
    # and that cell's magnitude must still be > 0 (not clipped to 0 by max()).
    negative_cells = np.any(signed_local < 0, axis=0)
    assert negative_cells.any()
    assert np.all(node["raw_node_heatmap"][negative_cells] > 0)


def test_exact_fpn_contribution_sums_to_each_node_score_not_just_the_path():
    """Per-node claim: each node's own map is a decomposition of THAT node's
    score, not just the whole path's. Note the bias correction — `score` from
    `decision_path` is `w.x + b`, but the weight-grid contribution only
    carries `w.x` (direction-signed), so the identity is `direction*(score-bias)`,
    same as `test_signed_evidence_map_sums_to_the_node_score_exactly` below."""
    rng = np.random.default_rng(1)
    feature_shape = (3, 4, 4)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)
    roi_align = _roi_align(feature_shape)

    fpn = _fpn(3, 16)
    fpn["p2"][:, 2:10, 2:10] = torch.rand(3, 8, 8) + 0.5
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])

    grid = roi_align(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )[0].numpy().astype(np.float32)

    path = tree.decision_path(grid.reshape(-1))
    node_maps = []
    for step in path:
        contribution = exact_fpn_contribution(
            tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE, path=[step],
        )
        node_maps.append(contribution)

        direction = 1.0 if step.went_left else -1.0
        bias = float(tree.node_bias[step.node_index])
        expected = direction * (step.score - bias)
        assert abs(float(contribution.sum()) - expected) < 1e-4

    # Additivity: the six (here: three) node maps sum to the path map exactly
    # — everything's linear, so this must hold before `abs()` collapses sign.
    path_contribution = exact_fpn_contribution(
        tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE, path=path,
    )
    summed_nodes = torch.stack(node_maps, dim=0).sum(dim=0)
    assert torch.allclose(summed_nodes, path_contribution, atol=1e-4)


def _two_opposing_nodes_tree(feature_shape: tuple[int, int, int]) -> SparseObliqueDecisionTreeClassifier:
    """Root goes left (+x), its left child goes right (-x), same cell: signed sum cancels."""
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[0, 0] = 1.0
    tree.node_weights[1, 0] = 1.0
    tree.node_bias[1] = -100.0
    return tree


def _grid(roi_align, fpn, box) -> np.ndarray:
    return roi_align(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )[0].numpy().astype(np.float32)


def test_path_map_is_the_per_node_panels_stacked():
    rng = np.random.default_rng(2)
    feature_shape = (3, 4, 4)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    roi_align = _roi_align(feature_shape)
    fpn = _fpn(3, 16)
    fpn["p2"][:, 2:10, 2:10] = torch.rand(3, 8, 8) + 0.5
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])
    grid = _grid(roi_align, fpn, box)

    path = tree.decision_path(grid.reshape(-1))
    panels = sum(
        exact_fpn_contribution(
            tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE, path=[step]
        ).abs().sum(0)
        for step in path
    )
    stacked = path_fpn_map(tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE)
    assert torch.allclose(stacked, panels, atol=1e-5)


def test_opposing_nodes_do_not_cancel_in_the_path_map():
    feature_shape = (1, 4, 4)
    tree = _two_opposing_nodes_tree(feature_shape)
    roi_align = _roi_align(feature_shape)
    fpn = _fpn(1, 16)
    fpn["p2"][0, 2:10, 2:10] = 2.0
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])
    grid = _grid(roi_align, fpn, box)

    path = tree.decision_path(grid.reshape(-1))
    assert [step.went_left for step in path] == [True, False]

    signed = exact_fpn_contribution(
        tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE, path=path
    )
    assert float(signed.abs().sum()) < 1e-5  # the old signed sum loses the cell

    stacked = path_fpn_map(tree, grid, roi_align, fpn, "p2", box, PROCESSED_SIZE)
    assert float(stacked.sum()) > 0.0


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
    # Setup: original box is 2x the processed box.
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
    test_node_heatmap_uses_absolute_value_not_positive_only()
    test_exact_fpn_contribution_sums_to_each_node_score_not_just_the_path()
    test_path_map_is_the_per_node_panels_stacked()
    test_opposing_nodes_do_not_cancel_in_the_path_map()
    test_explain_hybrid_detection_projects_attribution_from_processed_box()
    print("OK")
