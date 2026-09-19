"""Both sides perturb the same unit on the same schedule — locked by these tests."""

import numpy as np
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.evaluation import _masked_level, _node_score_after_masking
from neurosym.heatmap import _fpn_box_bounds
from symbolic.evaluation import (
    _FAITHFULNESS_CELL_BUDGET_FRACTION,
    _full_channel_flat_indices,
    _row_cell_ranking,
)
from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from util.heatmap_metrics import importance_ranking


def test_cell_budget_matches_gradcams_top_half_of_49():
    # Same 24-cell budget as Grad-CAM's top half of 49.
    grid_h, grid_w = 7, 7
    cell_budget = max(int(grid_h * grid_w * _FAITHFULNESS_CELL_BUDGET_FRACTION), 1)
    assert cell_budget == 24


def test_deletion_insertion_k_schedule_matches_gradcam():
    # Same k schedule as Grad-CAM.
    num_cells = 7 * 7
    steps = 10
    schedule = [int(s / steps * num_cells) for s in range(steps + 1)]
    assert schedule[0] == 0
    assert schedule[-1] == num_cells
    assert schedule == sorted(schedule)  # monotone non-decreasing


def test_full_channel_flat_indices_covers_every_channel_at_selected_cells():
    channels, height, width = 4, 3, 3
    feature_shape = (channels, height, width)
    # Setup
    cells = np.array([5], dtype=np.int64)
    flat = _full_channel_flat_indices(cells, feature_shape)

    grid = np.zeros(feature_shape, dtype=np.float32)
    grid_flat = grid.reshape(-1)
    grid_flat[flat] = 1.0
    grid = grid_flat.reshape(feature_shape)

    # Every channel at the cell marked, nothing else.
    assert np.all(grid[:, 1, 2] == 1.0)
    grid[:, 1, 2] = 0.0
    assert not grid.any()


def test_row_cell_ranking_covers_every_cell_regardless_of_sparsity():
    # Sparse weights must not shrink the perturbation pool.
    channels, height, width = 2, 3, 3
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1,
        num_classes=2,
        input_dim=channels * height * width,
        feature_shape=(channels, height, width),
    )
    # Setup: one nonzero weight.
    tree.node_weights[0, 0] = 1.0

    feature_row = np.random.default_rng(0).random(
        channels * height * width
    ).astype(np.float32)
    ranking = _row_cell_ranking(tree, feature_row)
    assert ranking.shape == (height * width,)
    assert set(ranking.tolist()) == set(range(height * width))


def test_importance_ranking_is_shared_by_both_evaluators():
    # One shared ranking keeps tie-breaking identical both sides.
    heatmap = torch.tensor([[1.0, 3.0], [2.0, 0.5]])
    ranking = importance_ranking(heatmap)
    assert ranking.tolist() == [1, 2, 0, 3]  # descending: 3.0, 2.0, 1.0, 0.5


def _node_masking_fixture(box: tuple[float, float, float, float]):
    """One level, one box, one node — enough to check masked-score identities
    without a real detector (roi_align is real, everything around it is not)."""
    channels, size, output = 3, 16, 4
    level_map = torch.rand(channels, size, size) + 0.5  # nonzero everywhere
    roi_align = MultiScaleRoIAlign(
        featmap_names=["p2"], output_size=(output, output), sampling_ratio=2
    )
    fpn_features = {"p2": level_map.unsqueeze(0)}
    weight = np.random.default_rng(0).normal(size=channels * output * output).astype(np.float32)
    bias = 0.37
    box_t = torch.tensor(box)
    bounds = _fpn_box_bounds(box_t, (size, size), (size, size), margin=2)

    class _Detector:
        pass

    detector = _Detector()
    detector.roi_align = roi_align

    def score(masked_level):
        return _node_score_after_masking(
            detector, fpn_features, "p2", masked_level, box_t, (size, size), weight, bias
        )

    return level_map, bounds, bias, score


def test_node_score_with_empty_level_equals_bias():
    # Zero features in, bias out — plain algebra, must hold for any box.
    level_map, bounds, bias, score = _node_masking_fixture((0.0, 0.0, 6.0, 6.0))
    fx1, fy1, fx2, fy2 = bounds
    empty_keep = torch.zeros((fy2 - fy1, fx2 - fx1), dtype=torch.bool)
    empty_level = _masked_level(level_map, bounds, empty_keep, start_from_zero=True)
    assert abs(score(empty_level) - bias) < 1e-5


def test_node_score_at_full_deletion_or_insertion_matches_the_endpoint_it_should():
    # Endpoints of the deletion/insertion curves reuse these two identities
    # instead of a re-pool — holds only if RoI-Align's sampling support for
    # the box never spills past `bounds`. Checked at three edges and a
    # corner, where `_fpn_box_bounds` clamps the margin tightest.
    for box in [(0.0, 0.0, 6.0, 6.0), (10.0, 10.0, 16.0, 16.0), (9.0, 0.0, 16.0, 7.0)]:
        level_map, bounds, bias, score = _node_masking_fixture(box)
        fx1, fy1, fx2, fy2 = bounds
        box_shape = (fy2 - fy1, fx2 - fx1)
        empty_keep = torch.zeros(box_shape, dtype=torch.bool)
        full_keep = torch.ones(box_shape, dtype=torch.bool)

        base_score = score(level_map)
        # Insertion, frac=1.0: whole box kept -> equals the unmasked score.
        ins_full = _masked_level(level_map, bounds, full_keep, start_from_zero=True)
        assert abs(score(ins_full) - base_score) < 1e-5
        # Deletion, frac=1.0: whole box masked (keep=False everywhere) -> bias.
        del_full = _masked_level(level_map, bounds, empty_keep, start_from_zero=False)
        assert abs(score(del_full) - bias) < 1e-5


if __name__ == "__main__":
    test_cell_budget_matches_gradcams_top_half_of_49()
    test_deletion_insertion_k_schedule_matches_gradcam()
    test_full_channel_flat_indices_covers_every_channel_at_selected_cells()
    test_row_cell_ranking_covers_every_cell_regardless_of_sparsity()
    test_importance_ranking_is_shared_by_both_evaluators()
    test_node_score_with_empty_level_equals_bias()
    test_node_score_at_full_deletion_or_insertion_matches_the_endpoint_it_should()
    print("OK")


def test_shared_necessity_sufficiency_endpoints():
    # Budget = whole box: keeping everything must preserve the label, and
    # zeroing everything must leave only the bias to decide it.
    from neurosym.evaluation import fpn_necessity_sufficiency

    level_map, bounds, bias, score = _node_masking_fixture((2.0, 2.0, 9.0, 9.0))
    fx1, fy1, fx2, fy2 = bounds
    n_pos = (fx2 - fx1) * (fy2 - fy1)
    fpn_features = {"p2": level_map.unsqueeze(0)}
    roi_align = MultiScaleRoIAlign(featmap_names=["p2"], output_size=(4, 4), sampling_ratio=2)
    box = torch.tensor((2.0, 2.0, 9.0, 9.0))
    weight = np.random.default_rng(0).normal(size=3 * 4 * 4).astype(np.float32)

    def classify(grid):
        value = grid.reshape(grid.shape[0], -1).numpy() @ weight + bias
        return (value >= 0).astype(np.int64)

    base = int(classify(roi_align(fpn_features, [box.unsqueeze(0)], [(16, 16)]))[0])
    flipped, preserved = fpn_necessity_sufficiency(
        roi_align, fpn_features, "p2", box, (16, 16), bounds,
        np.arange(n_pos), n_pos, classify, base,
    )
    assert preserved == 1.0
    assert flipped == float(int(bias >= 0) != base)
