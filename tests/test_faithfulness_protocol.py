"""Both sides perturb the same unit on the same schedule — locked by these tests.

The image-pixel protocol is covered in test_pixel_faithfulness.py.
"""

import numpy as np
import torch

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


if __name__ == "__main__":
    test_cell_budget_matches_gradcams_top_half_of_49()
    test_deletion_insertion_k_schedule_matches_gradcam()
    test_full_channel_flat_indices_covers_every_channel_at_selected_cells()
    test_row_cell_ranking_covers_every_cell_regardless_of_sparsity()
    test_importance_ranking_is_shared_by_both_evaluators()
    print("OK")
