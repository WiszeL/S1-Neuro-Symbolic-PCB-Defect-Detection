"""The pooled-grid evidence maps must sum back to the node score exactly — linearity, not luck.

The image-pixel maps are checked against a finite difference in test_pixel_attribution.py.
"""

import numpy as np

from neurosym.heatmap import compute_node_local_evidence_maps
from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def test_signed_evidence_map_sums_to_the_node_score_exactly():
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
    magnitude of the signed contribution, not just the positive half."""
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


if __name__ == "__main__":
    test_signed_evidence_map_sums_to_the_node_score_exactly()
    test_node_heatmap_uses_absolute_value_not_positive_only()
    print("OK")
