"""Smoke check: SparseObliqueDecisionTreeClassifier state_dict round-trip
after removing the unused selected_feature_indices/original_input_dim machinery."""

import numpy as np

from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def test_state_dict_round_trip_preserves_predictions():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2,
        num_classes=3,
        input_dim=8,
        feature_shape=(2, 2, 2),
        class_names=("bg", "a", "b"),
    )
    rng = np.random.default_rng(0)
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)

    features = rng.random((20, 8)).astype(np.float32)
    predictions_before = tree.predict(features)

    state = tree.to_state_dict()
    assert "original_input_dim" not in state
    assert "selected_feature_indices" not in state

    restored = SparseObliqueDecisionTreeClassifier.from_state_dict(state)
    predictions_after = restored.predict(features)
    assert np.array_equal(predictions_before, predictions_after)
    assert np.allclose(tree.node_weights, restored.node_weights)


def test_node_weight_grid_reshapes_to_feature_shape():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=12, feature_shape=(3, 2, 2)
    )
    grid = tree.node_weight_grid(0)
    assert grid.shape == (3, 2, 2)


def test_routing_confidence_preserves_routing_and_breaks_ties():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=3, input_dim=8, feature_shape=(2, 2, 2)
    )
    rng = np.random.default_rng(1)
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)
    # Simulate a pruned node: all-zero weights must not affect confidence.
    tree.node_weights[2] = 0.0
    tree.node_bias[2] = 0.0

    features = rng.random((50, 8)).astype(np.float32)
    leaves, confidence = tree.predict_leaf_indices_and_routing_confidence(features)

    assert np.array_equal(leaves, tree.predict_leaf_indices(features))
    assert np.all(confidence > 0.0) and np.all(confidence <= 1.0)
    # Samples sharing a leaf get distinct margin-based scores (tie-break).
    for leaf in np.unique(leaves):
        members = confidence[leaves == leaf]
        if len(members) > 1:
            assert len(np.unique(members)) > 1

    # A fully pruned tree contributes no margin factors at all.
    tree.node_weights[:] = 0.0
    tree.node_bias[:] = 0.0
    _, pruned_confidence = tree.predict_leaf_indices_and_routing_confidence(features)
    assert np.allclose(pruned_confidence, 1.0)


if __name__ == "__main__":
    test_state_dict_round_trip_preserves_predictions()
    test_node_weight_grid_reshapes_to_feature_shape()
    test_routing_confidence_preserves_routing_and_breaks_ties()
    print("OK")
