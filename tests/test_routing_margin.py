"""The detection score is the plain routing-margin product: Π σ(|f_i|) over the active path nodes."""

import numpy as np

from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def _random_tree(seed: int = 0) -> SparseObliqueDecisionTreeClassifier:
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=3, input_dim=8, feature_shape=(2, 2, 2)
    )
    rng = np.random.default_rng(seed)
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)
    tree.node_weights[2] = 0.0  # pruned node: skipped
    tree.node_bias[2] = 0.0
    return tree


def test_confidence_is_the_product_of_sigmoids_on_active_nodes():
    tree = _random_tree()
    features = np.random.default_rng(1).random((40, 8)).astype(np.float32)

    _, confidence = tree.predict_leaf_indices_and_routing_confidence(features)

    expected = []
    for feature in features:
        product = 1.0
        for step in tree.decision_path(feature):
            if np.any(tree.node_weights[step.node_index] != 0.0):
                product *= 1.0 / (1.0 + np.exp(-abs(step.score)))
        expected.append(product)
    assert np.allclose(confidence, expected, atol=1e-6)


def test_state_dict_round_trip_keeps_the_confidence():
    tree = _random_tree()
    features = np.random.default_rng(2).random((20, 8)).astype(np.float32)

    restored = SparseObliqueDecisionTreeClassifier.from_state_dict(tree.to_state_dict())

    assert np.allclose(
        tree.predict_leaf_indices_and_routing_confidence(features)[1],
        restored.predict_leaf_indices_and_routing_confidence(features)[1],
    )


if __name__ == "__main__":
    test_confidence_is_the_product_of_sigmoids_on_active_nodes()
    test_state_dict_round_trip_keeps_the_confidence()
    print("ok")
