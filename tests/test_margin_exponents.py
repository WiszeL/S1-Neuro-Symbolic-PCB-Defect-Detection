"""γ = 1 is the plain routing-margin product; the fit recovers exponents and keeps γ ≥ 0."""

import numpy as np
import pytest

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.train import solve_margin_exponents


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


def test_unit_exponents_reproduce_the_plain_product():
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


def test_exponents_scale_the_log_certainty_of_their_node():
    tree = _random_tree()
    features = np.random.default_rng(2).random((20, 8)).astype(np.float32)
    leaves, log_certainties = tree.path_log_certainties(features)
    tree.margin_exponents[:] = 0.5

    _, confidence = tree.predict_leaf_indices_and_routing_confidence(features)

    assert np.array_equal(leaves, tree.predict_leaf_indices(features))
    assert np.allclose(confidence, np.exp(0.5 * log_certainties.sum(axis=1)), atol=1e-6)


def test_exponents_round_trip_and_are_required():
    tree = _random_tree()
    tree.margin_exponents[:] = np.arange(tree.num_internal_nodes, dtype=np.float32) / 10

    state = tree.to_state_dict()
    restored = SparseObliqueDecisionTreeClassifier.from_state_dict(state)
    assert np.allclose(restored.margin_exponents, tree.margin_exponents)

    del state["margin_exponents"]
    with pytest.raises(KeyError):
        SparseObliqueDecisionTreeClassifier.from_state_dict(state)


def test_fit_recovers_exponents_and_keeps_them_nonnegative():
    rng = np.random.default_rng(3)
    log_certainties = -rng.uniform(0.01, 1.0, size=(3000, 5))
    log_certainties[:, 4] = 0.0  # node never on a path: no data, stays 1
    true_exponents = np.array([0.2, 1.0, 0.5, 0.0, 1.0])
    targets = np.exp(log_certainties @ true_exponents)

    exponents, fit = solve_margin_exponents(log_certainties, targets)

    assert np.allclose(exponents[:4], true_exponents[:4], atol=1e-3)
    assert exponents[4] == 1.0
    assert np.all(exponents >= 0.0)
    assert fit["bce_fitted"] <= fit["bce_plain"]


if __name__ == "__main__":
    test_unit_exponents_reproduce_the_plain_product()
    test_exponents_scale_the_log_certainty_of_their_node()
    test_exponents_round_trip_and_are_required()
    test_fit_recovers_exponents_and_keeps_them_nonnegative()
    print("OK")
