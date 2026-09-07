"""TAO fixes locked: init scale, weighted objective, accept-only-improvements."""

import numpy as np

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.tao import (
    _accept_or_reject_node_update,
    _INIT_WEIGHT_SCALE,
    _reduced_problem_objective,
    evaluate_tree,
    fit_tree_with_tao,
    initialize_tree_weights,
)


def _linearly_separable_dataset(seed: int = 0, n: int = 400, dim: int = 6):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    true_weights = rng.normal(size=(dim,)).astype(np.float32)
    scores = features @ true_weights
    labels = (scores > np.median(scores)).astype(np.int64)  # 2 balanced classes
    return features, labels


def test_init_weight_scale_matches_papers_gaussian_0_1():
    assert _INIT_WEIGHT_SCALE == 1.0


def test_initialize_tree_weights_uses_configured_scale():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=2, input_dim=10
    )
    initialize_tree_weights(tree, random_state=0)
    # Spread must read ~1, not ~0.001.
    assert np.std(tree.node_weights) > 0.3


def test_initialize_tree_weights_randomizes_leaf_labels():
    # Papers randomize leaves too; several leaves must not all agree.
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=4, num_classes=5, input_dim=6
    )
    initialize_tree_weights(tree, random_state=0)
    assert len(set(tree.leaf_labels.tolist())) > 1


def test_evaluate_tree_objective_is_class_weighted_when_given():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=4
    )
    features = np.zeros((4, 4), dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    tree.leaf_labels[:] = [1, 1]  # both leaves predict class 1 -> misclassifies the 0s

    unweighted = evaluate_tree(tree, features, labels)
    class_weights = np.array([1.0, 5.0], dtype=np.float32)  # weight is indexed by TRUE label
    weighted = evaluate_tree(tree, features, labels, class_weights=class_weights)

    # Baseline loss.
    assert unweighted["objective"] == 2.0
    # Flipped weights must actually bite.
    class_weights_flipped = np.array([3.0, 1.0], dtype=np.float32)
    weighted_flipped = evaluate_tree(
        tree, features, labels, class_weights=class_weights_flipped
    )
    assert weighted_flipped["objective"] == 6.0
    assert weighted["objective"] == 2.0  # class_weights[0]=1.0 leaves it unchanged


def test_acceptance_check_never_increases_the_local_objective():
    rng = np.random.default_rng(3)
    n, dim = 200, 5
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = rng.integers(0, 2, size=n).astype(np.int64)
    sample_weights = np.ones(n, dtype=np.float32)

    old_weights = rng.normal(size=(dim,)).astype(np.float32)
    old_bias = float(rng.normal())
    effective_lambda = 0.1

    for _ in range(20):
        # Fake solver proposal.
        new_weights = rng.normal(size=(dim,)).astype(np.float32)
        new_bias = float(rng.normal())

        old_objective = _reduced_problem_objective(
            features, labels, sample_weights, old_weights, old_bias, effective_lambda
        )
        chosen_weights, chosen_bias = _accept_or_reject_node_update(
            features, labels, sample_weights,
            old_weights, old_bias, new_weights, new_bias, effective_lambda,
        )
        chosen_objective = _reduced_problem_objective(
            features, labels, sample_weights, chosen_weights, chosen_bias, effective_lambda
        )
        assert chosen_objective <= old_objective + 1e-9


def test_acceptance_check_always_accepts_an_unfit_node():
    dim = 5
    features = np.zeros((10, dim), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)
    sample_weights = np.ones(10, dtype=np.float32)
    new_weights = np.ones(dim, dtype=np.float32)
    new_bias = 1.0

    chosen_weights, chosen_bias = _accept_or_reject_node_update(
        features, labels, sample_weights,
        np.zeros(dim, dtype=np.float32), 0.0,
        new_weights, new_bias, effective_lambda=0.1,
    )
    assert np.array_equal(chosen_weights, new_weights)
    assert chosen_bias == new_bias


def test_fit_tree_with_tao_runs_end_to_end_without_teacher_confidence():
    features, labels = _linearly_separable_dataset()
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=2, input_dim=features.shape[1]
    )
    history = fit_tree_with_tao(
        tree,
        features,
        labels,
        iterations=5,
        l1_lambda=1e-3,
        sparsity_alpha=0.0,
        random_state=0,
    )
    assert len(history) > 0
    final_accuracy = history[-1]["mimic_accuracy"]
    assert final_accuracy > 0.8  # near-linearly-separable data, small tree


def test_fit_tree_with_tao_does_not_deadlock_on_dominant_class():
    # Deadlock guard: dominant class + majority vote means no node ever fits.
    rng = np.random.default_rng(7)
    n, dim = 900, 6
    features = rng.normal(size=(n, dim)).astype(np.float32)
    true_weights = rng.normal(size=(dim,)).astype(np.float32)
    scores = features @ true_weights
    # Setup: minority at the bottom, dominant class on top.
    threshold = np.quantile(scores, 1 / 3)
    labels = np.where(scores <= threshold, rng.integers(1, 3, size=n), 0).astype(
        np.int64
    )
    majority_baseline = float((labels == 0).mean())

    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=3, input_dim=dim
    )
    history = fit_tree_with_tao(
        tree, features, labels, iterations=10, l1_lambda=1e-3, random_state=0
    )

    assert len(history) > 0
    final = history[-1]
    assert final["active_internal_nodes"] > 0
    assert len(set(tree.leaf_labels.tolist())) > 1
    assert final["mimic_accuracy"] > majority_baseline


if __name__ == "__main__":
    test_init_weight_scale_matches_papers_gaussian_0_1()
    test_initialize_tree_weights_uses_configured_scale()
    test_initialize_tree_weights_randomizes_leaf_labels()
    test_evaluate_tree_objective_is_class_weighted_when_given()
    test_acceptance_check_never_increases_the_local_objective()
    test_acceptance_check_always_accepts_an_unfit_node()
    test_fit_tree_with_tao_runs_end_to_end_without_teacher_confidence()
    test_fit_tree_with_tao_does_not_deadlock_on_dominant_class()
    print("OK")
