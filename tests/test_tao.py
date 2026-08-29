"""Smoke checks for the TAO fidelity fixes: N(0,1) init, class-weighted
convergence objective, and the cheap per-node acceptance check that never
lets a node's own reduced-problem objective get worse."""

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
    # A N(0, 1) sample of this size should not look like a N(0, 1e-3) sample.
    assert np.std(tree.node_weights) > 0.3


def test_initialize_tree_weights_randomizes_leaf_labels():
    # Both papers randomize leaf labels alongside hyperplanes (Hada Fig.1 /
    # Kairgeldin Fig.5). A tree with several leaves and several classes
    # should not have every leaf land on the same label by chance.
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

    # Two misclassified 0-labeled samples, weight 1.0 each -> unweighted loss 2.0.
    assert unweighted["objective"] == 2.0
    # Same two samples, but weighted by class_weights[0] = 1.0 -> unchanged here;
    # flip the weighting to confirm it actually reads class_weights[labels].
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
        # A random "candidate" fit standing in for what the L1-logistic
        # surrogate solver might occasionally propose.
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
    # Regression test for the cold-start deadlock: when one class dominates
    # (here ~67%, matching neg_ratio=2 background fraction), a majority-vote
    # leaf init makes every leaf agree on the dominant class, zeroing every
    # node's |left_loss - right_loss| split signal so no node can ever fit.
    # This must not happen: fails on the pre-fix majority-vote leaf init,
    # passes with initialize_tree_weights's random leaf-label bootstrap.
    rng = np.random.default_rng(7)
    n, dim = 900, 6
    features = rng.normal(size=(n, dim)).astype(np.float32)
    true_weights = rng.normal(size=(dim,)).astype(np.float32)
    scores = features @ true_weights
    # Bottom third minority classes 1/2, top two-thirds dominant class 0.
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
