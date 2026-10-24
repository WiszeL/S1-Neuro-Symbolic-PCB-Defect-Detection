"""TAO fixes locked: init scale, weighted objective, accept-only-improvements."""

import io

import numpy as np
import torch

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.tao import (
    NODE_STATUSES,
    _accept_or_reject_node_update,
    _INIT_WEIGHT_SCALE,
    _reduced_problem_objective,
    evaluate_tree,
    fit_tree_with_tao,
    initialize_tree_weights,
    node_statistics,
    solve_l1_logistic_reduced_problem,
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
        chosen_weights, chosen_bias, _ = _accept_or_reject_node_update(
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

    chosen_weights, chosen_bias, accepted = _accept_or_reject_node_update(
        features, labels, sample_weights,
        np.zeros(dim, dtype=np.float32), 0.0,
        new_weights, new_bias, effective_lambda=0.1,
    )
    assert np.array_equal(chosen_weights, new_weights)
    assert chosen_bias == new_bias
    assert accepted


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

    nodes = history[0]["nodes"]
    assert len(nodes) == tree.num_internal_nodes
    assert {
        "node_index", "depth", "status", "samples_at_node", "care_count", "warm",
        "solver_iters", "stop_reason", "stop_ratio", "nonzero", "seconds",
    } <= nodes[0].keys()
    assert {node["status"] for node in nodes + history[1]["nodes"]} <= set(NODE_STATUSES)
    assert not any(node["warm"] for node in nodes)  # Iteration 1 of a fresh tree solves from zero.
    assert any(node["warm"] for node in history[1]["nodes"])
    buffer = io.BytesIO()  # The node log must survive the checkpoint round trip.
    torch.save({"history": history}, buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True)["history"][0]["nodes"] == nodes


def test_node_statistics_follow_the_routing():
    features, labels = _linearly_separable_dataset()
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=2, input_dim=features.shape[1]
    )
    fit_tree_with_tao(tree, features, labels, iterations=3, l1_lambda=1e-3, random_state=0)

    node_counts, leaf_counts = node_statistics(tree, features, labels)

    assert node_counts[0] == len(labels)
    for node in range(tree.num_internal_nodes):
        assert node_counts[node] == node_counts[tree.left_child(node)] + node_counts[tree.right_child(node)]
    assert [sum(row) for row in leaf_counts] == node_counts[tree.num_internal_nodes :]
    assert np.sum(leaf_counts, axis=0).tolist() == np.bincount(labels).tolist()


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


def _liblinear_objective(features, labels, sample_weights, weights, bias, l1_lambda):
    """sklearn-liblinear L1R_LR objective in float64: bias penalized like a weight."""
    sign = np.where(labels == 1, 1.0, -1.0)
    margin = sign * (features.astype(np.float64) @ weights.astype(np.float64) + bias)
    loss = np.sum(sample_weights / l1_lambda * np.logaddexp(0.0, -margin))
    return float(loss + np.abs(weights).sum() + abs(bias))


def _sparse_logistic_problem():
    rng = np.random.default_rng(0)
    n, dim = 3000, 40
    features = rng.normal(size=(n, dim)).astype(np.float32)
    true_weights = np.zeros(dim, dtype=np.float32)
    true_weights[:5] = 2.0 * rng.normal(size=5)
    noisy = features @ true_weights + 0.5 * rng.normal(size=n)
    labels = (noisy > 0).astype(np.int64)
    sample_weights = rng.uniform(0.2, 2.0, size=n).astype(np.float32)
    return features, labels, sample_weights


def test_l1_logistic_solver_reaches_liblinear_optimum():
    from sklearn.linear_model import LogisticRegression  # Reference only.

    features, labels, sample_weights = _sparse_logistic_problem()
    dim, l1_lambda = features.shape[1], 50.0

    reference = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0 / l1_lambda, tol=1e-6, max_iter=1000
    ).fit(features, labels, sample_weight=sample_weights)
    ref_weights = reference.coef_[0].astype(np.float32)
    ref_bias = float(reference.intercept_[0])

    weights, bias, info = solve_l1_logistic_reduced_problem(
        features, labels, sample_weights,
        l1_lambda=l1_lambda, tolerance=1e-5,  # This toy problem's float32 floor sits above 1e-6.
    )

    ours = _liblinear_objective(features, labels, sample_weights, weights, bias, l1_lambda)
    theirs = _liblinear_objective(
        features, labels, sample_weights, ref_weights, ref_bias, l1_lambda
    )
    assert info["stop_reason"] == "converged"
    assert ours <= theirs * (1 + 1e-4)
    assert np.sum(weights == 0.0) >= dim // 2  # The L1 penalty actually bites.
    ours_left = (features @ weights + bias) >= 0.0
    theirs_left = (features @ ref_weights + ref_bias) >= 0.0
    assert (ours_left == theirs_left).mean() >= 0.995


def test_warm_start_reaches_the_same_optimum_in_fewer_iterations():
    features, labels, sample_weights = _sparse_logistic_problem()
    kwargs = dict(l1_lambda=50.0, tolerance=1e-5)  # Float32 floor of this toy problem sits above 1e-6.
    cold_w, cold_b, cold = solve_l1_logistic_reduced_problem(
        features, labels, sample_weights, **kwargs
    )
    noise = np.random.default_rng(1).normal(size=cold_w.shape).astype(np.float32)
    warm_w, warm_b, warm = solve_l1_logistic_reduced_problem(
        features, labels, sample_weights,
        init=(cold_w + 0.1 * noise, cold_b + 0.1), **kwargs,
    )
    _, _, at_optimum = solve_l1_logistic_reduced_problem(
        features, labels, sample_weights, init=(cold_w, cold_b), **kwargs
    )
    f_cold = _liblinear_objective(features, labels, sample_weights, cold_w, cold_b, 50.0)
    f_warm = _liblinear_objective(features, labels, sample_weights, warm_w, warm_b, 50.0)
    # Production rule: float32 can end a fit "abnormal" at its floor, never at the cap, and within ratio 30.
    assert warm["stop_reason"] in ("converged", "abnormal") and warm["stop_ratio"] <= 30
    assert abs(f_warm - f_cold) <= 1e-4 * f_cold
    assert at_optimum["iterations"] < cold["iterations"]


if __name__ == "__main__":
    test_init_weight_scale_matches_papers_gaussian_0_1()
    test_initialize_tree_weights_uses_configured_scale()
    test_initialize_tree_weights_randomizes_leaf_labels()
    test_evaluate_tree_objective_is_class_weighted_when_given()
    test_acceptance_check_never_increases_the_local_objective()
    test_acceptance_check_always_accepts_an_unfit_node()
    test_fit_tree_with_tao_runs_end_to_end_without_teacher_confidence()
    test_node_statistics_follow_the_routing()
    test_fit_tree_with_tao_does_not_deadlock_on_dominant_class()
    test_l1_logistic_solver_reaches_liblinear_optimum()
    test_warm_start_reaches_the_same_optimum_in_fewer_iterations()
    print("OK")
