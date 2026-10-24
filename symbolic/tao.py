from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .sodt import SparseObliqueDecisionTreeClassifier
from util.features import ensure_float32


# 1024 rows x 3136 x 4 B = 12.8 MB stays in the i5-13500's 24 MB L3 between the solver's two matvecs.
_TAO_CHUNK_SIZE = 1024

# The solver is memory-bound: 20 BLAS threads contend for the same memory channels. Measured on a
# 120k-row node: 62 ms per evaluation at 6 threads vs 112 ms at 20.
_SOLVER_BLAS_THREADS = 6


# Gaussian init per both papers; the scale only picks which local optimum the first fit lands in.
_INIT_WEIGHT_SCALE = 1.0

# Every value record["status"] can take; the dashboard and tests are keyed on it.
NODE_STATUSES = ("accepted", "rejected", "no_split_gain", "empty")

# Solver info for a node that never reached the solver (empty, or nothing to gain).
_NO_SOLVE = {"iterations": 0, "stop_reason": None, "stop_ratio": None}


def initialize_tree_weights(
    tree: SparseObliqueDecisionTreeClassifier,
    random_state: int = 42,
) -> None:
    generator = np.random.default_rng(random_state)
    tree.node_weights = generator.normal(
        loc=0.0,
        scale=_INIT_WEIGHT_SCALE,
        size=tree.node_weights.shape,
    ).astype(np.float32)
    tree.node_bias = generator.normal(
        loc=0.0,
        scale=_INIT_WEIGHT_SCALE,
        size=tree.node_bias.shape,
    ).astype(np.float32)
    # Random leaf labels too — a majority vote would deadlock TAO on dominant classes.
    tree.leaf_labels = generator.integers(
        0, tree.num_classes, size=tree.leaf_labels.shape
    ).astype(np.int64)


def _is_contiguous_indices(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return True
    return bool(
        np.all(
            indices
            == np.arange(indices[0], indices[0] + indices.size, dtype=indices.dtype)
        )
    )


def _select_rows(source: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, source.shape[1]), dtype=source.dtype)

    # Slices stay on disk when rows are contiguous.
    if _is_contiguous_indices(indices):
        return source[int(indices[0]) : int(indices[-1]) + 1]

    # Read in chunks so scattered rows don't stall on disk.
    if indices.size <= _TAO_CHUNK_SIZE:
        return source[indices]

    out = np.empty((indices.size, source.shape[1]), dtype=source.dtype)
    for start in range(0, indices.size, _TAO_CHUNK_SIZE):
        stop = min(start + _TAO_CHUNK_SIZE, indices.size)
        out[start:stop] = source[indices[start:stop]]
    return out


def _score_rows_for_node(
    features: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: float,
    chunk_size: int = _TAO_CHUNK_SIZE,
) -> np.ndarray:
    scores = np.empty((indices.size,), dtype=np.float32)
    for start in range(0, indices.size, chunk_size):
        stop = min(start + chunk_size, indices.size)
        rows = _select_rows(features, indices[start:stop])
        scores[start:stop] = (rows @ weights) + bias
    return scores


def _predict_from_node_for_indices(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    indices: np.ndarray,
    node_index: int,
    chunk_size: int = _TAO_CHUNK_SIZE,
) -> np.ndarray:
    predictions = np.empty((indices.size,), dtype=np.int64)
    for start in range(0, indices.size, chunk_size):
        stop = min(start + chunk_size, indices.size)
        rows = _select_rows(features, indices[start:stop])
        predictions[start:stop] = tree.predict_from_node(rows, node_index)
    return predictions


def compute_reduced_sets(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
) -> dict[int, np.ndarray]:
    reduced_sets: dict[int, np.ndarray] = {
        0: np.arange(features.shape[0], dtype=np.int64)
    }

    for node_index in range(tree.num_internal_nodes):
        node_indices = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))
        if node_indices.size == 0:
            reduced_sets[tree.left_child(node_index)] = np.zeros((0,), dtype=np.int64)
            reduced_sets[tree.right_child(node_index)] = np.zeros((0,), dtype=np.int64)
            continue

        scores = _score_rows_for_node(
            features,
            node_indices,
            tree.node_weights[node_index],
            tree.node_bias[node_index],
        )
        go_left = scores >= 0.0
        reduced_sets[tree.left_child(node_index)] = node_indices[go_left]
        reduced_sets[tree.right_child(node_index)] = node_indices[~go_left]

    for leaf_offset in range(tree.num_leaves):
        reduced_sets.setdefault(
            tree.num_internal_nodes + leaf_offset, np.zeros((0,), dtype=np.int64)
        )

    return reduced_sets


def node_statistics(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[int], list[list[int]]]:
    """Rows reaching every node, and each leaf's class counts (what the tree diagram prints)."""
    reduced_sets = compute_reduced_sets(tree, features)
    node_sample_counts = [
        int(reduced_sets[node].size)
        for node in range(tree.num_internal_nodes + tree.num_leaves)
    ]
    leaf_class_counts = [
        np.bincount(
            labels[reduced_sets[tree.num_internal_nodes + leaf]],
            minlength=tree.num_classes,
        ).tolist()
        for leaf in range(tree.num_leaves)
    ]
    return node_sample_counts, leaf_class_counts


def update_leaf_predictions(
    tree: SparseObliqueDecisionTreeClassifier,
    labels: np.ndarray,
    reduced_sets: dict[int, np.ndarray],
    class_weights: np.ndarray | None = None,
) -> None:
    for leaf_offset in range(tree.num_leaves):
        node_index = tree.num_internal_nodes + leaf_offset
        node_indices = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))
        if node_indices.size == 0:
            continue

        counts = np.bincount(labels[node_indices], minlength=tree.num_classes).astype(
            np.float32
        )
        # Weighting after counting is exact.
        if class_weights is not None:
            counts = counts * class_weights.astype(np.float32)
        tree.leaf_labels[leaf_offset] = int(counts.argmax())


def _logistic_loss_and_grad(
    weights: np.ndarray,
    bias: float,
    features: np.ndarray,
    signs: np.ndarray,
    costs: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Cost-weighted logistic loss and its gradient in one chunked float32 pass."""
    weights32 = weights.astype(np.float32)
    loss = 0.0
    grad_w = np.zeros(weights.shape, dtype=np.float64)
    grad_b = 0.0
    for start in range(0, features.shape[0], _TAO_CHUNK_SIZE):
        stop = start + _TAO_CHUNK_SIZE
        rows = features[start:stop]
        sign, cost = signs[start:stop], costs[start:stop]
        margin = sign * ((rows @ weights32) + bias)
        loss += float(cost @ np.logaddexp(0.0, -margin))
        residual = -cost * sign * expit(-margin)
        grad_w += rows.T @ residual.astype(np.float32)
        grad_b += residual.sum()
    return loss, grad_w, grad_b


def _l1_violation(z: np.ndarray, grad: np.ndarray) -> float:
    """Projected-gradient L1 norm over z >= 0, the quantity LIBLINEAR's L1 stop test shrinks."""
    return float(np.abs(np.where(z > 0.0, grad, np.minimum(grad, 0.0))).sum())


def solve_l1_logistic_reduced_problem(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    l1_lambda: float = 0.1,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    zero_threshold: float = 1e-5,
    init: tuple[np.ndarray, float] | None = None,
    solver_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """LIBLINEAR's L1R_LR problem, solved without copying `features`.

    min  sum_i (s_i / lambda) * log(1 + exp(-y_i (w.x_i + b)))  +  |w|_1 + |b|
    The bias is penalized like sklearn's liblinear does. L-BFGS-B on w = p - q,
    b = bp - bq (all >= 0), one chunked pass over `features` per evaluation.
    `init` = (weights, bias) warm-starts the (convex) fit: same optimum, fewer iterations.
    """
    dim = features.shape[1]
    positive = labels == 1
    positives = int(positive.sum())
    if positives in (0, labels.size):  # One class: no split signal, constant answer.
        bias = 1.0 if positives else -1.0
        info = {"iterations": 0, "stop_reason": "trivial", "surrogate": None, "stop_ratio": None}
        return np.zeros((dim,), dtype=np.float32), bias, info

    costs = sample_weights.astype(np.float64) / max(float(l1_lambda), 1e-12)
    signs = np.where(positive, 1.0, -1.0)
    # LIBLINEAR stops once the violation falls to eps x its value at w = 0.
    eps = tolerance * max(min(positives, labels.size - positives), 1) / labels.size
    state: dict[str, Any] = {"iterations": 0}
    # scipy keeps `fun` alive in a reference cycle past this call; clearing this list frees the batch.
    data = [features]

    def unpack(z: np.ndarray) -> tuple[np.ndarray, float]:
        return z[:dim] - z[dim : 2 * dim], z[2 * dim] - z[2 * dim + 1]

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = unpack(z)
        loss, grad_w, grad_b = _logistic_loss_and_grad(w, b, data[0], signs, costs)
        grad = np.concatenate(
            [grad_w + 1.0, 1.0 - grad_w, [grad_b + 1.0, 1.0 - grad_b]]
        )
        state["surrogate"], state["grad"] = loss + z.sum(), grad
        if "initial" not in state:  # First evaluation is at w = 0.
            state["initial"] = _l1_violation(z, grad)
        return state["surrogate"], grad

    def on_iteration(z: np.ndarray) -> None:
        state["iterations"] += 1
        violation = _l1_violation(z, state["grad"])  # grad is from z's own evaluation.
        target = eps * state["initial"]
        # Taken here, not after minimize: a failed line search leaves result.x behind the last fun() call.
        state["stop_ratio"] = violation / max(target, 1e-300)
        if solver_callback is not None and state["iterations"] % 10 == 0:
            solver_callback(
                {
                    "solver_iter": state["iterations"],
                    "surrogate": state["surrogate"],
                    "stop_ratio": state["stop_ratio"],
                }
            )
        if violation <= target:
            state["converged"] = True
            raise StopIteration

    start = np.zeros(2 * dim + 2)
    try:
        with threadpool_limits(limits=_SOLVER_BLAS_THREADS, user_api="blas"):
            if init is not None:
                fun(start)  # The stop target stays defined at w = 0, as in LIBLINEAR, wherever we start.
                w0, b0 = init[0].astype(np.float64), init[1]
                start[:dim], start[dim : 2 * dim] = np.maximum(w0, 0.0), np.maximum(-w0, 0.0)
                start[2 * dim], start[2 * dim + 1] = max(b0, 0.0), max(-b0, 0.0)
            result = minimize(
                fun,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=Bounds(0.0, np.inf),
                callback=on_iteration,
                # ftol/gtol 0: only the LIBLINEAR rule above (or the cap) ends the fit.
                options={"maxiter": max_iter, "maxfun": 5 * max_iter, "ftol": 0.0, "gtol": 0.0},
            )
    finally:
        data.clear()

    if state.get("converged") or state["initial"] == 0.0:
        stop_reason = "converged"
    else:
        stop_reason = "cap" if result.status == 1 else "abnormal"
    learned_weights, learned_bias = unpack(result.x)
    learned_weights = learned_weights.astype(np.float32)
    learned_weights[np.abs(learned_weights) < zero_threshold] = 0.0
    info = {
        "iterations": state["iterations"],
        "stop_reason": stop_reason,
        "surrogate": float(result.fun),
        # <= 1 means converged; None = the line search failed before the first iteration finished.
        "stop_ratio": state.get("stop_ratio", 0.0 if state["initial"] == 0.0 else None),
    }
    return learned_weights, float(learned_bias), info


def _reduced_problem_objective(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    bias: float,
    effective_lambda: float,
) -> float:
    """This node's own score: mistakes plus sparsity penalty."""
    scores = _score_rows_for_node(features, np.arange(features.shape[0]), weights, bias)
    go_left = scores >= 0.0  # Same float32 score and left-rule as sodt.py.
    loss = float(np.sum((go_left != labels) * sample_weights, dtype=np.float64))
    penalty = float(effective_lambda) * float(np.sum(np.abs(weights), dtype=np.float64))
    return loss + penalty


def _accept_or_reject_node_update(
    batch_features: np.ndarray,
    batch_labels: np.ndarray,
    batch_sample_weights: np.ndarray,
    old_weights: np.ndarray,
    old_bias: float,
    new_weights: np.ndarray,
    new_bias: float,
    effective_lambda: float,
) -> tuple[np.ndarray, float, bool]:
    """Keep the new fit only if this node's own score didn't get worse: (weights, bias, accepted)."""
    if not np.any(old_weights) and old_bias == 0.0:
        return new_weights, new_bias, True  # No baseline yet, so the first fit always wins.
    old_objective = _reduced_problem_objective(
        batch_features, batch_labels, batch_sample_weights,
        old_weights, old_bias, effective_lambda,
    )
    new_objective = _reduced_problem_objective(
        batch_features, batch_labels, batch_sample_weights,
        new_weights, new_bias, effective_lambda,
    )
    if new_objective <= old_objective:
        return new_weights, new_bias, True
    return old_weights, old_bias, False


def evaluate_tree(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    l1_lambda: float = 0.0,
    sparsity_alpha: float = 0.0,
    reduced_sets: dict[int, np.ndarray] | None = None,
    class_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    predictions = tree.predict(features)
    accuracy = float((predictions == labels).mean())
    nonzero_counts = tree.nonzero_weight_counts()

    # Full training score for the convergence check, weighted like training.
    misclassified = (predictions != labels).astype(np.float64)
    if class_weights is not None:
        misclassified *= class_weights[labels]
    classification_loss = float(misclassified.sum())
    l1_penalty = 0.0
    if l1_lambda > 0.0:
        if reduced_sets is None:
            reduced_sets = compute_reduced_sets(tree, features)
        for node_index in range(tree.num_internal_nodes):
            node_rs_size = reduced_sets.get(
                node_index, np.zeros((0,), dtype=np.int64)
            ).size
            node_l1 = float(np.sum(np.abs(tree.node_weights[node_index])))
            h_alpha = float(max(node_rs_size, 1) ** sparsity_alpha)
            l1_penalty += l1_lambda * h_alpha * node_l1
    objective = classification_loss + l1_penalty

    return {
        "mimic_accuracy": accuracy,
        "nonzero_weights": sum(nonzero_counts),
        "mean_nonzero_per_node": float(np.mean(nonzero_counts))
        if nonzero_counts
        else 0.0,
        "active_internal_nodes": sum(count > 0 for count in nonzero_counts),
        "objective": objective,
    }


def raise_if_tree_collapsed(tree: SparseObliqueDecisionTreeClassifier) -> None:
    """Refuse one-class-for-everything trees — they'd train fine and fail silently."""
    nonzero_counts = tree.nonzero_weight_counts()
    active_internal_nodes = sum(count > 0 for count in nonzero_counts)
    distinct_leaf_labels = len(set(tree.leaf_labels.tolist()))
    if active_internal_nodes == 0 or distinct_leaf_labels <= 1:
        raise RuntimeError(
            f"Tree collapsed: {active_internal_nodes} active internal nodes, "
            f"{distinct_leaf_labels} distinct leaf label(s) — it predicts one "
            "class for every input. Likely a TAO cold-start deadlock (a "
            "dominant class, e.g. background under a low neg_ratio, made "
            "every leaf's majority vote agree, so no node ever had a split "
            "signal). Not a valid trained model; do not use this checkpoint."
        )


def postprocess_tree(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...] | None = None,
    verbose: bool = False,
    class_weights: np.ndarray | None = None,
) -> int:
    reduced_sets = compute_reduced_sets(tree, features)
    update_leaf_predictions(tree, labels, reduced_sets, class_weights=class_weights)

    # Nothing ever reaches this leaf, so don't let it guess.
    for leaf_offset in range(tree.num_leaves):
        leaf_node = tree.num_internal_nodes + leaf_offset
        if reduced_sets.get(leaf_node, np.zeros((0,), dtype=np.int64)).size == 0:
            tree.leaf_labels[leaf_offset] = 0

    if verbose:
        print(f"Walking {tree.num_internal_nodes} internal nodes (reverse BFS)...")

    pruned_count = 0

    def _subtree_leaf_labels(node_index: int) -> set[int]:
        if tree.is_leaf_node(node_index):
            leaf_pos = tree.leaf_position(node_index)
            return {int(tree.leaf_labels[leaf_pos])}
        left = tree.left_child(node_index)
        right = tree.right_child(node_index)
        return _subtree_leaf_labels(left) | _subtree_leaf_labels(right)

    def _zero_subtree(node_index: int) -> None:
        if tree.is_leaf_node(node_index):
            return
        tree.node_weights[node_index] = 0.0
        tree.node_bias[node_index] = 0.0
        _zero_subtree(tree.left_child(node_index))
        _zero_subtree(tree.right_child(node_index))

    for node_index in range(tree.num_internal_nodes - 1, -1, -1):
        node_rs = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))

        if node_rs.size == 0:
            if np.any(tree.node_weights[node_index] != 0.0):
                if verbose:
                    depth = tree.depth_of(node_index)
                    print(
                        f"  Node {node_index} (depth {depth}): pruned — dead branch (0 samples)"
                    )
                _zero_subtree(node_index)
                pruned_count += 1
            continue

        reachable_labels = _subtree_leaf_labels(node_index)
        if len(reachable_labels) == 1:
            if verbose:
                depth = tree.depth_of(node_index)
                if class_names:
                    leaf_class = class_names[next(iter(reachable_labels))]
                    print(
                        f"  Node {node_index} (depth {depth}): pruned — pure subtree (all leaves → {leaf_class})"
                    )
                else:
                    print(f"  Node {node_index} (depth {depth}): pruned — pure subtree")
            _zero_subtree(node_index)
            pruned_count += 1

    return pruned_count


def fit_tree_with_tao(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int = 40,
    l1_lambda: float = 1e-3,
    sparsity_alpha: float = 0.0,
    logistic_max_iter: int = 200,
    tolerance: float = 1e-4,
    zero_threshold: float = 1e-5,
    convergence_tolerance: float = 1e-6,
    random_state: int = 42,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    node_progress_callback: Callable[[dict[str, Any]], None] | None = None,
    class_weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    features = ensure_float32(features)
    labels = np.asarray(labels, dtype=np.int64)

    if features.ndim != 2:
        raise ValueError(
            "TAO expects a 2D feature matrix of shape [num_samples, num_features]."
        )

    freshly_initialized = np.allclose(tree.node_weights, 0.0)
    if freshly_initialized:
        initialize_tree_weights(tree, random_state=random_state)

    history: list[dict[str, Any]] = []
    iteration_range = range(iterations)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(iteration_range, desc=progress_desc or "TAO", leave=False)
        iteration_range = progress_bar

    # Leaf labels carry over from last iteration's refresh — recomputing here repeats the same work.
    # Fresh trees keep their random leaves instead; voting now would deadlock TAO on dominant classes.
    reduced_sets = compute_reduced_sets(tree, features)
    if not freshly_initialized:
        update_leaf_predictions(tree, labels, reduced_sets, class_weights=class_weights)

    def emit(event: dict[str, Any], **fields: Any) -> None:
        if node_progress_callback is not None:
            node_progress_callback({**event, **fields})

    def solve_node(
        node_index: int,
        node_indices: np.ndarray,
        node_event: dict[str, Any],
        warm: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Refit one node: (status, solver_info). Its batch dies with this frame."""
        if node_indices.size == 0:
            tree.node_weights[node_index] = 0.0
            tree.node_bias[node_index] = 0.0
            return "empty", _NO_SOLVE

        emit(node_event, phase="stage", stage="routing")
        node_labels = labels[node_indices]
        left_predictions = _predict_from_node_for_indices(
            tree, features, node_indices, tree.left_child(node_index)
        )
        right_predictions = _predict_from_node_for_indices(
            tree, features, node_indices, tree.right_child(node_index)
        )

        left_loss = (left_predictions != node_labels).astype(np.float32)
        right_loss = (right_predictions != node_labels).astype(np.float32)
        sample_weights = np.abs(left_loss - right_loss)

        # Misrouting weighted classes costs more, so boundaries lean away from background.
        if class_weights is not None:
            sample_weights *= class_weights[node_labels].astype(np.float32)

        positive_weight_mask = sample_weights > 0.0
        care_count = int(positive_weight_mask.sum())
        node_event["care_count"] = care_count
        if care_count == 0:
            return "no_split_gain", _NO_SOLVE

        pseudolabels = (left_loss <= right_loss).astype(np.int64)

        # Penalty sized by this node's sample count (Kairgeldin's alpha weighting).
        effective_lambda = float(
            l1_lambda * float(max(node_indices.size, 1) ** sparsity_alpha)
        )

        emit(node_event, phase="stage", stage="loading")
        # Only this node's care rows leave the disk; the solver reads them in chunks.
        batch_features = _select_rows(features, node_indices[positive_weight_mask])
        batch_labels = pseudolabels[positive_weight_mask]
        batch_sample_weights = sample_weights[positive_weight_mask]

        old_weights = tree.node_weights[node_index].copy()
        old_bias = float(tree.node_bias[node_index])
        emit(node_event, phase="stage", stage="solving")
        weights, bias, solver_info = solve_l1_logistic_reduced_problem(
            batch_features,
            batch_labels,
            batch_sample_weights,
            l1_lambda=effective_lambda,
            max_iter=logistic_max_iter,
            tolerance=tolerance,
            zero_threshold=zero_threshold,
            init=(old_weights, old_bias) if warm else None,
            solver_callback=lambda info: emit(node_event, phase="solver", **info),
        )

        # Accept
        chosen_weights, chosen_bias, accepted = _accept_or_reject_node_update(
            batch_features,
            batch_labels,
            batch_sample_weights,
            old_weights,
            old_bias,
            weights,
            bias,
            effective_lambda,
        )
        tree.node_weights[node_index] = chosen_weights
        tree.node_bias[node_index] = chosen_bias
        return ("accepted" if accepted else "rejected"), solver_info

    for iteration_index in iteration_range:
        iteration_start = perf_counter()
        # Iteration 1 of a fresh tree solves from zero, not from the random init.
        warm = iteration_index > 0 or not freshly_initialized
        node_records: list[dict[str, Any]] = []

        # Solve nodes
        for node_index in range(tree.num_internal_nodes - 1, -1, -1):
            node_started = perf_counter()
            node_indices = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))
            node_event = {
                "phase": "start",
                "iteration": iteration_index + 1,
                "iterations": iterations,
                "node_index": node_index,
                "num_internal_nodes": tree.num_internal_nodes,
                "depth": tree.depth_of(node_index),
                "samples_at_node": int(node_indices.size),
                "care_count": 0,  # solve_node fills this in once it has counted.
            }
            emit(node_event)

            status, solver_info = solve_node(node_index, node_indices, node_event, warm)
            record = {
                "node_index": node_index,
                "depth": node_event["depth"],
                "status": status,
                "samples_at_node": node_event["samples_at_node"],
                "care_count": node_event["care_count"],
                "warm": warm and node_event["care_count"] > 0,
                "solver_iters": solver_info["iterations"],
                "stop_reason": solver_info["stop_reason"],
                "stop_ratio": solver_info["stop_ratio"],
                "nonzero": int(np.count_nonzero(tree.node_weights[node_index])),
                "seconds": perf_counter() - node_started,
            }
            node_records.append(record)
            emit(node_event, **record, phase="end")

        # Refresh routing
        reduced_sets = compute_reduced_sets(tree, features)
        update_leaf_predictions(tree, labels, reduced_sets, class_weights=class_weights)

        metrics = evaluate_tree(
            tree,
            features,
            labels,
            l1_lambda=l1_lambda,
            sparsity_alpha=sparsity_alpha,
            reduced_sets=reduced_sets,
            class_weights=class_weights,
        )
        metrics["iteration"] = iteration_index + 1
        metrics["nodes"] = node_records
        metrics["duration_seconds"] = perf_counter() - iteration_start
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(metrics)
        if progress_bar is not None:
            progress_bar.set_postfix(
                mimic=f"{metrics['mimic_accuracy']:.4f}",
                nz=metrics["nonzero_weights"],
                obj=f"{metrics['objective']:.1f}",
            )

        # Stop when the score barely moves.
        if len(history) >= 2:
            prev_obj = history[-2]["objective"]
            curr_obj = metrics["objective"]
            relative_decrease = abs(prev_obj - curr_obj) / max(abs(prev_obj), 1e-12)
            if relative_decrease < convergence_tolerance:
                if progress_bar is not None:
                    progress_bar.set_postfix_str(
                        f"converged (Δobj={relative_decrease:.2e})"
                    )
                    progress_bar.close()
                break

    return history
