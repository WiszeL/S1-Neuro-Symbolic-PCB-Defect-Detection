from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .sodt import SparseObliqueDecisionTreeClassifier
from util.features import ensure_float32


_TAO_CHUNK_SIZE = 2048


# Both papers specify Gaussian(0,1) init at the decision nodes (Hada Fig.1 /
# Kairgeldin Fig.5); TAO's first iteration refits every node from scratch
# anyway, so the scale only affects which local optimum the first fit lands in.
_INIT_WEIGHT_SCALE = 1.0


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
    # Both papers randomize leaf labels too (Hada Fig.1 / Kairgeldin Fig.5).
    # This matters beyond fidelity: a majority-vote leaf init makes every leaf
    # agree on the dominant class whenever one class dominates the data (e.g.
    # background under neg_ratio subsampling), which zeroes every node's
    # |left_loss - right_loss| signal and deadlocks TAO before it starts.
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

    # Optimization: if indices are contiguous, we can use a slice which is
    # extremely fast and returns a view if source is a memmap.
    if _is_contiguous_indices(indices):
        return source[int(indices[0]) : int(indices[-1]) + 1]

    # Chunked reads prevent massive numpy IO scatter-gather freezes
    # when selecting thousands of fragmented rows from a memmap.
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
        # Per-class weights scale counts uniformly within each class, so
        # weighting after binning is exact. Weighted counts feed both the
        # leaf label and the distribution so argmax semantics stay consistent.
        if class_weights is not None:
            counts = counts * class_weights.astype(np.float32)
        tree.leaf_labels[leaf_offset] = int(counts.argmax())
        tree.leaf_distributions[leaf_offset] = (counts + tree.leaf_smoothing) / (
            counts.sum() + (tree.leaf_smoothing * tree.num_classes)
        )


def _prepare_reduced_problem_batch(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    indices: np.ndarray | None = None,
    random_state: int = 42,
    max_solver_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Cap and materialize the rows a node's reduced problem will fit on.

    Split out from the solver so the acceptance check (below) can score the
    old vs. new weights on the exact same batch without a second data read.
    Returns None for an empty reduced set.
    """
    if labels.size == 0:
        return None

    # ---- Subsample cap for LIBLINEAR memory safety ----
    # max_solver_samples > 0: cap to that many samples
    # max_solver_samples == 0: no cap (use all samples)
    # The regularization C is computed from the actual node size (N_i),
    # so this cap only affects solver sample count, not regularization strength.
    n = labels.size
    if max_solver_samples > 0 and n > max_solver_samples:
        rng = np.random.RandomState(random_state)
        keep = np.sort(rng.choice(n, size=max_solver_samples, replace=False))
        labels = labels[keep]
        sample_weights = sample_weights[keep]
        if indices is not None:
            indices = indices[keep]
        else:
            features = features[keep]

    # Avoid materializing the full 'features' memmap.
    # We only materialize the specific rows needed for this node.
    if indices is not None:
        features = _select_rows(features, indices)
    else:
        features = np.asarray(features, dtype=np.float32)

    labels = np.asarray(labels, dtype=np.int64)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    return features, labels, sample_weights


def solve_l1_logistic_reduced_problem(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    indices: np.ndarray | None = None,
    l1_lambda: float = 0.1,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    zero_threshold: float = 1e-5,
    random_state: int = 42,
    max_solver_samples: int = 0,
) -> tuple[np.ndarray, float]:
    batch = _prepare_reduced_problem_batch(
        features, labels, sample_weights, indices, random_state, max_solver_samples
    )
    if batch is None:
        dim = features.shape[1]
        return np.zeros((dim,), dtype=np.float32), 0.0

    batch_features, batch_labels, batch_sample_weights = batch

    unique_labels = np.unique(batch_labels)
    if unique_labels.size == 1:
        bias = 1.0 if int(unique_labels[0]) == 1 else -1.0
        return np.zeros((batch_features.shape[1],), dtype=np.float32), float(bias)

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the paper-faithful TAO reduced-problem solver."
        ) from exc

    # ---------------------------------------------------------------------
    # Match the paper solver family: L1 logistic regression with LIBLINEAR
    # ---------------------------------------------------------------------
    # Kairgeldin eq. 4: RP_i = Σ loss + λ·|R_i|^α · ‖w‖₁
    # LIBLINEAR minimises: ‖w‖₁ + C·Σ loss, so 1/C = λ·|R_i|^α.
    # The caller pre-computes the effective λ (= λ·N_i^α) and passes it
    # as l1_lambda.  We set C = 1 / l1_lambda.

    # We convert directly to float64 here. This is what LIBLINEAR requires.
    # By doing it now and letting the previous 'features' reference go,
    # we avoid having both float32 and float64 copies in RAM simultaneously.
    features_64 = batch_features.astype(np.float64)

    effective_lambda = max(float(l1_lambda), 1e-12)
    effective_C = max(1.0 / effective_lambda, 1e-12)
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        fit_intercept=True,
        C=effective_C,
        random_state=random_state,
        tol=tolerance,
        max_iter=max_iter,
    )

    model.fit(features_64, batch_labels, sample_weight=batch_sample_weights)

    learned_weights = model.coef_[0].astype(np.float32)
    learned_weights[np.abs(learned_weights) < zero_threshold] = 0.0
    bias = float(model.intercept_[0])

    return learned_weights, bias


def _reduced_problem_objective(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    bias: float,
    effective_lambda: float,
) -> float:
    """RP_i(θ) = Σ weighted 0/1-loss + λ·‖w‖₁ (Kairgeldin eq. 2/4)."""
    scores = features.astype(np.float64) @ weights.astype(np.float64) + bias
    predicted_left = (scores >= 0.0).astype(np.int64)  # matches sodt.py routing
    loss = float(
        np.sum((predicted_left != labels).astype(np.float64) * sample_weights)
    )
    penalty = float(effective_lambda) * float(np.sum(np.abs(weights)))
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
) -> tuple[np.ndarray, float]:
    """Keep the new fit only if it does not increase the node's own RP
    objective (Hada Fig.1 / Kairgeldin App.A.2: "accepting updates only if
    they improve (2)"). The L1-logistic surrogate can occasionally make the
    true weighted 0/1 objective worse; this is the cheap local guard against
    that, using the exact batch already fit — no full-tree re-evaluation.
    """
    if not np.any(old_weights) and old_bias == 0.0:
        return new_weights, new_bias  # nothing fit yet at this node
    old_objective = _reduced_problem_objective(
        batch_features, batch_labels, batch_sample_weights,
        old_weights, old_bias, effective_lambda,
    )
    new_objective = _reduced_problem_objective(
        batch_features, batch_labels, batch_sample_weights,
        new_weights, new_bias, effective_lambda,
    )
    if new_objective <= old_objective:
        return new_weights, new_bias
    return old_weights, old_bias


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

    # Compute the full TAO objective: Σ 0/1-loss + λ Σ_i |R_i|^α ‖w_i‖₁
    # (Kairgeldin eq. 3).  Used for convergence checking. Weighted by
    # class_weights when training is class-weighted, so the convergence
    # check tracks the objective TAO actually minimizes, not a different one.
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
    """Guard against silently saving a dead tree.

    A collapsed tree (every internal node pruned, or every leaf voting the
    same class) predicts one label for all inputs. It trains and evaluates
    without error, so nothing else in the pipeline would ever flag it.
    """
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
                    depth = int(np.floor(np.log2(node_index + 1)))
                    print(
                        f"  Node {node_index} (depth {depth}): pruned — dead branch (0 samples)"
                    )
                _zero_subtree(node_index)
                pruned_count += 1
            continue

        reachable_labels = _subtree_leaf_labels(node_index)
        if len(reachable_labels) == 1:
            if verbose:
                depth = int(np.floor(np.log2(node_index + 1)))
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

    # Leaf predictions for the initial routing. Refreshed once at the end of
    # each iteration below and reused as-is at the top of the next iteration
    # — nothing mutates node_weights/node_bias between those two points, so
    # recomputing again here would just repeat the same work.
    #
    # On a fresh init, keep the random leaf labels initialize_tree_weights just
    # drew instead of majority-voting them: with an imbalanced label
    # distribution (e.g. background under neg_ratio subsampling), a majority
    # vote makes every leaf agree on the dominant class, which zeroes every
    # node's |left_loss - right_loss| split signal and deadlocks TAO before
    # the first node ever fits. The random labels give the first iteration
    # something to route against; update_leaf_predictions takes over as
    # normal from the end of iteration 1 onward.
    reduced_sets = compute_reduced_sets(tree, features)
    if not freshly_initialized:
        update_leaf_predictions(tree, labels, reduced_sets, class_weights=class_weights)

    for iteration_index in iteration_range:
        iteration_start = perf_counter()

        # ---------------------------------------------------------------------
        # Solve each internal-node reduced problem in reverse breadth-first order
        # ---------------------------------------------------------------------
        for node_index in range(tree.num_internal_nodes - 1, -1, -1):
            node_indices = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))
            node_event = {
                "phase": "start",
                "iteration": iteration_index + 1,
                "iterations": iterations,
                "node_index": node_index,
                "node_number": node_index + 1,
                "node_position": tree.num_internal_nodes - node_index,
                "num_internal_nodes": tree.num_internal_nodes,
                "samples_at_node": int(node_indices.size),
                "logistic_max_iter": logistic_max_iter,
            }
            if node_progress_callback is not None:
                node_progress_callback(node_event)

            if node_indices.size == 0:
                tree.node_weights[node_index] = 0.0
                tree.node_bias[node_index] = 0.0
                if node_progress_callback is not None:
                    node_progress_callback(
                        {**node_event, "phase": "end", "status": "empty"}
                    )
                continue

            node_labels = labels[node_indices]
            left_predictions = _predict_from_node_for_indices(
                tree,
                features,
                node_indices,
                tree.left_child(node_index),
            )
            right_predictions = _predict_from_node_for_indices(
                tree,
                features,
                node_indices,
                tree.right_child(node_index),
            )

            left_loss = (left_predictions != node_labels).astype(np.float32)
            right_loss = (right_predictions != node_labels).astype(np.float32)
            sample_weights = np.abs(left_loss - right_loss)

            # Per-class weights make misrouting the weighted classes costlier
            # in the node's reduced problem (e.g. upweight minority defect
            # classes so boundaries lean away from background).
            if class_weights is not None:
                sample_weights *= class_weights[node_labels].astype(np.float32)

            positive_weight_mask = sample_weights > 0.0

            if not np.any(positive_weight_mask):
                if node_progress_callback is not None:
                    node_progress_callback(
                        {**node_event, "phase": "end", "status": "no_split_gain"}
                    )
                continue

            pseudolabels = (left_loss <= right_loss).astype(np.int64)
            solver_indices = node_indices[positive_weight_mask]

            # Kairgeldin eq. 3-4: RP_i = Σ loss + λ·h_α(|R_i|)·‖w‖₁
            # where h_α(t) = t^α.  LIBLINEAR uses ‖w‖₁ + C·Σ loss,
            # so C = 1 / (λ · |R_i|^α).
            effective_lambda = float(
                l1_lambda * float(max(node_indices.size, 1) ** sparsity_alpha)
            )

            # Cap LIBLINEAR samples to prevent memory exhaustion.
            # L1 logistic regression on 30K random samples gives nearly
            # identical solutions to the full set, and avoids allocating
            # multi-GB float64 matrices that cause swap death.
            solver_cap = 30_000

            # Pass the full features memmap and the subset indices separately.
            # This allows the solver to stream the data in chunks rather than
            # materializing a multi-gigabyte array in RAM.
            batch = _prepare_reduced_problem_batch(
                features,
                pseudolabels[positive_weight_mask],
                sample_weights[positive_weight_mask],
                indices=solver_indices,
                random_state=random_state,
                max_solver_samples=solver_cap,
            )
            batch_features, batch_labels, batch_sample_weights = batch

            weights, bias = solve_l1_logistic_reduced_problem(
                batch_features,
                batch_labels,
                batch_sample_weights,
                l1_lambda=effective_lambda,
                max_iter=logistic_max_iter,
                tolerance=tolerance,
                zero_threshold=zero_threshold,
                random_state=random_state,
            )

            # Accept the fit only if it does not worsen this node's own
            # reduced-problem objective, scored on the same materialized
            # batch — no extra data read.
            weights, bias = _accept_or_reject_node_update(
                batch_features,
                batch_labels,
                batch_sample_weights,
                tree.node_weights[node_index].copy(),
                float(tree.node_bias[node_index]),
                weights,
                bias,
                effective_lambda,
            )
            tree.node_weights[node_index] = weights
            tree.node_bias[node_index] = bias

            if node_progress_callback is not None:
                node_progress_callback(
                    {
                        **node_event,
                        "phase": "end",
                        "status": "fit",
                        "effective_lambda": effective_lambda,
                    }
                )

        # Refresh leaf predictions and reduced sets under the routing just
        # fit above — scores this iteration's metrics and carries over as
        # the starting state for the next iteration's node solving.
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

        # Convergence check (Hada Fig. 1 / Kairgeldin Fig. 5):
        # stop when the objective function decreases less than a tolerance.
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
