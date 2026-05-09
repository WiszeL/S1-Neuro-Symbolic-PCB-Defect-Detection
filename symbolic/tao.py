from __future__ import annotations

from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .sodt import SparseObliqueDecisionTreeClassifier


_TAO_CHUNK_SIZE = 512


def initialize_tree_weights(
    tree: SparseObliqueDecisionTreeClassifier,
    random_state: int = 42,
    scale: float = 1e-3,
) -> None:
    generator = np.random.default_rng(random_state)
    tree.node_weights = generator.normal(
        loc=0.0,
        scale=scale,
        size=tree.node_weights.shape,
    ).astype(np.float32)
    tree.node_bias = generator.normal(
        loc=0.0,
        scale=scale,
        size=tree.node_bias.shape,
    ).astype(np.float32)


def _is_contiguous_indices(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return True
    return bool(
        np.all(
            indices
            == np.arange(indices[0], indices[0] + indices.size, dtype=indices.dtype)
        )
    )


def _select_rows(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if _is_contiguous_indices(indices):
        return (
            features[int(indices[0]) : int(indices[-1]) + 1]
            if indices.size > 0
            else features[:0]
        )
    return features[indices]


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
) -> None:
    for leaf_offset in range(tree.num_leaves):
        node_index = tree.num_internal_nodes + leaf_offset
        node_indices = reduced_sets.get(node_index, np.zeros((0,), dtype=np.int64))
        if node_indices.size == 0:
            continue

        counts = np.bincount(labels[node_indices], minlength=tree.num_classes).astype(
            np.float32
        )
        tree.leaf_labels[leaf_offset] = int(counts.argmax())
        tree.leaf_distributions[leaf_offset] = (counts + tree.leaf_smoothing) / (
            counts.sum() + (tree.leaf_smoothing * tree.num_classes)
        )


def solve_l1_logistic_reduced_problem(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    l1_lambda: float,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    zero_threshold: float = 1e-5,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    if features.shape[0] == 0:
        return np.zeros((features.shape[1],), dtype=np.float32), 0.0

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)

    positive_weight_mask = sample_weights > 0.0
    if not np.any(positive_weight_mask):
        return np.zeros((features.shape[1],), dtype=np.float32), 0.0

    features = features[positive_weight_mask]
    labels = labels[positive_weight_mask]
    sample_weights = sample_weights[positive_weight_mask]

    unique_labels = np.unique(labels)
    if unique_labels.size == 1:
        bias = 1.0 if int(unique_labels[0]) == 1 else -1.0
        return np.zeros((features.shape[1],), dtype=np.float32), float(bias)

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the paper-faithful TAO reduced-problem solver."
        ) from exc

    # ---------------------------------------------------------------------
    # Match the paper solver family: L1 logistic regression with LIBLINEAR
    # ---------------------------------------------------------------------
    # The papers name the solver family but do not spell out the sklearn C mapping,
    # so we use the standard inverse-regularization convention C = 1 / lambda.
    effective_lambda = max(float(l1_lambda), 1e-12)
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        fit_intercept=True,
        C=1.0 / effective_lambda,
        random_state=random_state,
        tol=tolerance,
        max_iter=max_iter,
    )
    model.fit(
        features,
        labels,
        sample_weight=sample_weights,
    )

    learned_weights = model.coef_[0].astype(np.float32)
    learned_weights[np.abs(learned_weights) < zero_threshold] = 0.0
    return learned_weights, float(model.intercept_[0])


def evaluate_tree(
    tree: SparseObliqueDecisionTreeClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    predictions = tree.predict(features)
    accuracy = float((predictions == labels).mean())
    nonzero_counts = tree.nonzero_weight_counts()
    return {
        "mimic_accuracy": accuracy,
        "nonzero_weights": int(sum(nonzero_counts)),
        "mean_nonzero_per_node": float(np.mean(nonzero_counts))
        if nonzero_counts
        else 0.0,
        "active_internal_nodes": int(sum(count > 0 for count in nonzero_counts)),
    }


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
    random_state: int = 42,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    node_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if features.ndim != 2:
        raise ValueError(
            "TAO expects a 2D feature matrix of shape [num_samples, num_features]."
        )

    if np.allclose(tree.node_weights, 0.0):
        initialize_tree_weights(tree, random_state=random_state)

    history: list[dict[str, Any]] = []
    iteration_range = range(iterations)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(iteration_range, desc=progress_desc or "TAO", leave=False)
        iteration_range = progress_bar

    for iteration_index in iteration_range:
        # ---------------------------------------------------------------------
        # Refresh the leaf predictions under the current routing
        # ---------------------------------------------------------------------
        reduced_sets = compute_reduced_sets(tree, features)
        update_leaf_predictions(tree, labels, reduced_sets)

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
            positive_weight_mask = sample_weights > 0.0

            if not np.any(positive_weight_mask):
                if node_progress_callback is not None:
                    node_progress_callback(
                        {**node_event, "phase": "end", "status": "no_split_gain"}
                    )
                continue

            pseudolabels = (left_loss <= right_loss).astype(np.int64)
            solver_indices = node_indices[positive_weight_mask]
            node_features = _select_rows(features, solver_indices)
            effective_lambda = float(
                l1_lambda * (max(node_indices.size, 1) ** (sparsity_alpha - 1.0))
            )
            weights, bias = solve_l1_logistic_reduced_problem(
                node_features,
                pseudolabels[positive_weight_mask],
                sample_weights[positive_weight_mask],
                l1_lambda=effective_lambda,
                max_iter=logistic_max_iter,
                tolerance=tolerance,
                zero_threshold=zero_threshold,
                random_state=random_state,
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

        reduced_sets = compute_reduced_sets(tree, features)
        update_leaf_predictions(tree, labels, reduced_sets)

        metrics = evaluate_tree(tree, features, labels)
        metrics["iteration"] = iteration_index + 1
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(metrics)
        if progress_bar is not None:
            progress_bar.set_postfix(
                mimic=f"{metrics['mimic_accuracy']:.4f}",
                nz=metrics["nonzero_weights"],
            )

    return history
