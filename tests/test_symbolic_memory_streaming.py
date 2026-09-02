"""Smoke checks for the 256ch memory-streaming fixes:

- prediction/gather paths keep the feature matrix at its on-disk float16
  dtype and upcast per chunk, giving values identical to a whole-array cast
- evaluate_tree, handed the routing it already computed, derives predictions
  without a second pass and gets exactly what tree.predict would return
"""

import numpy as np

from symbolic.sodt import SparseObliqueDecisionTreeClassifier, _iter_float32_chunks
from symbolic.tao import (
    _select_rows,
    compute_reduced_sets,
    evaluate_tree,
    fit_tree_with_tao,
    update_leaf_predictions,
)


def _fitted_tree(seed: int = 0, n: int = 500, dim: int = 8):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = (features @ rng.normal(size=dim) > 0).astype(np.int64)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3, num_classes=2, input_dim=dim
    )
    fit_tree_with_tao(tree, features, labels, iterations=4, random_state=seed)
    return tree, features, labels


def test_float16_chunks_match_whole_array_cast():
    rng = np.random.default_rng(1)
    feats16 = rng.normal(size=(4096, 12)).astype(np.float16)
    whole = feats16.astype(np.float32)

    rebuilt = np.empty_like(whole)
    for start, stop, chunk in _iter_float32_chunks(feats16):
        assert chunk.dtype == np.float32
        rebuilt[start:stop] = chunk
    assert np.array_equal(rebuilt, whole)


def test_select_rows_returns_float32_identical_to_source():
    rng = np.random.default_rng(2)
    source16 = rng.normal(size=(5000, 10)).astype(np.float16)

    contiguous = np.arange(100, 400, dtype=np.int64)
    scattered = rng.choice(5000, size=3000, replace=False).astype(np.int64)
    small = rng.choice(5000, size=50, replace=False).astype(np.int64)

    for idx in (contiguous, scattered, small):
        rows = _select_rows(source16, idx)
        assert rows.dtype == np.float32
        assert np.array_equal(rows, source16[idx].astype(np.float32))


def test_evaluate_tree_derives_predictions_from_reduced_sets():
    tree, features, labels = _fitted_tree()

    reduced_sets = compute_reduced_sets(tree, features)
    update_leaf_predictions(tree, labels, reduced_sets)

    with_rs = evaluate_tree(tree, features, labels, reduced_sets=reduced_sets)
    without_rs = evaluate_tree(tree, features, labels)

    assert with_rs["mimic_accuracy"] == without_rs["mimic_accuracy"]
    assert with_rs["objective"] == without_rs["objective"]

    # And the derived label vector itself equals a full tree.predict pass.
    derived = np.empty((labels.shape[0],), dtype=np.int64)
    for leaf_offset in range(tree.num_leaves):
        rows = reduced_sets.get(tree.num_internal_nodes + leaf_offset)
        if rows is not None and rows.size:
            derived[rows] = tree.leaf_labels[leaf_offset]
    assert np.array_equal(derived, tree.predict(features))


if __name__ == "__main__":
    test_float16_chunks_match_whole_array_cast()
    test_select_rows_returns_float32_identical_to_source()
    test_evaluate_tree_derives_predictions_from_reduced_sets()
    print("OK")
