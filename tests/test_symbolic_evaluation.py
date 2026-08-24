"""Smoke check: _deletion_insertion_auc's insertion curve starts from the
tree's actual all-zero-input prediction, not a hardcoded 0.0."""

import numpy as np

from symbolic.evaluation import _deletion_insertion_auc
from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def test_insertion_step0_uses_real_zero_input_confidence():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=4, feature_shape=(1, 2, 2)
    )
    tree.node_weights[0] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    tree.leaf_distributions[0] = np.array([0.1, 0.9], dtype=np.float32)
    tree.leaf_distributions[1] = np.array([0.9, 0.1], dtype=np.float32)
    tree.leaf_labels[:] = [1, 0]

    rng = np.random.default_rng(0)
    features = np.abs(rng.random((5, 4))).astype(np.float32) + 0.5
    preds = tree.predict(features)

    deletion_auc, insertion_auc = _deletion_insertion_auc(
        tree, features, preds, num_samples=5, steps=4
    )

    zero_probs = tree.predict_proba(np.zeros((5, 4), dtype=np.float32))
    expected = float(zero_probs[np.arange(5), preds].mean())

    # A flat curve at the true zero-input confidence should score close to
    # that confidence, not be dragged toward 0 by a fake floor.
    assert insertion_auc > 0.85
    assert abs(insertion_auc - expected) < 1e-4
    assert abs(deletion_auc - expected) < 1e-4


def test_degenerate_all_zero_tree_stays_flat():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=4, feature_shape=(1, 2, 2)
    )
    features = np.random.default_rng(1).random((5, 4)).astype(np.float32)
    preds = tree.predict(features)

    deletion_auc, insertion_auc = _deletion_insertion_auc(
        tree, features, preds, num_samples=5, steps=4
    )
    assert abs(deletion_auc - 0.5) < 1e-6
    assert abs(insertion_auc - 0.5) < 1e-6


if __name__ == "__main__":
    test_insertion_step0_uses_real_zero_input_confidence()
    test_degenerate_all_zero_tree_stays_flat()
    print("OK")
