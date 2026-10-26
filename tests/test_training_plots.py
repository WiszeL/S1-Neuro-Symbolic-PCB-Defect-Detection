"""The dashboard draws from a real TAO history."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.tao import NODE_STATUSES, fit_tree_with_tao
from symbolic.visualization import (
    _NODE_STATUS,
    plot_training_dashboard,
)


def test_plots_render_from_a_real_history():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 8)).astype(np.float32)
    labels = (features @ rng.normal(size=8) > 0).astype(np.int64)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=2, input_dim=8, feature_shape=(2, 2, 2)
    )
    history = fit_tree_with_tao(tree, features, labels, iterations=3, l1_lambda=1e-3, random_state=0)
    history.append(  # What prune_symbolic_tree appends: no node log.
        {"iteration": 4, "nonzero_weights": 1, "active_internal_nodes": 1, "postprocess_pruned_nodes": 0}
    )

    assert set(_NODE_STATUS) == set(NODE_STATUSES)
    plot_training_dashboard(history)
    plt.close("all")


if __name__ == "__main__":
    test_plots_render_from_a_real_history()
    print("OK")
