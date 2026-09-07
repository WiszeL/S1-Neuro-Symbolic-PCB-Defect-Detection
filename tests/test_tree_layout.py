"""Tree layout stays bounded, even with a live off-path grandchild."""

import numpy as np
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.inference import explain_hybrid_detection
from neurosym.visualization import _pruned_tree_layout
from symbolic.sodt import SparseObliqueDecisionTreeClassifier

FEATURE_SHAPE = (3, 5, 5)
DEPTH = 4
PROCESSED_SIZE = (48, 48)
PADDED_SIZE = (48, 48)


def _forced_path_tree(rng: np.random.Generator) -> SparseObliqueDecisionTreeClassifier:
    """Weights chosen so a large bias — not the (small, random) feature dot
    product — decides direction, so `decision_path` on ANY input provably
    follows the alternating path this builds (see WHAT-I-DID: a routing
    tree's decision depends on `w.x + b`, so a big enough `b` pins it)."""
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=DEPTH, num_classes=2, input_dim=int(np.prod(FEATURE_SHAPE)),
        feature_shape=FEATURE_SHAPE, class_names=("background", "spur"),
    )
    node = 0
    for depth in range(DEPTH):
        tree.node_weights[node] = rng.normal(size=tree.input_dim).astype(np.float32) * 0.01
        go_left = depth % 2 == 0
        tree.node_bias[node] = 100.0 if go_left else -100.0
        # An off-path sibling with a live (non-pruned) grandchild — the real
        # crowding case: not fully zero, so a naive walker recurses into it.
        off_child = tree.right_child(node) if go_left else tree.left_child(node)
        if off_child < tree.num_internal_nodes:
            grandchild = tree.left_child(off_child)
            if grandchild < tree.num_internal_nodes:
                tree.node_weights[grandchild] = rng.normal(size=tree.input_dim).astype(
                    np.float32
                ) * 0.01
                tree.node_bias[grandchild] = 5.0
        node = tree.left_child(node) if go_left else tree.right_child(node)
    tree.leaf_labels[:] = 0
    tree.leaf_labels[node - tree.num_internal_nodes] = 1  # "spur"
    return tree


class _FakeModel:
    def __init__(self, tree, roi_align):
        self.symbolic_tree = tree

        class _Detector:
            pass

        self.detector = _Detector()
        self.detector.roi_align = roi_align


def _build_explanation():
    roi_align = MultiScaleRoIAlign(
        featmap_names=["p2"], output_size=FEATURE_SHAPE[1:], sampling_ratio=2
    )
    fpn_map = torch.rand(FEATURE_SHAPE[0], 48, 48)
    proposal_box = torch.tensor([8.0, 8.0, 32.0, 32.0])
    pooled = roi_align(
        {"p2": fpn_map.unsqueeze(0)}, [proposal_box.unsqueeze(0)], [PROCESSED_SIZE]
    )
    tree = _forced_path_tree(np.random.default_rng(0))
    model = _FakeModel(tree, roi_align)

    # 1-based labels (0 is background), so class_names needs an entry before
    # "spur" — matches how the detector's class_names are always indexed.
    detection = {
        "pooled_features": pooled,
        "proposal_boxes": proposal_box.unsqueeze(0),
        "proposal_boxes_processed": proposal_box.unsqueeze(0),
        "boxes": proposal_box.unsqueeze(0),
        "labels": torch.tensor([2]),
        "scores": torch.tensor([0.9]),
        "symbolic_leaf_indices": torch.tensor(
            [int(tree.leaf_index_for_feature(pooled[0].numpy().reshape(-1)))]
        ),
        "symbolic_probabilities": torch.tensor([[0.1, 0.9]]),
        "symbolic_level_indices": torch.tensor([0]),
        "featmap_names": ["p2"],
        "fpn_features": {"p2": fpn_map},
        "padded_image_size": PADDED_SIZE,
        "processed_image_size": PROCESSED_SIZE,
    }
    explanation = explain_hybrid_detection(
        model, detection, detection_index=0, image_shape=(48, 48)
    )
    return explanation, tree


def test_pruned_tree_layout_stays_bounded_even_with_a_live_off_path_grandchild():
    explanation, tree = _build_explanation()
    layout = _pruned_tree_layout(explanation, tree, class_names=("open", "spur"))
    # Path nodes + one off-path sibling per level + the final leaf — never
    # the whole tree (up to 2**DEPTH leaves), regardless of what's pruned.
    assert len(layout.visible_nodes) <= 2 * DEPTH + 2


def test_pruned_tree_layout_coords_cover_every_visible_node():
    # visible_nodes and coords come from two separate walks (drawing structure
    # vs. drawing position) — if they ever disagree, rendering hits a
    # KeyError on a missing coord.
    explanation, tree = _build_explanation()
    layout = _pruned_tree_layout(explanation, tree, class_names=("open", "spur"))
    assert set(layout.coords) == layout.visible_nodes


def test_pruned_tree_layout_sibling_gap_never_shrinks_with_depth():
    # Regression: the old "x = average of children" layout squeezed this
    # tree's shape (one path node + one dangling leaf per level) tighter at
    # every level. Gap must stay constant instead.
    explanation, tree = _build_explanation()
    layout = _pruned_tree_layout(explanation, tree, class_names=("open", "spur"))
    xs_per_y: dict[float, list[float]] = {}
    for x, y in layout.coords.values():
        xs_per_y.setdefault(round(y, 6), []).append(x)
    gaps = [max(xs) - min(xs) for xs in xs_per_y.values() if len(xs) > 1]
    assert len(set(gaps)) == 1


if __name__ == "__main__":
    test_pruned_tree_layout_stays_bounded_even_with_a_live_off_path_grandchild()
    test_pruned_tree_layout_coords_cover_every_visible_node()
    test_pruned_tree_layout_sibling_gap_never_shrinks_with_depth()
    print("OK")
