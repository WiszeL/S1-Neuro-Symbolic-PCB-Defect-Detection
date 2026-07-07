"""Smoke check: node heatmaps project onto the proposal box, not the regressed
detection box, since the RoI-Align grid was pooled from the proposal box."""

import numpy as np
import torch

from neurosym.inference import explain_hybrid_detection
from symbolic.sodt import SparseObliqueDecisionTreeClassifier


class _FakeModel:
    def __init__(self, tree: SparseObliqueDecisionTreeClassifier) -> None:
        self.symbolic_tree = tree


def _single_leaf_tree(feature_shape: tuple[int, int, int]) -> SparseObliqueDecisionTreeClassifier:
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[0, 0] = 1.0
    tree.leaf_labels[:] = [0, 1]
    return tree


def test_node_heatmap_projects_onto_proposal_box_not_detection_box():
    feature_shape = (2, 4, 4)
    tree = _single_leaf_tree(feature_shape)
    model = _FakeModel(tree)

    pooled_features = torch.rand(1, *feature_shape)
    proposal_box = torch.tensor([10.0, 10.0, 20.0, 20.0])
    detection_box = torch.tensor([50.0, 50.0, 60.0, 60.0])

    detection = {
        "pooled_features": pooled_features,
        "proposal_boxes": proposal_box.unsqueeze(0),
        "boxes": detection_box.unsqueeze(0),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
        "symbolic_leaf_indices": torch.tensor([1]),
        "symbolic_probabilities": torch.tensor([[0.1, 0.9]]),
    }

    explanation = explain_hybrid_detection(
        model, detection, detection_index=0, image_shape=(64, 64)
    )
    canvas = explanation["node_explanations"][0]["projected_node_heatmap_on_proposal_box"]

    proposal_region_hit = bool(canvas[10:20, 10:20].abs().sum() > 0)
    detection_region_hit = bool(canvas[50:60, 50:60].abs().sum() > 0)

    assert proposal_region_hit
    assert not detection_region_hit


if __name__ == "__main__":
    test_node_heatmap_projects_onto_proposal_box_not_detection_box()
    print("OK")
