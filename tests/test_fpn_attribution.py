"""Contribution maps: H_i = Σ_c |Gradient × Input| on the FPN level the RoI read (draft Eq 3.5 to 3.8)."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from neuro.faster_rcnn import NeuroFasterRCNN
from neurosym.heatmap import fpn_node_maps, fpn_node_maps_many
from neurosym.hybrid import NeuroSymbolicDetector
from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from util.config import load_yaml

IMAGE_SIZE = (80, 96)
MIN_SIZE = 96  # short side 80 -> 96, so the transform really resizes
BOXES = torch.tensor([[8.0, 8.0, 60.0, 60.0], [20.0, 10.0, 90.0, 70.0], [0.0, 0.0, 40.0, 50.0]])


@pytest.fixture(scope="module")
def model() -> NeuroSymbolicDetector:
    torch.manual_seed(0)
    neuro_config = load_yaml(Path("neuro.yaml"), dict)
    train_config = deepcopy(load_yaml(Path("neuro_train.yaml"), dict))
    neuro_config["net"]["backbone_pretrained"] = False
    train_config["preprocessing"]["test_min_size"] = MIN_SIZE
    train_config["preprocessing"]["max_size"] = 2 * MIN_SIZE
    detector = NeuroFasterRCNN(neuro_config=neuro_config, train_config=train_config)

    rng = np.random.default_rng(0)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=3,
        num_classes=detector.num_classes,
        input_dim=64 * 7 * 7,
        feature_shape=(64, 7, 7),
    )
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape)
    hybrid = NeuroSymbolicDetector(detector, tree, device="cpu")
    hybrid.detector.eval()
    return hybrid


def test_maps_are_non_negative_and_overview_is_their_sum(model):
    torch.manual_seed(1)
    image = torch.rand(3, *IMAGE_SIZE)
    result = fpn_node_maps(model, image, BOXES[0])

    assert result["node_maps"].shape == (len(result["path"]), *IMAGE_SIZE)
    assert (result["node_maps"] >= 0).all()
    np.testing.assert_allclose(
        result["overview_map"], result["node_maps"].sum(axis=0), rtol=1e-4, atol=1e-6
    )


def test_gradient_times_input_is_complete_over_the_pyramid(model):
    """Σ Gradient × Input over every FPN cell equals the node values' linear part (RoI Align is linear)."""
    torch.manual_seed(2)
    image = torch.rand(3, *IMAGE_SIZE)
    detector = model.detector
    nodes = fpn_node_maps(model, image, BOXES[1])["node_indices"]
    weights = torch.as_tensor(model.symbolic_tree.node_weights[nodes]).reshape(len(nodes), -1)

    with torch.no_grad():
        images_list, _ = detector.transform([image], None)
        features = detector.backbone(images_list.tensors)
    leaves = {name: level.clone().requires_grad_(True) for name, level in features.items()}
    pooled = detector.roi_align(leaves, [BOXES[1:2]], images_list.image_sizes)
    linear_part = (weights @ pooled.reshape(-1)).sum()
    gradients = torch.autograd.grad(linear_part, list(leaves.values()), allow_unused=True)
    total = sum(float((g * f).sum()) for g, f in zip(gradients, features.values()) if g is not None)
    np.testing.assert_allclose(total, float(linear_part), rtol=1e-4)


def test_cropped_maps_equal_full_level_maps(model):
    """Working on a crop around the box must give the same H_i as the gradient on the whole level."""
    torch.manual_seed(4)
    image = torch.rand(3, *IMAGE_SIZE)
    detector = model.detector
    box = BOXES[1]
    result = fpn_node_maps(model, image, box)
    weights = torch.as_tensor(model.symbolic_tree.node_weights[result["node_indices"]])

    with torch.no_grad():
        images_list, _ = detector.transform([image], None)
        features = detector.backbone(images_list.tensors)
    leaves = {name: level.clone().requires_grad_(True) for name, level in features.items()}
    pooled = detector.roi_align(leaves, [box[None]], images_list.image_sizes)
    for index, node_weights in enumerate(weights):
        gradients = torch.autograd.grad(
            (node_weights.reshape(-1) * pooled.reshape(-1)).sum(), list(leaves.values()), retain_graph=True, allow_unused=True
        )
        total = sum(float((g * f).abs().sum(1).sum()) for g, f in zip(gradients, features.values()) if g is not None)
        # Placing spreads a cell over its pixels, so the mass is conserved up to resize rounding.
        np.testing.assert_allclose(result["node_maps"][index].sum(), total, rtol=0.1)


def test_many_boxes_equal_one_box_at_a_time(model):
    torch.manual_seed(3)
    image = torch.rand(3, *IMAGE_SIZE)
    many = fpn_node_maps_many(model, image, BOXES)

    for box, result in zip(BOXES, many):
        single = fpn_node_maps(model, image, box)
        assert result["node_indices"] == single["node_indices"]
        np.testing.assert_allclose(result["overview_map"], single["overview_map"], rtol=1e-5, atol=1e-6)
