"""The lean per-node maps must equal the exact maps the display path builds."""

import numpy as np
import pytest
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.heatmap import compute_exact_attribution, node_fpn_maps
from neurosym.inference import compute_node_fpn_maps, explain_hybrid_detection
from symbolic.sodt import SparseObliqueDecisionTreeClassifier

PROCESSED_SIZE = (32, 32)
PADDED_SIZE = (32, 32)
ATOL = 1e-6
RTOL = 1e-5


def _random_tree(
    feature_shape: tuple[int, int, int], seed: int, max_depth: int = 3
) -> SparseObliqueDecisionTreeClassifier:
    rng = np.random.default_rng(seed)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=max_depth,
        num_classes=2,
        input_dim=int(np.prod(feature_shape)),
        feature_shape=feature_shape,
    )
    tree.node_weights[:] = rng.normal(size=tree.node_weights.shape).astype(np.float32)
    tree.node_bias[:] = rng.normal(size=tree.node_bias.shape).astype(np.float32)
    return tree


def _single_level_pool(feature_shape: tuple[int, int, int]) -> MultiScaleRoIAlign:
    return MultiScaleRoIAlign(
        featmap_names=["p2"], output_size=feature_shape[1:], sampling_ratio=2
    )


def _grid(pool, fpn, box, image_size=PROCESSED_SIZE) -> np.ndarray:
    return pool(
        {k: v.unsqueeze(0) for k, v in fpn.items()}, [box.unsqueeze(0)], [image_size]
    )[0].numpy().astype(np.float32)


def _old_maps(tree, grid, pool, fpn, level, box, processed_size, padded_size):
    path = tree.decision_path(grid.reshape(-1))
    node_maps = [
        compute_exact_attribution(
            tree, grid, pool, fpn, level, box, processed_size, padded_size, path=[step]
        )
        for step in path
    ]
    path_map = compute_exact_attribution(
        tree, grid, pool, fpn, level, box, processed_size, padded_size, path=path
    )
    return node_maps, path_map


def _assert_same(new, old):
    new_nodes, new_path = new
    old_nodes, old_path = old
    assert len(new_nodes) == len(old_nodes)
    for got, want in zip(new_nodes, old_nodes):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)
    assert new_path.shape == old_path.shape
    np.testing.assert_allclose(new_path, old_path, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_node_maps_and_path_map_equal_the_old_exact_attribution(seed):
    feature_shape = (3, 4, 4)
    tree = _random_tree(feature_shape, seed)
    pool = _single_level_pool(feature_shape)
    torch.manual_seed(seed)
    fpn = {"p2": torch.rand(3, 16, 16)}
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])
    grid = _grid(pool, fpn, box)

    new = node_fpn_maps(
        tree, grid, pool, fpn["p2"], box, PROCESSED_SIZE, PADDED_SIZE
    )
    old = _old_maps(tree, grid, pool, fpn, "p2", box, PROCESSED_SIZE, PADDED_SIZE)
    _assert_same(new, old)
    assert len(new[0]) >= 2  # a real multi-step path, not a single node


def test_each_level_matches_when_boxes_land_on_different_levels():
    feature_shape = (3, 7, 7)
    tree = _random_tree(feature_shape, seed=3)
    pool = MultiScaleRoIAlign(
        featmap_names=["p2", "p3"], output_size=7, sampling_ratio=2
    )
    size = (512, 512)
    torch.manual_seed(3)
    fpn = {"p2": torch.rand(3, 128, 128), "p3": torch.rand(3, 64, 64)}
    names = ["p2", "p3"]

    levels_seen = set()
    for box in (
        torch.tensor([100.0, 120.0, 140.0, 160.0]),  # ~40 px -> p2
        torch.tensor([10.0, 10.0, 490.0, 490.0]),  # ~480 px -> p3
    ):
        grid = _grid(pool, fpn, box, size)  # first call also sets pool.map_levels
        level = names[int(pool.map_levels([box.unsqueeze(0)])[0])]
        levels_seen.add(level)

        new = node_fpn_maps(tree, grid, pool, fpn[level], box, size, size)
        old = _old_maps(tree, grid, pool, fpn, level, box, size, size)
        _assert_same(new, old)
    assert levels_seen == {"p2", "p3"}


def test_empty_crop_returns_single_zero_cell_like_the_old_path():
    feature_shape = (2, 4, 4)
    tree = _random_tree(feature_shape, seed=4)
    pool = _single_level_pool(feature_shape)
    fpn = {"p2": torch.rand(2, 16, 16)}
    box = torch.tensor([30.0, 30.0, 30.0, 30.0])  # zero-area box: crop collapses
    grid = _grid(pool, fpn, box)

    new = node_fpn_maps(tree, grid, pool, fpn["p2"], box, PROCESSED_SIZE, PADDED_SIZE)
    old = _old_maps(tree, grid, pool, fpn, "p2", box, PROCESSED_SIZE, PADDED_SIZE)
    _assert_same(new, old)
    assert new[1].shape == (1, 1) and float(new[1].sum()) == 0.0


class _RoIWrapper:
    """Callable like the detector's RoIAlign, exposing `.pool` like it too."""

    def __init__(self, pool) -> None:
        self.pool = pool

    def __call__(self, *args, **kwargs):
        return self.pool(*args, **kwargs)


class _FakeModel:
    def __init__(self, tree, pool) -> None:
        self.symbolic_tree = tree

        class _Detector:
            pass

        self.detector = _Detector()
        self.detector.roi_align = _RoIWrapper(pool)


def test_wrapper_equals_the_maps_explain_hybrid_detection_builds():
    feature_shape = (3, 4, 4)
    tree = _random_tree(feature_shape, seed=5)
    pool = _single_level_pool(feature_shape)
    model = _FakeModel(tree, pool)
    torch.manual_seed(5)
    fpn = {"p2": torch.rand(3, 16, 16)}
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])
    grid = pool(
        {"p2": fpn["p2"].unsqueeze(0)}, [box.unsqueeze(0)], [PROCESSED_SIZE]
    )

    detection = {
        "pooled_features": grid,
        "proposal_boxes": box.unsqueeze(0),
        "proposal_boxes_processed": box.unsqueeze(0),
        "boxes": box.unsqueeze(0),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
        "symbolic_leaf_indices": torch.tensor([1]),
        "symbolic_probabilities": torch.tensor([[0.1, 0.9]]),
        "symbolic_level_indices": torch.tensor([0]),
        "featmap_names": ["p2"],
        "fpn_features": fpn,
        "padded_image_size": PADDED_SIZE,
        "processed_image_size": PROCESSED_SIZE,
    }

    lean = compute_node_fpn_maps(model, detection, [0])
    assert len(lean) == 1 and lean[0]["detection_index"] == 0
    full = explain_hybrid_detection(
        model, detection, detection_index=0, image_shape=(64, 64)
    )

    old_nodes = [n["exact_attribution"] for n in full["node_explanations"]]
    _assert_same(
        (lean[0]["node_maps"], lean[0]["path_map"]),
        (old_nodes, full["path_exact_attribution"]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_maps_computed_on_gpu_match_the_cpu_maps():
    feature_shape = (3, 4, 4)
    tree = _random_tree(feature_shape, seed=6)
    pool = _single_level_pool(feature_shape)
    torch.manual_seed(6)
    fpn = {"p2": torch.rand(3, 16, 16)}
    box = torch.tensor([4.0, 4.0, 20.0, 20.0])
    grid = _grid(pool, fpn, box)

    new = node_fpn_maps(
        tree, grid, pool, fpn["p2"].cuda(), box, PROCESSED_SIZE, PADDED_SIZE
    )
    old = _old_maps(tree, grid, pool, fpn, "p2", box, PROCESSED_SIZE, PADDED_SIZE)
    new_nodes, new_path = new
    for got, want in zip(new_nodes, old[0]):
        np.testing.assert_allclose(got, want, atol=1e-5)  # atomicAdd order on GPU
    np.testing.assert_allclose(new_path, old[1], atol=1e-5)


if __name__ == "__main__":
    for seed in (0, 1, 2):
        test_node_maps_and_path_map_equal_the_old_exact_attribution(seed)
    test_each_level_matches_when_boxes_land_on_different_levels()
    test_empty_crop_returns_single_zero_cell_like_the_old_path()
    test_wrapper_equals_the_maps_explain_hybrid_detection_builds()
    if torch.cuda.is_available():
        test_maps_computed_on_gpu_match_the_cpu_maps()
    print("OK")
