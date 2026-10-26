"""The pixel masking protocol on a stub model whose answer depends on one known tile."""

import numpy as np
import torch

from neurosym.evaluation import (
    PATCH_SIZE,
    box_tile_mask,
    keep_mask_to_pixels,
    memoize_by_keep_mask,
    patch_scores,
    pixel_masking_metrics,
    random_order_box_first,
)

GRID = (4, 4)
KEY_TILE = 5


def _evaluate(keep: np.ndarray) -> tuple[int, float]:
    # Label and score both hinge on the one tile that matters; every other tile is noise.
    kept = bool(keep.reshape(-1)[KEY_TILE])
    return int(kept), 1.0 if kept else 0.1


def _order(first: int) -> np.ndarray:
    return np.array([first] + [tile for tile in range(16) if tile != first])


def test_patch_scores_sum_each_tile_and_keep_the_total():
    pixel_map = np.ones((PATCH_SIZE + 4, PATCH_SIZE + 8))
    scores = patch_scores(pixel_map)

    assert scores.shape == (2, 2)
    assert scores[0, 0] == PATCH_SIZE * PATCH_SIZE
    assert scores[1, 1] == 4 * 8  # edge tile counts only real pixels
    assert scores.sum() == pixel_map.sum()


def test_keep_mask_to_pixels_crops_to_the_image():
    keep = np.array([[True, False], [False, True]])
    pixels = keep_mask_to_pixels(keep, (PATCH_SIZE + 4, PATCH_SIZE + 8))

    assert pixels.shape == (PATCH_SIZE + 4, PATCH_SIZE + 8)
    assert pixels[0, 0] and not pixels[0, -1] and not pixels[-1, 0] and pixels[-1, -1]


def test_box_tile_mask_marks_touched_tiles_and_never_comes_back_empty():
    box = torch.tensor([0.0, 0.0, 2 * PATCH_SIZE, PATCH_SIZE + 1])  # 2 columns, 2 rows
    mask = box_tile_mask(box, GRID)
    assert mask.sum() == 4 and mask[:2, :2].all()
    assert box_tile_mask(torch.tensor([5.0, 5.0, 5.0, 5.0]), GRID).sum() == 1  # zero-size box


def test_random_order_puts_the_box_tiles_first_and_covers_every_tile():
    mask = box_tile_mask(torch.tensor([PATCH_SIZE, PATCH_SIZE, 2 * PATCH_SIZE, 2 * PATCH_SIZE]), GRID)
    order = random_order_box_first(np.random.default_rng(0), mask)

    assert sorted(order.tolist()) == list(range(16))
    assert mask.reshape(-1)[order[: int(mask.sum())]].all()


def test_good_order_flips_and_preserves_while_a_bad_one_does_not():
    good = pixel_masking_metrics(_evaluate, _order(KEY_TILE), GRID, budget=1, box_tiles=4, auc_steps=4)
    bad = pixel_masking_metrics(_evaluate, _order(0), GRID, budget=1, box_tiles=4, auc_steps=4)

    assert (good["necessity"], good["sufficiency"]) == (1.0, 1.0)
    assert (bad["necessity"], bad["sufficiency"]) == (0.0, 0.0)
    assert good["deletion_auc"] < bad["deletion_auc"]
    assert good["insertion_auc"] > bad["insertion_auc"]


def test_memoize_runs_each_keep_mask_once():
    calls = []
    cached = memoize_by_keep_mask(lambda keep: calls.append(1) or keep.sum())
    mask = np.ones(GRID, dtype=bool)

    assert cached(mask) == cached(mask.copy()) == 16
    assert len(calls) == 1
