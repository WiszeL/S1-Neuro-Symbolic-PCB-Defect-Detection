"""Smoke check: shared heatmap metric helpers (util/heatmap_metrics.py)."""

import torch

from util.heatmap_metrics import normalize_heatmap, pointing_score, topk_region_overlap


def test_normalize_heatmap_sums_to_one():
    heatmap = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    normalized = normalize_heatmap(heatmap)
    assert abs(normalized.sum().item() - 1.0) < 1e-6
    assert (normalized >= 0).all()


def test_normalize_heatmap_all_negative_is_zero():
    heatmap = torch.tensor([[-1.0, -2.0]])
    normalized = normalize_heatmap(heatmap)
    assert normalized.sum().item() == 0.0


def test_pointing_score_hit_and_miss():
    heatmap = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    hit_mask = torch.tensor([[False, False], [True, False]])
    miss_mask = torch.tensor([[True, False], [False, False]])
    assert pointing_score(heatmap, hit_mask) == 1.0
    assert pointing_score(heatmap, miss_mask) == 0.0


def test_topk_region_overlap_perfect_match():
    heatmap = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    gt_mask = torch.tensor([[False, False], [True, True]])
    assert topk_region_overlap(heatmap, gt_mask) == 1.0


def test_topk_region_overlap_no_gt_is_zero():
    heatmap = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    gt_mask = torch.zeros((2, 2), dtype=torch.bool)
    assert topk_region_overlap(heatmap, gt_mask) == 0.0


if __name__ == "__main__":
    test_normalize_heatmap_sums_to_one()
    test_normalize_heatmap_all_negative_is_zero()
    test_pointing_score_hit_and_miss()
    test_topk_region_overlap_perfect_match()
    test_topk_region_overlap_no_gt_is_zero()
    print("OK")
