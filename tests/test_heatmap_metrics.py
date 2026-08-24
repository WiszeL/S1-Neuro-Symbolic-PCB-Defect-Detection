"""Smoke check: shared heatmap metric helpers (util/heatmap_metrics.py)."""

import torch

from util.heatmap_metrics import (
    LOW_GT_COVERAGE_THRESHOLD,
    evaluate_random_baseline_spatial_metrics,
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)


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


def test_stratified_spatial_result_empty_input_is_nan():
    result = stratified_spatial_result([], [], [])
    assert result["evaluated_roi_count"] == 0
    import math

    assert math.isnan(result["box_grounded_roi_overlap"])
    assert math.isnan(result["pointing_score"])
    assert result["low_gt_coverage"]["evaluated_roi_count"] == 0


def test_stratified_spatial_result_splits_by_coverage_threshold():
    overlap = [1.0, 0.0, 0.5]
    pointing = [1.0, 0.0, 1.0]
    coverage = [0.1, 0.9, LOW_GT_COVERAGE_THRESHOLD]  # 0.1 < thresh, 0.9 and thresh itself are not
    result = stratified_spatial_result(overlap, pointing, coverage)
    assert result["evaluated_roi_count"] == 3
    # Only the coverage=0.1 row is strictly below LOW_GT_COVERAGE_THRESHOLD.
    assert result["low_gt_coverage"]["evaluated_roi_count"] == 1
    assert result["low_gt_coverage"]["box_grounded_roi_overlap"] == 1.0
    assert result["low_gt_coverage"]["pointing_score"] == 1.0


def _full_coverage_roi_population(n: int):
    # Proposal == GT box -> the GT mask covers the entire 7x7 grid, so
    # pointing/overlap are 1.0 regardless of the random heatmap's values.
    proposal_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * n)
    matched_gt_boxes = proposal_boxes.clone()
    has_matched_gt = torch.ones(n, dtype=torch.bool)
    gt_iou = torch.ones(n)
    return proposal_boxes, matched_gt_boxes, has_matched_gt, gt_iou


def test_random_baseline_full_coverage_always_scores_perfectly():
    proposal_boxes, matched_gt_boxes, has_matched_gt, gt_iou = (
        _full_coverage_roi_population(20)
    )
    result = evaluate_random_baseline_spatial_metrics(
        proposal_boxes, matched_gt_boxes, has_matched_gt, gt_iou, random_state=0
    )
    assert result["evaluated_roi_count"] == 20
    assert result["pointing_score"] == 1.0
    assert result["box_grounded_roi_overlap"] == 1.0
    # Full-coverage RoIs never fall in the low-coverage stratum.
    assert result["low_gt_coverage"]["evaluated_roi_count"] == 0


def test_random_baseline_respects_min_proposal_iou_filter():
    proposal_boxes, matched_gt_boxes, has_matched_gt, gt_iou = (
        _full_coverage_roi_population(10)
    )
    gt_iou[:5] = 0.1  # below a 0.5 threshold, should be excluded
    result = evaluate_random_baseline_spatial_metrics(
        proposal_boxes,
        matched_gt_boxes,
        has_matched_gt,
        gt_iou,
        min_proposal_iou=0.5,
        random_state=0,
    )
    assert result["evaluated_roi_count"] == 5


def test_random_baseline_empty_when_no_matches():
    proposal_boxes, matched_gt_boxes, _, gt_iou = _full_coverage_roi_population(5)
    has_matched_gt = torch.zeros(5, dtype=torch.bool)
    result = evaluate_random_baseline_spatial_metrics(
        proposal_boxes, matched_gt_boxes, has_matched_gt, gt_iou
    )
    assert result["evaluated_roi_count"] == 0


if __name__ == "__main__":
    test_normalize_heatmap_sums_to_one()
    test_normalize_heatmap_all_negative_is_zero()
    test_pointing_score_hit_and_miss()
    test_topk_region_overlap_perfect_match()
    test_topk_region_overlap_no_gt_is_zero()
    test_stratified_spatial_result_empty_input_is_nan()
    test_stratified_spatial_result_splits_by_coverage_threshold()
    test_random_baseline_full_coverage_always_scores_perfectly()
    test_random_baseline_respects_min_proposal_iou_filter()
    test_random_baseline_empty_when_no_matches()
    print("OK")
