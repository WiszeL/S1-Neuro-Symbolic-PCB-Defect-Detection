"""Hand-computed VOC AP: the scorer must match mmdet's eval_map, not COCO."""

import pytest
import torch

from neuro.train import VOCMeanAP


def _example(det_boxes, det_scores, gt_boxes):
    prediction = {
        "boxes": torch.tensor(det_boxes, dtype=torch.float32),
        "scores": torch.tensor(det_scores, dtype=torch.float32),
        "labels": torch.ones(len(det_scores), dtype=torch.int64),
    }
    target = {
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
    }
    return [prediction], [target]


def _score(det_boxes, det_scores, gt_boxes):
    metric = VOCMeanAP(1, [0.5])
    metric.update(*_example(det_boxes, det_scores, gt_boxes))
    return float(metric.compute()[0, 0])


GT = [[0, 0, 10, 10], [20, 20, 30, 30]]
# TP 0.9, FP 0.8, TP 0.7 -> recalls [.5, .5, 1], precisions [1, .5, 2/3]
DETS = [[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]]
SCORES = [0.9, 0.8, 0.7]


def test_area_ap():
    assert _score(DETS, SCORES, GT) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_duplicate_detection_is_false_positive():
    # Second box also overlaps GT 0 but GT 0 is taken; no fallback to the next GT.
    ap = _score([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [[0, 0, 10, 10]])
    assert ap == pytest.approx(1.0)  # recall hits 1 at precision 1 before the FP


def test_legacy_plus_one_moves_iou_across_threshold():
    gt, det = [[0, 0, 3, 3]], [[0, 0, 3, 1.2]]  # IoU 0.4 plain, 0.55 with +1
    assert _score(det, [0.9], gt) == pytest.approx(1.0)


def test_class_without_gt_is_nan():
    metric = VOCMeanAP(2, [0.5])
    metric.update(*_example(DETS, SCORES, GT))
    ap = metric.compute()
    assert ap.shape == (1, 2) and ap[0, 1] != ap[0, 1]
