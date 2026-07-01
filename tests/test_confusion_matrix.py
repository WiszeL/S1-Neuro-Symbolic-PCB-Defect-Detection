"""Smoke check: _global_confusion_matrix shows real cross-class confusion,
while per-label TP/FP/FN (precision/recall) stays unchanged."""

import torch

from neuro.train import _confusion_and_per_class


def test_cross_class_confusion_and_unchanged_precision_recall():
    # Higher-score pred (class 2) wrongly claims a class-1 GT box; a correct
    # class-1 match exists elsewhere; one class-2 GT is missed entirely.
    pred = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([2, 1]),
    }
    gt = {
        "boxes": torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [50.0, 50.0, 60.0, 60.0]]
        ),
        "labels": torch.tensor([1, 1, 2]),
    }

    result = _confusion_and_per_class(
        pred, gt, iou_threshold=0.5, score_threshold=0.3, num_classes=2
    )

    # Off-diagonal: GT class 1 matched by a class-2 prediction.
    assert result["confusion"][1, 2].item() == 1
    # On-diagonal: correct class-1 match.
    assert result["confusion"][1, 1].item() == 1
    # Missed class-2 GT box -> false negative against background.
    assert result["confusion"][2, 0].item() == 1

    # Per-label TP/FP/FN must match the original per-label-isolated matching
    # (drives precision/recall/f1 and must not shift with this fix).
    assert result["class_tp"].tolist() == [0, 1, 0]
    assert result["class_fp"].tolist() == [0, 0, 1]
    assert result["class_fn"].tolist() == [0, 1, 1]


if __name__ == "__main__":
    test_cross_class_confusion_and_unchanged_precision_recall()
    print("OK")
