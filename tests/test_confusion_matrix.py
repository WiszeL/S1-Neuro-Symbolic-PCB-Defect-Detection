"""Wrong-class matches land off-diagonal; precision/recall counts don't move."""

import torch

from neuro.train import _confusion_and_per_class


def test_cross_class_confusion_and_unchanged_precision_recall():
    # Setup: wrong-class claim, one clean match, one missed box.
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

    # Off-diagonal: wrong-class claim.
    assert result["confusion"][1, 2].item() == 1
    # On-diagonal: clean match.
    assert result["confusion"][1, 1].item() == 1
    # Miss counted against background.
    assert result["confusion"][2, 0].item() == 1

    # Old counts must not shift with this fix.
    assert result["class_tp"].tolist() == [0, 1, 0]
    assert result["class_fp"].tolist() == [0, 0, 1]
    assert result["class_fn"].tolist() == [0, 1, 1]


if __name__ == "__main__":
    test_cross_class_confusion_and_unchanged_precision_recall()
    print("OK")
