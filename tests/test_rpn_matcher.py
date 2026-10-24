"""RPN rescue of a GT's best anchor must respect the IoU floor (mmdet min_pos_iou)."""

import torch
from torchvision.models.detection import _utils as det_utils

from neuro.faster_rcnn import MinIoUMatcher

# rows = GT, cols = anchors. GT0's best anchor is 0.2 (too weak); GT1's is 0.35.
IOU = torch.tensor(
    [
        [0.2, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.35, 0.1],
    ]
)


def test_weak_best_anchor_is_not_rescued_above_the_floor():
    matches = MinIoUMatcher(0.7, 0.3, 0.3)(IOU)
    assert matches.tolist() == [-1, -1, 1, -1]


def test_floor_zero_is_torchvision_exactly():
    torch.manual_seed(0)
    iou = torch.rand(5, 200) * 0.8
    iou[0] *= 0.25  # GT0's best anchor stays below 0.2, so a 0.3 floor must matter
    reference = det_utils.Matcher(0.7, 0.3, allow_low_quality_matches=True)(iou)
    assert torch.equal(MinIoUMatcher(0.7, 0.3, 0.0)(iou), reference)
    assert reference.tolist() != MinIoUMatcher(0.7, 0.3, 0.3)(iou).tolist()
