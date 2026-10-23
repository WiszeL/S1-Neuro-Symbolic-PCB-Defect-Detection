"""RPN rescue of a GT's best anchor must respect the paper's IoU > 0.3 floor."""

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


def test_weak_best_anchor_is_not_rescued():
    matches = MinIoUMatcher(0.7, 0.3, 0.3)(IOU)
    assert matches.tolist() == [-1, -1, 1, -1]


def test_torchvision_matcher_would_rescue_it():
    matches = det_utils.Matcher(0.7, 0.3, allow_low_quality_matches=True)(IOU)
    assert matches.tolist() == [0, -1, 1, -1]
