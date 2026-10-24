"""aligned=False must stay torchvision-identical; aligned=True must shift sampling."""

import torch
from torchvision.ops import MultiScaleRoIAlign

from neuro.faster_rcnn import AlignedMultiScaleRoIAlign

NAMES = ["p2", "p3", "p4", "p5", "p6"]


def _inputs():
    torch.manual_seed(0)
    features = {
        name: torch.rand(2, 8, 160 // 2**level, 160 // 2**level)
        for level, name in enumerate(NAMES)
    }
    boxes = [
        torch.tensor([[10.0, 12.0, 52.0, 60.0], [30.0, 30.0, 150.0, 140.0]]),
        torch.tensor([[5.0, 5.0, 40.0, 38.0]]),
    ]
    return features, boxes, [(640, 640), (640, 640)]


def test_unaligned_matches_torchvision():
    features, boxes, shapes = _inputs()
    reference = MultiScaleRoIAlign(NAMES, 7, 2)(features, boxes, shapes)
    ours = AlignedMultiScaleRoIAlign(NAMES, 7, 2, aligned=False)(features, boxes, shapes)
    assert torch.equal(reference, ours)


def test_aligned_changes_pooled_values():
    features, boxes, shapes = _inputs()
    unaligned = AlignedMultiScaleRoIAlign(NAMES, 7, 2, aligned=False)(features, boxes, shapes)
    aligned = AlignedMultiScaleRoIAlign(NAMES, 7, 2, aligned=True)(features, boxes, shapes)
    assert aligned.shape == unaligned.shape
    assert not torch.allclose(aligned, unaligned)
