"""Smoke check: project_gt_box_to_roi_grid projects a known box onto a known grid."""

import torch

from util.geometry import project_gt_box_to_roi_grid


def test_full_overlap_marks_whole_grid():
    proposal = torch.tensor([0.0, 0.0, 7.0, 7.0])
    gt = torch.tensor([0.0, 0.0, 7.0, 7.0])
    mask = project_gt_box_to_roi_grid(proposal, gt, grid_shape=(7, 7))
    assert mask.shape == (7, 7)
    assert mask.all()


def test_quarter_overlap_marks_quarter_grid():
    proposal = torch.tensor([0.0, 0.0, 8.0, 8.0])
    gt = torch.tensor([0.0, 0.0, 4.0, 4.0])
    mask = project_gt_box_to_roi_grid(proposal, gt, grid_shape=(8, 8))
    assert mask.sum().item() == 16
    assert mask[:4, :4].all()
    assert not mask[4:, :].any()
    assert not mask[:, 4:].any()


def test_no_overlap_returns_zero_mask():
    proposal = torch.tensor([0.0, 0.0, 7.0, 7.0])
    gt = torch.tensor([100.0, 100.0, 110.0, 110.0])
    mask = project_gt_box_to_roi_grid(proposal, gt, grid_shape=(7, 7))
    assert not mask.any()


if __name__ == "__main__":
    test_full_overlap_marks_whole_grid()
    test_quarter_overlap_marks_quarter_grid()
    test_no_overlap_returns_zero_mask()
    print("OK")
