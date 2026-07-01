from __future__ import annotations

import torch
from torch import Tensor


def project_gt_box_to_roi_grid(
    proposal_box: Tensor,
    matched_gt_box: Tensor,
    grid_shape: tuple[int, int],
) -> Tensor:
    proposal = torch.as_tensor(proposal_box, dtype=torch.float32)
    gt_box = torch.as_tensor(matched_gt_box, dtype=torch.float32)
    grid_height, grid_width = grid_shape[0], grid_shape[1]

    proposal_width = max(float(proposal[2] - proposal[0]), 1e-6)
    proposal_height = max(float(proposal[3] - proposal[1]), 1e-6)

    px1 = torch.clamp(
        ((gt_box[0] - proposal[0]) / proposal_width) * grid_width,
        min=0.0,
        max=grid_width,
    )
    py1 = torch.clamp(
        ((gt_box[1] - proposal[1]) / proposal_height) * grid_height,
        min=0.0,
        max=grid_height,
    )
    px2 = torch.clamp(
        ((gt_box[2] - proposal[0]) / proposal_width) * grid_width,
        min=0.0,
        max=grid_width,
    )
    py2 = torch.clamp(
        ((gt_box[3] - proposal[1]) / proposal_height) * grid_height,
        min=0.0,
        max=grid_height,
    )

    if px2 <= px1 or py2 <= py1:
        return torch.zeros((grid_height, grid_width), dtype=torch.bool)

    cols = torch.arange(grid_width, dtype=torch.float32)
    rows = torch.arange(grid_height, dtype=torch.float32)
    cell_x1 = cols.unsqueeze(0)
    cell_y1 = rows.unsqueeze(1)
    cell_x2 = cols.unsqueeze(0) + 1
    cell_y2 = rows.unsqueeze(1) + 1

    iw = torch.clamp(torch.minimum(cell_x2, px2) - torch.maximum(cell_x1, px1), min=0)
    ih = torch.clamp(torch.minimum(cell_y2, py2) - torch.maximum(cell_y1, py1), min=0)

    return (iw > 0) & (ih > 0)
