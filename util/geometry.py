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
    grid_height, grid_width = int(grid_shape[0]), int(grid_shape[1])

    proposal_width = max(float(proposal[2] - proposal[0]), 1e-6)
    proposal_height = max(float(proposal[3] - proposal[1]), 1e-6)

    projected_x1 = ((gt_box[0] - proposal[0]) / proposal_width) * grid_width
    projected_y1 = ((gt_box[1] - proposal[1]) / proposal_height) * grid_height
    projected_x2 = ((gt_box[2] - proposal[0]) / proposal_width) * grid_width
    projected_y2 = ((gt_box[3] - proposal[1]) / proposal_height) * grid_height

    projected_x1 = float(torch.clamp(projected_x1, min=0.0, max=float(grid_width)))
    projected_y1 = float(torch.clamp(projected_y1, min=0.0, max=float(grid_height)))
    projected_x2 = float(torch.clamp(projected_x2, min=0.0, max=float(grid_width)))
    projected_y2 = float(torch.clamp(projected_y2, min=0.0, max=float(grid_height)))

    mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
    if projected_x2 <= projected_x1 or projected_y2 <= projected_y1:
        return mask

    for row_index in range(grid_height):
        for col_index in range(grid_width):
            cell_x1 = float(col_index)
            cell_y1 = float(row_index)
            cell_x2 = float(col_index + 1)
            cell_y2 = float(row_index + 1)
            intersection_width = min(cell_x2, projected_x2) - max(cell_x1, projected_x1)
            intersection_height = min(cell_y2, projected_y2) - max(cell_y1, projected_y1)
            if intersection_width > 0.0 and intersection_height > 0.0:
                mask[row_index, col_index] = True

    return mask