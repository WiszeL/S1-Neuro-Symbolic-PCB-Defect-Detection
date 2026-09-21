"""Fast per-node scores must match a real re-pool."""

import numpy as np
import torch
from torchvision.ops import MultiScaleRoIAlign

from neurosym.evaluation import _masked_level, _node_score_after_masking, fpn_deletion_insertion_auc
from neurosym.heatmap import _fpn_box_bounds, exact_fpn_contribution
from symbolic.sodt import SparseObliqueDecisionTreeClassifier

CHANNELS, SIZE, OUTPUT = 3, 16, 4


def _fixture(box):
    torch.manual_seed(0)
    level = torch.rand(CHANNELS, SIZE, SIZE) + 0.5
    roi_align = MultiScaleRoIAlign(featmap_names=["p2"], output_size=(OUTPUT, OUTPUT), sampling_ratio=2)
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1, num_classes=2, input_dim=CHANNELS * OUTPUT * OUTPUT,
        feature_shape=(CHANNELS, OUTPUT, OUTPUT), class_names=("bg", "spur"),
    )
    tree.node_weights[0] = np.random.default_rng(0).normal(size=tree.input_dim).astype(np.float32)
    tree.node_bias[0] = 0.37
    box_t = torch.tensor(box)
    bounds = _fpn_box_bounds(box_t, (SIZE, SIZE), (SIZE, SIZE), margin=2)
    batched = {"p2": level.unsqueeze(0)}
    grid = roi_align(batched, [box_t.unsqueeze(0)], [(SIZE, SIZE)])[0].numpy().astype(np.float32)
    step = tree.decision_path(grid.reshape(-1))[0]
    return level, roi_align, tree, box_t, bounds, batched, grid, step


def test_closed_form_masked_node_score_matches_a_real_repool():
    # Three boxes: interior, and two touching the level edge where bounds clamp tightest.
    for box in [(2.0, 2.0, 11.0, 11.0), (0.0, 0.0, 6.0, 6.0), (9.0, 0.0, 16.0, 7.0)]:
        level, roi_align, tree, box_t, bounds, batched, grid, step = _fixture(box)
        fx1, fy1, fx2, fy2 = bounds
        weight, bias = tree.node_weights[0], float(tree.node_bias[0])
        direction = 1.0 if step.went_left else -1.0

        # exact_fpn_contribution wants unbatched features, roi_align wants batched.
        contribution = exact_fpn_contribution(
            tree, grid, roi_align, {"p2": level}, "p2", box_t, (SIZE, SIZE), path=[step]
        )
        signed = contribution.sum(0) * direction
        signed_box = signed[fy1:fy2, fx1:fx2].numpy().reshape(-1).astype(np.float64)
        order = np.argsort(contribution.abs().sum(0)[fy1:fy2, fx1:fx2].numpy().reshape(-1))[::-1]
        n = signed_box.size

        class _Detector:
            pass

        detector = _Detector()
        detector.roi_align = roi_align

        for k in (0, n // 4, n // 2, 3 * n // 4, n):
            selected = np.zeros(n, dtype=bool)
            selected[order[:k]] = True
            for insert in (False, True):
                keep = selected if insert else ~selected
                masked = _masked_level(
                    level, bounds, torch.from_numpy(keep.reshape(fy2 - fy1, fx2 - fx1)),
                    start_from_zero=insert,
                )
                real = _node_score_after_masking(
                    detector, batched, "p2", masked, box_t, (SIZE, SIZE), weight, bias
                )
                closed = bias + signed_box[selected if insert else ~selected].sum()
                assert abs(closed - real) < 1e-4, (box, k, insert, closed, real)


def test_deletion_auc_below_insertion_auc_when_biggest_pixels_go_first():
    # score = sum of the pooled grid, box pixels ranked by their own value. Deleting the
    # biggest first drops the curve faster than inserting them first raises it, so
    # deletion AUC < insertion AUC, and both are bounded by [0, 1] for a positive score.
    level, roi_align, tree, box_t, bounds, batched, grid, step = _fixture((2.0, 2.0, 11.0, 11.0))
    fx1, fy1, fx2, fy2 = bounds
    order = np.argsort(level[:, fy1:fy2, fx1:fx2].sum(0).numpy().reshape(-1))[::-1]
    base_grid = roi_align(batched, [box_t.unsqueeze(0)], [(SIZE, SIZE)])
    del_auc, ins_auc = fpn_deletion_insertion_auc(
        roi_align, batched, "p2", box_t, (SIZE, SIZE), bounds, order, base_grid,
        lambda g: float(g.sum()),
    )
    assert 0.0 < del_auc < ins_auc < 1.0


if __name__ == "__main__":
    test_closed_form_masked_node_score_matches_a_real_repool()
    test_deletion_auc_below_insertion_auc_when_biggest_pixels_go_first()
    print("OK")
