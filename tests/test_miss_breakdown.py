"""Each FRCNN hit that NeSy misses lands in exactly one cause: score, class or none."""

import numpy as np
import torch

from neurosym.evaluation import breakdown_false_positives, breakdown_nesy_misses
from symbolic.sodt import SparseObliqueDecisionTreeClassifier


def _detections(boxes, scores, labels, pooled=None):
    detection = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    if pooled is not None:
        detection["pooled_features"] = pooled
    return detection


def test_misses_split_into_score_class_and_none():
    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=2, num_classes=3, input_dim=4, feature_shape=(1, 2, 2)
    )
    tree.node_weights[:] = np.random.default_rng(0).normal(size=tree.node_weights.shape)
    pooled = torch.rand(3, 1, 2, 2)

    box = [0.0, 0.0, 10.0, 10.0]
    far = [100.0, 100.0, 110.0, 110.0]
    targets = [{"boxes": torch.tensor([box, box, box, box]), "labels": torch.tensor([1, 1, 1, 1])}]
    # FRCNN finds all four (same box, so the same detection serves each).
    frcnn = [_detections([box], [0.9], [1])]
    # NeSy: label-1 box under the threshold, plus a wrong-class box above it, plus a far box.
    nesy = [_detections([box, box, far], [0.3, 0.8, 0.9], [1, 2, 1], pooled=pooled)]

    result = breakdown_nesy_misses(frcnn, nesy, targets, tree, 0.5, 0.5)

    assert result["frcnn_hits"] == {1: 4}
    # "score" outranks "class": a same-class box under the threshold exists.
    assert result["causes"] == {1: {"score": 4, "class": 0, "none": 0}}
    assert sum(result["weakest_depth"].values()) == 4
    assert len(result["score_misses"]) == 4

    # Precomputed log certainties give the same misses and weakest depths as pooled_features.
    nesy_cached = [dict(nesy[0], log_certainties=torch.from_numpy(
        tree.path_log_certainties(pooled.flatten(start_dim=1).numpy())[1]
    ))]
    cached = breakdown_nesy_misses(frcnn, nesy_cached, targets, tree, 0.5, 0.5)
    assert cached["weakest_depth"] == result["weakest_depth"]
    assert len(cached["score_misses"]) == 4

    nesy_no_weak = [_detections([box, far], [0.8, 0.9], [2, 1], pooled=pooled[:2])]
    assert breakdown_nesy_misses(frcnn, nesy_no_weak, targets, tree, 0.5, 0.5)["causes"] == {
        1: {"score": 0, "class": 4, "none": 0}
    }

    nesy_none = [_detections([far], [0.9], [1], pooled=pooled[:1])]
    assert breakdown_nesy_misses(frcnn, nesy_none, targets, tree, 0.5, 0.5)["causes"] == {
        1: {"score": 0, "class": 0, "none": 4}
    }


def test_false_positives_split_into_four_kinds():
    a, b = [0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]
    partial, far = [0.0, 0.0, 10.0, 3.5], [200.0, 200.0, 210.0, 210.0]
    targets = [{"boxes": torch.tensor([a, b]), "labels": torch.tensor([1, 2])}]
    # a: TP (0.9) then duplicate (0.8); b as class 1: wrong class; partial: IoU 0.35; far: nothing;
    # b as class 2: TP; a final 0.2 box under the cutoff is ignored.
    nesy = [_detections(
        [a, a, b, partial, far, b, far], [0.9, 0.8, 0.7, 0.6, 0.55, 0.95, 0.2], [1, 1, 1, 1, 1, 2, 1]
    )]
    reference = [_detections([far], [0.9], [1])]

    result = breakdown_false_positives(nesy, targets, 0.5, 0.5, reference=reference)

    assert result["counts"] == {"duplicate": 1, "wrong_class": 1, "mislocalized": 1, "background": 1}
    assert result["shared"] == {"duplicate": 0, "wrong_class": 0, "mislocalized": 0, "background": 1}
    assert result["total"] == 4


if __name__ == "__main__":
    test_misses_split_into_score_class_and_none()
    test_false_positives_split_into_four_kinds()
    print("OK")
