from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neuro.datasets import DeepPCBDataset
from neuro.detector import build_detector
from neuro.utils import next_run_artifacts
from symbolic.dataset import flatten_exported_symbolic_payload
from symbolic.sodt import SparseObliqueDecisionTreeClassifier
from symbolic.tao import fit_tree_with_tao


def test_deeppcb_dataset_resolves_test_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "DeepPCB"
    image_dir = dataset_root / "group00001" / "00001"
    annotation_dir = dataset_root / "group00001" / "00001_not"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    Image.new("L", (640, 640), color=0).save(image_dir / "00001000_test.jpg")
    (annotation_dir / "00001000.txt").write_text("10 20 30 40 1\n", encoding="utf-8")
    (dataset_root / "trainval.txt").write_text(
        "group00001/00001/00001000.jpg group00001/00001_not/00001000.txt\n",
        encoding="utf-8",
    )

    dataset = DeepPCBDataset(dataset_root=dataset_root, split_file="trainval.txt")
    image, target = dataset[0]

    assert image.shape == (3, 640, 640)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]


def test_run_artifacts_increment(tmp_path: Path) -> None:
    artifacts = next_run_artifacts(tmp_path)
    assert artifacts.run_name == "run1"

    artifacts.checkpoint_path.write_bytes(b"checkpoint")
    next_artifacts = next_run_artifacts(tmp_path)
    assert next_artifacts.run_name == "run2"


def test_detector_builds() -> None:
    config = {
        "model": {
            "num_classes": 6,
            "class_names": ["open", "short", "mouse_bite", "spur", "pinhole", "spurious_copper"],
            "backbone": {"pretrained": False, "freeze_batch_norm": False},
            "neck": {"attention_reduction": 16},
            "image": {
                "train_min_sizes": [480, 560, 640],
                "test_min_size": 640,
                "max_size": 880,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "anchors": {
                "sizes": [[16], [32], [64], [128], [256]],
                "aspect_ratios": [[0.5, 1.0, 2.0]] * 5,
            },
            "roi": {
                "featmap_names": ["p2", "p3", "p4", "p5", "p6"],
                "pool_size": 7,
                "sampling_ratio": 2,
                "representation_size": 1024,
            },
            "rpn": {
                "fg_iou_thresh": 0.7,
                "bg_iou_thresh": 0.3,
                "batch_size_per_image": 256,
                "positive_fraction": 0.5,
                "pre_nms_top_n_train": 2000,
                "pre_nms_top_n_test": 1000,
                "post_nms_top_n_train": 1000,
                "post_nms_top_n_test": 300,
                "nms_thresh": 0.7,
            },
            "box": {
                "fg_iou_thresh": 0.5,
                "bg_iou_thresh": 0.5,
                "batch_size_per_image": 512,
                "positive_fraction": 0.25,
                "bbox_reg_weights": None,
                "score_thresh": 0.05,
                "detections_per_img": 100,
            },
            "soft_nms": {
                "enabled": True,
                "method": "linear",
                "iou_thresh": 0.5,
                "sigma": 0.5,
                "score_thresh": 0.001,
            },
        }
    }

    model = build_detector(config)
    assert model.roi_heads.box_predictor.cls_score.out_features == 7


def test_symbolic_payload_flattening() -> None:
    payload = {
        "class_names": ("__background__", "open"),
        "records": [
            {
                "image_id": 0,
                "image_path": "sample.jpg",
                "proposal_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
                "pooled_features": torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
                "teacher_labels": torch.tensor([0, 1], dtype=torch.int64),
                "teacher_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
                "teacher_scores": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
            }
        ],
    }

    bundle = flatten_exported_symbolic_payload(payload, include_background=True)
    assert bundle.feature_grids.shape == (2, 1, 2, 2)
    assert bundle.feature_vectors.shape == (2, 4)
    assert bundle.teacher_labels.tolist() == [0, 1]


def test_sparse_oblique_tree_trains_and_roundtrips() -> None:
    features = np.array(
        [
            [-1.0, -1.0],
            [-0.5, -0.25],
            [0.5, 0.25],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    tree = SparseObliqueDecisionTreeClassifier(
        max_depth=1,
        num_classes=2,
        input_dim=2,
        feature_shape=(1, 1, 2),
        class_names=("__background__", "open"),
    )
    history = fit_tree_with_tao(tree, features, labels, iterations=3, logistic_max_iter=50, l1_lambda=1e-3)
    restored_tree = SparseObliqueDecisionTreeClassifier.from_state_dict(tree.to_state_dict())

    assert history
    assert restored_tree.predict(features).tolist() == tree.predict(features).tolist()
