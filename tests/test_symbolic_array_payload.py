from pathlib import Path

import numpy as np
import torch

from symbolic.dataset import flatten_exported_symbolic_payload


def test_symbolic_array_payload_samples_before_feature_loading(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.dat"
    metadata_path = tmp_path / "metadata.pt"
    features = np.arange(24, dtype=np.float16).reshape(6, 1, 2, 2)
    feature_array = np.memmap(feature_path, dtype=np.float16, mode="w+", shape=features.shape)
    feature_array[:] = features
    feature_array.flush()
    del feature_array

    metadata = {
        "teacher_labels": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        "teacher_logits": torch.zeros((6, 3), dtype=torch.float16),
        "teacher_scores": torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
                [0.2, 0.8, 0.0],
                [0.1, 0.0, 0.9],
                [0.2, 0.0, 0.8],
            ],
            dtype=torch.float16,
        ),
        "proposal_boxes": torch.zeros((6, 4), dtype=torch.float32),
        "transformed_proposal_boxes": torch.zeros((6, 4), dtype=torch.float32),
        "matched_gt_boxes": torch.zeros((6, 4), dtype=torch.float32),
        "has_matched_gt": torch.ones((6,), dtype=torch.bool),
        "image_ids": torch.arange(6, dtype=torch.int64),
        "image_sizes": torch.full((6, 2), 640, dtype=torch.int64),
        "transformed_image_sizes": torch.full((6, 2), 640, dtype=torch.int64),
        "gt_labels": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),
        "gt_iou": torch.ones((6,), dtype=torch.float16),
        "image_paths": tuple(f"image_{index}.jpg" for index in range(6)),
    }
    torch.save(metadata, metadata_path)

    payload = {
        "storage_format": "array_memmap_v1",
        "class_names": ("__background__", "open", "short"),
        "feature_shape": (1, 2, 2),
        "metadata_path": metadata_path.as_posix(),
        "feature_storage": {
            "path": feature_path.as_posix(),
            "dtype": "float16",
            "shape": features.shape,
        },
    }

    bundle = flatten_exported_symbolic_payload(
        payload,
        include_background=False,
        max_samples_total=2,
        random_state=0,
    )

    assert bundle.feature_grids.shape == (2, 1, 2, 2)
    assert bundle.feature_vectors.shape == (2, 4)
    assert bundle.teacher_labels.tolist() == [1, 2]
