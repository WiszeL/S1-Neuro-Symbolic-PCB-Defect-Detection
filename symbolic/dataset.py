from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SymbolicArrayBundle:
    feature_grids: np.ndarray
    feature_vectors: np.ndarray
    teacher_labels: Tensor
    proposal_boxes: Tensor
    matched_gt_boxes: Tensor | None
    has_matched_gt: Tensor | None
    gt_iou: Tensor | None
    feature_shape: tuple[int, int, int]
    class_names: tuple[str, ...]


def _numpy_dtype(storage_dtype: str) -> np.dtype[Any]:
    if storage_dtype == "float16":
        return np.dtype("float16")
    if storage_dtype == "float32":
        return np.dtype("float32")
    raise ValueError(f"Unsupported symbolic storage dtype: {storage_dtype}")


def _resolve_artifact_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    normalized_path = Path(str(path_value).replace("\\", "/"))
    if normalized_path.exists():
        return normalized_path

    return path


def _positive_all_neg_ratio_indices(
    labels: Tensor,
    neg_ratio: float,
    random_state: int,
    background_label: int = 0,
) -> Tensor:
    """Select ALL positive (non-background) RoIs and N× background RoIs.

    This mirrors the common object detection sampling strategy (e.g. Faster R-CNN
    RPN uses 1:1, but 1:3 is standard for the second stage).  The tree sees every
    single defect example, and the ``random_state`` controls only which background
    RoIs are sampled.

    Args:
        labels: 1-D tensor of integer class labels.
        neg_ratio: How many background RoIs per total positive count.
                   e.g. 3.0 means ``num_bg = 3 * num_positive``.
        random_state: Seed for the background random shuffle.
        background_label: Which label value is "background" (default 0).
    """
    positive_mask = labels != background_label
    negative_mask = labels == background_label

    positive_indices = torch.where(positive_mask)[0]
    negative_indices = torch.where(negative_mask)[0]

    total_positive = positive_indices.shape[0]
    max_neg = int(total_positive * neg_ratio)

    if negative_indices.shape[0] <= max_neg:
        # Not enough backgrounds — take them all
        selected_neg = negative_indices
    else:
        generator = torch.Generator()
        generator.manual_seed(random_state)
        perm = torch.randperm(negative_indices.shape[0], generator=generator)
        selected_neg = negative_indices[perm[:max_neg]]

    return torch.cat([positive_indices, selected_neg], dim=0)


def _selected_indices_from_metadata(
    metadata: dict[str, Any],
    random_state: int,
    neg_ratio: float | None = None,
) -> Tensor:
    labels = metadata["teacher_labels"]

    if neg_ratio is not None:
        # ALL positive RoIs + neg_ratio × background RoIs.
        keep = _positive_all_neg_ratio_indices(
            labels,
            neg_ratio=neg_ratio,
            random_state=random_state,
        )
    else:
        keep = torch.arange(labels.shape[0], dtype=torch.int64)

    # Sort indices for efficient sequential memmap access
    return torch.sort(keep).values


def _index_optional_tensor(tensor: Tensor | None, indices: Tensor) -> Tensor | None:
    if tensor is None:
        return None
    return tensor[indices]


def _is_full_contiguous_selection(indices: Tensor, row_count: int) -> bool:
    if int(indices.numel()) != int(row_count):
        return False
    if row_count == 0:
        return True
    expected = torch.arange(row_count, dtype=torch.int64)
    return bool(torch.equal(indices.cpu(), expected))


def _feature_cache_path(
    payload: dict[str, Any],
    indices: Tensor,
    feature_dtype: np.dtype[Any],
) -> Path:
    feature_storage = payload["feature_storage"]
    feature_path = _resolve_artifact_path(feature_storage["path"])
    storage_dir = _resolve_artifact_path(
        payload.get("storage_dir", feature_path.parent)
    )
    if not storage_dir.exists():
        storage_dir = feature_path.parent

    index_array = indices.cpu().numpy().astype(np.int64, copy=False)
    digest = hashlib.sha1()
    digest.update(str(feature_path).encode("utf-8"))
    digest.update(str(tuple(feature_storage["shape"])).encode("utf-8"))
    digest.update(str(feature_storage["dtype"]).encode("utf-8"))
    digest.update(str(feature_dtype).encode("utf-8"))
    digest.update(index_array.tobytes())
    key = digest.hexdigest()[:16]
    return (
        storage_dir / f"features_{feature_dtype.name}_{index_array.shape[0]}_{key}.dat"
    )


def _open_selected_feature_grids(
    payload: dict[str, Any],
    indices: Tensor,
    feature_dtype: str,
    cache_chunk_size: int,
) -> np.ndarray:
    feature_storage = payload["feature_storage"]
    feature_path = _resolve_artifact_path(feature_storage["path"])
    storage_shape = tuple(int(value) for value in feature_storage["shape"])
    if len(storage_shape) != 4:
        raise ValueError(
            f"Expected feature storage shape [N, C, H, W], got {storage_shape}."
        )
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing symbolic feature storage: {feature_path}")

    source_dtype = _numpy_dtype(feature_storage["dtype"])
    target_dtype = _numpy_dtype(feature_dtype)
    source = np.memmap(
        feature_path,
        dtype=source_dtype,
        mode="r",
        shape=storage_shape,
    )

    row_count = storage_shape[0]
    selected_count = int(indices.numel())
    grid_shape = storage_shape[1:]
    if (
        _is_full_contiguous_selection(indices, row_count)
        and source_dtype == target_dtype
    ):
        return source

    cache_path = _feature_cache_path(payload, indices, target_dtype)
    expected_bytes = int(selected_count * np.prod(grid_shape) * target_dtype.itemsize)
    if not cache_path.exists() or cache_path.stat().st_size != expected_bytes:
        source_flat = source.reshape(row_count, -1)
        cache = np.memmap(
            cache_path,
            dtype=target_dtype,
            mode="w+",
            shape=(selected_count, *grid_shape),
        )
        cache_flat = cache.reshape(selected_count, -1)
        keep = indices.cpu().numpy().astype(np.int64, copy=False)
        chunk_size = max(cache_chunk_size, 1)
        for start in range(0, selected_count, chunk_size):
            stop = min(start + chunk_size, selected_count)
            cache_flat[start:stop] = source_flat[keep[start:stop]].astype(
                target_dtype, copy=False
            )
        cache.flush()
        del cache

    return np.memmap(
        cache_path,
        dtype=target_dtype,
        mode="r",
        shape=(selected_count, *grid_shape),
    )


def open_exported_symbolic_array_payload(
    payload: dict[str, Any],
    random_state: int = 42,
    feature_dtype: str = "float32",
    cache_chunk_size: int = 256,
    neg_ratio: float | None = None,
) -> SymbolicArrayBundle:
    if payload.get("storage_format") != "array_memmap_v1":
        raise ValueError(
            "Unsupported symbolic export format. Regenerate the export to create "
            "an array_memmap_v1 symbolic artifact."
        )

    metadata_path = _resolve_artifact_path(payload["metadata_path"])
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)

    keep = _selected_indices_from_metadata(
        metadata,
        random_state=random_state,
        neg_ratio=neg_ratio,
    )

    feature_grids = _open_selected_feature_grids(
        payload,
        keep,
        feature_dtype=feature_dtype,
        cache_chunk_size=cache_chunk_size,
    )

    return SymbolicArrayBundle(
        feature_grids=feature_grids,
        feature_vectors=feature_grids.reshape(feature_grids.shape[0], -1),
        teacher_labels=metadata["teacher_labels"][keep],
        proposal_boxes=metadata["proposal_boxes"][keep],
        matched_gt_boxes=_index_optional_tensor(metadata.get("matched_gt_boxes"), keep),
        has_matched_gt=_index_optional_tensor(metadata.get("has_matched_gt"), keep),
        gt_iou=_index_optional_tensor(metadata.get("gt_iou"), keep),
        feature_shape=tuple(int(value) for value in payload["feature_shape"]),
        class_names=tuple(payload["class_names"]),
    )
