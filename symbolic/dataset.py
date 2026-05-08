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


def _balanced_sample_indices(
    labels: Tensor,
    max_samples_total: int,
    random_state: int,
) -> Tensor:
    if labels.shape[0] <= max_samples_total:
        return torch.arange(labels.shape[0], dtype=torch.int64)

    generator = torch.Generator()
    generator.manual_seed(random_state)

    unique_labels = torch.unique(labels)
    per_class_quota = max(max_samples_total // max(unique_labels.numel(), 1), 1)
    selected_indices: list[Tensor] = []
    remaining_indices: list[Tensor] = []

    for label in unique_labels.tolist():
        class_indices = torch.where(labels == label)[0]
        shuffled = class_indices[torch.randperm(class_indices.shape[0], generator=generator)]
        selected_indices.append(shuffled[:per_class_quota])
        remaining_indices.append(shuffled[per_class_quota:])

    combined = torch.cat(selected_indices, dim=0)
    if combined.shape[0] >= max_samples_total:
        return combined[:max_samples_total]

    if remaining_indices:
        leftover = torch.cat(remaining_indices, dim=0)
        if leftover.shape[0] > 0:
            shuffled_leftover = leftover[torch.randperm(leftover.shape[0], generator=generator)]
            needed = max_samples_total - combined.shape[0]
            combined = torch.cat([combined, shuffled_leftover[:needed]], dim=0)

    return combined


def _selected_indices_from_metadata(
    metadata: dict[str, Any],
    max_samples_total: int | None,
    random_state: int,
) -> Tensor:
    labels = metadata["teacher_labels"]
    candidate_indices = torch.arange(labels.shape[0], dtype=torch.int64)

    if max_samples_total is None:
        return candidate_indices

    return _balanced_sample_indices(
        labels[candidate_indices],
        max_samples_total=max_samples_total,
        random_state=random_state,
    )


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
    storage_dir = _resolve_artifact_path(payload.get("storage_dir", feature_path.parent))
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
    return storage_dir / f"features_{feature_dtype.name}_{index_array.shape[0]}_{key}.dat"


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
        raise ValueError(f"Expected feature storage shape [N, C, H, W], got {storage_shape}.")
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
    if _is_full_contiguous_selection(indices, row_count) and source_dtype == target_dtype:
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
        chunk_size = max(int(cache_chunk_size), 1)
        for start in range(0, selected_count, chunk_size):
            stop = min(start + chunk_size, selected_count)
            cache_flat[start:stop] = source_flat[keep[start:stop]].astype(target_dtype, copy=False)
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
    max_samples_total: int | None = None,
    random_state: int = 42,
    feature_dtype: str = "float32",
    cache_chunk_size: int = 256,
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
        max_samples_total=max_samples_total,
        random_state=random_state,
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