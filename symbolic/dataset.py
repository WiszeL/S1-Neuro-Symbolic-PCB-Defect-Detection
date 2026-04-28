from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SymbolicTensorBundle:
    feature_grids: Tensor
    feature_vectors: Tensor
    teacher_labels: Tensor
    teacher_logits: Tensor
    teacher_scores: Tensor
    proposal_boxes: Tensor
    transformed_proposal_boxes: Tensor
    matched_gt_boxes: Tensor | None
    has_matched_gt: Tensor | None
    image_ids: Tensor
    image_sizes: Tensor
    transformed_image_sizes: Tensor
    gt_labels: Tensor | None
    gt_iou: Tensor | None
    image_paths: tuple[str, ...]
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
    include_background: bool,
    min_teacher_score: float | None,
    max_samples_total: int | None,
    random_state: int,
) -> Tensor:
    labels = metadata["teacher_labels"]
    scores = metadata["teacher_scores"].max(dim=1).values

    mask = torch.ones((labels.shape[0],), dtype=torch.bool)
    if not include_background:
        mask &= labels > 0
    if min_teacher_score is not None:
        mask &= scores >= float(min_teacher_score)

    candidate_indices = torch.where(mask)[0]
    if candidate_indices.numel() == 0:
        raise ValueError("The symbolic export did not contain any RoI samples after filtering.")

    if max_samples_total is None:
        return candidate_indices

    relative_keep = _balanced_sample_indices(
        labels[candidate_indices],
        max_samples_total=max_samples_total,
        random_state=random_state,
    )
    return candidate_indices[relative_keep]


def _index_optional_tensor(tensor: Tensor | None, indices: Tensor) -> Tensor | None:
    if tensor is None:
        return None
    return tensor[indices]


def _load_array_feature_grids(payload: dict[str, Any], indices: Tensor) -> Tensor:
    feature_storage = payload["feature_storage"]
    feature_path = _resolve_artifact_path(feature_storage["path"])
    feature_shape = tuple(int(value) for value in feature_storage["shape"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing symbolic feature storage: {feature_path}")

    feature_array = np.memmap(
        feature_path,
        dtype=_numpy_dtype(feature_storage["dtype"]),
        mode="r",
        shape=feature_shape,
    )
    selected = np.asarray(feature_array[indices.cpu().numpy()])
    return torch.from_numpy(selected)


def _flatten_array_memmap_payload(
    payload: dict[str, Any],
    include_background: bool,
    min_teacher_score: float | None,
    max_samples_total: int | None,
    random_state: int,
) -> SymbolicTensorBundle:
    metadata_path = _resolve_artifact_path(payload["metadata_path"])
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    keep = _selected_indices_from_metadata(
        metadata,
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=max_samples_total,
        random_state=random_state,
    )

    feature_grids = _load_array_feature_grids(payload, keep)
    selected_image_paths = tuple(str(metadata["image_paths"][int(index)]) for index in keep.tolist())

    return SymbolicTensorBundle(
        feature_grids=feature_grids,
        feature_vectors=feature_grids.flatten(start_dim=1),
        teacher_labels=metadata["teacher_labels"][keep],
        teacher_logits=metadata["teacher_logits"][keep],
        teacher_scores=metadata["teacher_scores"][keep],
        proposal_boxes=metadata["proposal_boxes"][keep],
        transformed_proposal_boxes=metadata["transformed_proposal_boxes"][keep],
        matched_gt_boxes=_index_optional_tensor(metadata.get("matched_gt_boxes"), keep),
        has_matched_gt=_index_optional_tensor(metadata.get("has_matched_gt"), keep),
        image_ids=metadata["image_ids"][keep],
        image_sizes=metadata["image_sizes"][keep],
        transformed_image_sizes=metadata["transformed_image_sizes"][keep],
        gt_labels=_index_optional_tensor(metadata.get("gt_labels"), keep),
        gt_iou=_index_optional_tensor(metadata.get("gt_iou"), keep),
        image_paths=selected_image_paths,
        feature_shape=tuple(int(value) for value in payload["feature_shape"]),
        class_names=tuple(payload["class_names"]),
    )


def flatten_exported_symbolic_payload(
    payload: dict[str, Any],
    include_background: bool = True,
    min_teacher_score: float | None = None,
    max_samples_total: int | None = None,
    random_state: int = 42,
) -> SymbolicTensorBundle:
    if payload.get("storage_format") != "array_memmap_v1":
        raise ValueError(
            "Unsupported symbolic export format. Regenerate the export to create "
            "an array_memmap_v1 symbolic artifact."
        )

    return _flatten_array_memmap_payload(
        payload,
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=max_samples_total,
        random_state=random_state,
    )


class SymbolicRoIDataset(Dataset[dict[str, Tensor | str | int | float | None]]):
    def __init__(
        self,
        export_path: str | Path,
        include_background: bool = True,
        min_teacher_score: float | None = None,
        max_samples_total: int | None = None,
        random_state: int = 42,
    ) -> None:
        payload = torch.load(export_path, map_location="cpu", weights_only=True)
        self.bundle = flatten_exported_symbolic_payload(
            payload,
            include_background=include_background,
            min_teacher_score=min_teacher_score,
            max_samples_total=max_samples_total,
            random_state=random_state,
        )

    def __len__(self) -> int:
        return int(self.bundle.teacher_labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float | None]:
        item: dict[str, Tensor | str | int | float | None] = {
            "feature_grid": self.bundle.feature_grids[index],
            "feature_vector": self.bundle.feature_vectors[index],
            "teacher_label": int(self.bundle.teacher_labels[index]),
            "teacher_logits": self.bundle.teacher_logits[index],
            "teacher_scores": self.bundle.teacher_scores[index],
            "proposal_box": self.bundle.proposal_boxes[index],
            "transformed_proposal_box": self.bundle.transformed_proposal_boxes[index],
            "image_id": int(self.bundle.image_ids[index]),
            "image_size": self.bundle.image_sizes[index],
            "transformed_image_size": self.bundle.transformed_image_sizes[index],
            "image_path": self.bundle.image_paths[index],
        }
        if self.bundle.matched_gt_boxes is not None:
            item["matched_gt_box"] = self.bundle.matched_gt_boxes[index]
        if self.bundle.has_matched_gt is not None:
            item["has_matched_gt"] = bool(self.bundle.has_matched_gt[index])
        if self.bundle.gt_labels is not None:
            item["gt_label"] = int(self.bundle.gt_labels[index])
        if self.bundle.gt_iou is not None:
            item["gt_iou"] = float(self.bundle.gt_iou[index])
        return item
