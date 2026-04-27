from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _resolve_shard_path(shard_path: str | Path) -> Path:
    path = Path(shard_path)
    if path.exists():
        return path

    normalized_path = Path(str(shard_path).replace("\\", "/"))
    if normalized_path.exists():
        return normalized_path

    return path


def _iter_symbolic_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("storage_format") == "sharded_pt_v1":
        return [
            torch.load(_resolve_shard_path(record["shard_path"]), map_location="cpu", weights_only=True)
            for record in payload["records"]
        ]
    return list(payload["records"])


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


def flatten_exported_symbolic_payload(
    payload: dict[str, Any],
    include_background: bool = True,
    min_teacher_score: float | None = None,
    max_samples_total: int | None = None,
    random_state: int = 42,
) -> SymbolicTensorBundle:
    feature_grids: list[Tensor] = []
    feature_vectors: list[Tensor] = []
    teacher_labels: list[Tensor] = []
    teacher_logits: list[Tensor] = []
    teacher_scores: list[Tensor] = []
    proposal_boxes: list[Tensor] = []
    transformed_proposal_boxes: list[Tensor] = []
    matched_gt_boxes: list[Tensor] = []
    has_matched_gt: list[Tensor] = []
    image_ids: list[Tensor] = []
    image_sizes: list[Tensor] = []
    transformed_image_sizes: list[Tensor] = []
    gt_labels: list[Tensor] = []
    gt_iou: list[Tensor] = []
    image_paths: list[str] = []

    for record in _iter_symbolic_records(payload):
        grids = record["pooled_features"]
        logits = record["teacher_logits"]
        labels = record["teacher_labels"]
        scores = record["teacher_scores"].max(dim=1).values
        transformed_boxes = record.get("transformed_proposal_boxes", record["proposal_boxes"])

        mask = torch.ones((labels.shape[0],), dtype=torch.bool)
        if not include_background:
            mask &= labels > 0
        if min_teacher_score is not None:
            mask &= scores >= float(min_teacher_score)

        if mask.sum() == 0:
            continue

        selected_grids = grids[mask]
        feature_grids.append(selected_grids)
        feature_vectors.append(selected_grids.flatten(start_dim=1))
        teacher_labels.append(labels[mask])
        teacher_logits.append(logits[mask])
        teacher_scores.append(record["teacher_scores"][mask])
        proposal_boxes.append(record["proposal_boxes"][mask])
        transformed_proposal_boxes.append(transformed_boxes[mask])
        image_ids.append(torch.full((int(mask.sum().item()),), int(record["image_id"]), dtype=torch.int64))
        image_sizes.append(record["image_size"].repeat(int(mask.sum().item()), 1))
        transformed_image_sizes.append(record["transformed_image_size"].repeat(int(mask.sum().item()), 1))
        image_paths.extend([str(record["image_path"])] * int(mask.sum().item()))

        if "matched_gt_boxes" in record:
            matched_gt_boxes.append(record["matched_gt_boxes"][mask])
        if "has_matched_gt" in record:
            has_matched_gt.append(record["has_matched_gt"][mask])
        if "gt_labels" in record:
            gt_labels.append(record["gt_labels"][mask])
        if "gt_iou" in record:
            gt_iou.append(record["gt_iou"][mask])

    if not feature_grids:
        raise ValueError("The symbolic export did not contain any RoI samples after filtering.")

    stacked_grids = torch.cat(feature_grids, dim=0)
    stacked_vectors = torch.cat(feature_vectors, dim=0)
    stacked_labels = torch.cat(teacher_labels, dim=0)
    stacked_logits = torch.cat(teacher_logits, dim=0)
    stacked_scores = torch.cat(teacher_scores, dim=0)
    stacked_boxes = torch.cat(proposal_boxes, dim=0)
    stacked_transformed_boxes = torch.cat(transformed_proposal_boxes, dim=0)
    stacked_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0) if matched_gt_boxes else None
    stacked_has_matched_gt = torch.cat(has_matched_gt, dim=0) if has_matched_gt else None
    stacked_image_ids = torch.cat(image_ids, dim=0)
    stacked_image_sizes = torch.cat(image_sizes, dim=0)
    stacked_transformed_image_sizes = torch.cat(transformed_image_sizes, dim=0)
    stacked_gt_labels = torch.cat(gt_labels, dim=0) if gt_labels else None
    stacked_gt_iou = torch.cat(gt_iou, dim=0) if gt_iou else None

    if max_samples_total is not None:
        keep = _balanced_sample_indices(
            stacked_labels,
            max_samples_total=max_samples_total,
            random_state=random_state,
        )
        stacked_grids = stacked_grids[keep]
        stacked_vectors = stacked_vectors[keep]
        stacked_labels = stacked_labels[keep]
        stacked_logits = stacked_logits[keep]
        stacked_scores = stacked_scores[keep]
        stacked_boxes = stacked_boxes[keep]
        stacked_transformed_boxes = stacked_transformed_boxes[keep]
        if stacked_matched_gt_boxes is not None:
            stacked_matched_gt_boxes = stacked_matched_gt_boxes[keep]
        if stacked_has_matched_gt is not None:
            stacked_has_matched_gt = stacked_has_matched_gt[keep]
        stacked_image_ids = stacked_image_ids[keep]
        stacked_image_sizes = stacked_image_sizes[keep]
        stacked_transformed_image_sizes = stacked_transformed_image_sizes[keep]
        image_paths = [image_paths[int(index)] for index in keep.tolist()]
        if stacked_gt_labels is not None:
            stacked_gt_labels = stacked_gt_labels[keep]
        if stacked_gt_iou is not None:
            stacked_gt_iou = stacked_gt_iou[keep]

    return SymbolicTensorBundle(
        feature_grids=stacked_grids,
        feature_vectors=stacked_vectors,
        teacher_labels=stacked_labels,
        teacher_logits=stacked_logits,
        teacher_scores=stacked_scores,
        proposal_boxes=stacked_boxes,
        transformed_proposal_boxes=stacked_transformed_boxes,
        matched_gt_boxes=stacked_matched_gt_boxes,
        has_matched_gt=stacked_has_matched_gt,
        image_ids=stacked_image_ids,
        image_sizes=stacked_image_sizes,
        transformed_image_sizes=stacked_transformed_image_sizes,
        gt_labels=stacked_gt_labels,
        gt_iou=stacked_gt_iou,
        image_paths=tuple(image_paths),
        feature_shape=tuple(int(value) for value in stacked_grids.shape[1:]),
        class_names=tuple(payload["class_names"]),
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
