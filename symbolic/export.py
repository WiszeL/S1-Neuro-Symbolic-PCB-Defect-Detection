from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from util.device import select_device
from util.io import ensure_dir


_METADATA_TENSOR_FIELDS = [
    "teacher_labels",
    "teacher_logits",
    "teacher_scores",
    "proposal_boxes",
    "transformed_proposal_boxes",
    "matched_gt_boxes",
    "has_matched_gt",
    "image_ids",
    "image_sizes",
    "transformed_image_sizes",
    "gt_labels",
    "gt_iou",
]


def _storage_dtypes(storage_dtype: str) -> tuple[torch.dtype, np.dtype[Any]]:
    if storage_dtype == "float16":
        return torch.float16, np.dtype("float16")
    if storage_dtype == "float32":
        return torch.float32, np.dtype("float32")
    raise ValueError("storage_dtype must be either 'float16' or 'float32'.")


def _truncate_feature_storage(
    feature_path: Path,
    row_count: int,
    feature_shape: tuple[int, int, int],
    numpy_dtype: np.dtype[Any],
) -> None:
    feature_count = int(np.prod(feature_shape))
    with feature_path.open("r+b") as feature_file:
        feature_file.truncate(row_count * feature_count * numpy_dtype.itemsize)


def _select_teacher_roi_indices(
    teacher_labels: torch.Tensor,
    teacher_scores: torch.Tensor,
    max_positive_rois_per_image: int,
    max_background_rois_per_image: int,
) -> torch.Tensor:
    if teacher_labels.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=teacher_labels.device)

    non_background_scores = teacher_scores[:, 1:].max(dim=1).values
    positive_indices = torch.where(teacher_labels > 0)[0]
    background_indices = torch.where(teacher_labels == 0)[0]

    if positive_indices.numel() > 0:
        positive_order = torch.argsort(non_background_scores[positive_indices], descending=True)
        positive_indices = positive_indices[positive_order[:max_positive_rois_per_image]]

    if background_indices.numel() > 0:
        background_order = torch.argsort(non_background_scores[background_indices], descending=True)
        background_indices = background_indices[background_order[:max_background_rois_per_image]]

    keep = torch.cat([positive_indices, background_indices], dim=0)
    if keep.numel() == 0:
        return keep
    return keep[torch.argsort(keep)]


def _match_proposals_to_ground_truth(
    proposal_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_boxes = target_boxes.to(device=proposal_boxes.device, dtype=proposal_boxes.dtype)
    target_labels = target_labels.to(device=proposal_boxes.device, dtype=torch.int64)

    if proposal_boxes.numel() == 0:
        return {
            "matched_gt_boxes": torch.zeros((0, 4), dtype=torch.float32, device=proposal_boxes.device),
            "matched_gt_labels": torch.zeros((0,), dtype=torch.int64, device=proposal_boxes.device),
            "matched_gt_iou": torch.zeros((0,), dtype=torch.float32, device=proposal_boxes.device),
            "has_matched_gt": torch.zeros((0,), dtype=torch.bool, device=proposal_boxes.device),
        }

    if target_boxes.numel() == 0:
        num_boxes = proposal_boxes.shape[0]
        return {
            "matched_gt_boxes": torch.zeros((num_boxes, 4), dtype=torch.float32, device=proposal_boxes.device),
            "matched_gt_labels": torch.zeros((num_boxes,), dtype=torch.int64, device=proposal_boxes.device),
            "matched_gt_iou": torch.zeros((num_boxes,), dtype=torch.float32, device=proposal_boxes.device),
            "has_matched_gt": torch.zeros((num_boxes,), dtype=torch.bool, device=proposal_boxes.device),
        }

    overlaps = box_iou(proposal_boxes, target_boxes)
    matched_iou, matched_indices = overlaps.max(dim=1)
    matched_boxes = target_boxes[matched_indices]
    matched_labels = target_labels[matched_indices]
    has_match = matched_iou > 0
    matched_boxes = torch.where(
        has_match[:, None],
        matched_boxes,
        torch.zeros_like(matched_boxes),
    )
    matched_labels = torch.where(
        has_match,
        matched_labels,
        torch.zeros_like(matched_labels),
    )
    return {
        "matched_gt_boxes": matched_boxes,
        "matched_gt_labels": matched_labels,
        "matched_gt_iou": matched_iou,
        "has_matched_gt": has_match,
    }


def _build_symbolic_record(
    dataset: Any,
    sample_index: int,
    target: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    keep: torch.Tensor,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    selected_proposal_boxes = teacher_output["proposal_boxes"][keep]
    matched_ground_truth = _match_proposals_to_ground_truth(
        proposal_boxes=selected_proposal_boxes,
        target_boxes=target["boxes"],
        target_labels=target["labels"],
    )

    record = {
        "image_id": int(target["image_id"]),
        "image_path": str(dataset.samples[sample_index].image_path),
        "proposal_boxes": selected_proposal_boxes.detach().cpu(),
        "transformed_proposal_boxes": teacher_output["transformed_proposal_boxes"][keep].detach().cpu(),
        "pooled_features": teacher_output["pooled_features"][keep].detach().to(dtype=torch_dtype).cpu(),
        "teacher_labels": teacher_output["teacher_labels"][keep].detach().cpu(),
        "teacher_logits": teacher_output["teacher_logits"][keep].detach().to(dtype=torch_dtype).cpu(),
        "teacher_scores": teacher_output["teacher_scores"][keep].detach().to(dtype=torch_dtype).cpu(),
        "image_size": teacher_output["image_size"].detach().cpu(),
        "transformed_image_size": teacher_output["transformed_image_size"].detach().cpu(),
        "matched_gt_boxes": matched_ground_truth["matched_gt_boxes"].detach().cpu(),
        "has_matched_gt": matched_ground_truth["has_matched_gt"].detach().cpu(),
    }

    if "gt_labels" in teacher_output:
        record["gt_labels"] = teacher_output["gt_labels"][keep].detach().cpu()
    else:
        record["gt_labels"] = matched_ground_truth["matched_gt_labels"].detach().cpu()

    if "gt_iou" in teacher_output:
        record["gt_iou"] = teacher_output["gt_iou"][keep].detach().to(dtype=torch_dtype).cpu()
    else:
        record["gt_iou"] = matched_ground_truth["matched_gt_iou"].detach().to(dtype=torch_dtype).cpu()

    return record


def _new_metadata_parts() -> dict[str, list[Any]]:
    metadata_parts = {field: [] for field in _METADATA_TENSOR_FIELDS}
    metadata_parts["image_paths"] = []
    return metadata_parts


def _append_repeated_image_metadata(
    record: dict[str, Any],
    metadata_parts: dict[str, list[Any]],
) -> None:
    num_rois = int(record["teacher_labels"].shape[0])
    metadata_parts["image_ids"].append(
        torch.full((num_rois,), int(record["image_id"]), dtype=torch.int64)
    )
    metadata_parts["image_sizes"].append(record["image_size"].repeat(num_rois, 1))
    metadata_parts["transformed_image_sizes"].append(
        record["transformed_image_size"].repeat(num_rois, 1)
    )
    metadata_parts["image_paths"].extend([str(record["image_path"])] * num_rois)


def _append_symbolic_metadata(
    record: dict[str, Any],
    metadata_parts: dict[str, list[Any]],
) -> None:
    metadata_parts["teacher_labels"].append(record["teacher_labels"])
    metadata_parts["teacher_logits"].append(record["teacher_logits"])
    metadata_parts["teacher_scores"].append(record["teacher_scores"])
    metadata_parts["proposal_boxes"].append(record["proposal_boxes"])
    metadata_parts["transformed_proposal_boxes"].append(record["transformed_proposal_boxes"])
    metadata_parts["matched_gt_boxes"].append(record["matched_gt_boxes"])
    metadata_parts["has_matched_gt"].append(record["has_matched_gt"])
    metadata_parts["gt_labels"].append(record["gt_labels"])
    metadata_parts["gt_iou"].append(record["gt_iou"])
    _append_repeated_image_metadata(record, metadata_parts)


def _finalize_symbolic_metadata(metadata_parts: dict[str, list[Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in _METADATA_TENSOR_FIELDS:
        parts = metadata_parts[field]
        metadata[field] = torch.cat(parts, dim=0) if parts else None
    metadata["image_paths"] = tuple(str(path) for path in metadata_parts["image_paths"])
    return metadata


@torch.inference_mode()
def extract_dataset_from_teacher(
    model: torch.nn.Module,
    dataset: Any,
    output_path: str | Path,
    device: str | None = None,
    max_positive_rois_per_image: int = 24,
    max_background_rois_per_image: int = 40,
    storage_dtype: str = "float16",
) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Skipping teacher dataset extraction; found existing artifact at {output_path}.")
        return output_path

    resolved_device = select_device(device)
    storage_dir = ensure_dir(output_path.with_suffix(""))
    feature_path = storage_dir / "features.dat"
    metadata_path = storage_dir / "metadata.pt"

    model = model.to(resolved_device)
    model.eval()

    manifest_records: list[dict[str, Any]] = []
    metadata_parts = _new_metadata_parts()
    torch_dtype, numpy_dtype = _storage_dtypes(storage_dtype)
    max_rois = len(dataset) * (max_positive_rois_per_image + max_background_rois_per_image)
    feature_memmap: np.memmap[Any, Any] | None = None
    feature_shape: tuple[int, int, int] | None = None
    row_count = 0

    for index in tqdm(range(len(dataset)), desc="Export symbolic teacher RoIs", leave=False):
        image, target = dataset[index]
        image = image.to(resolved_device)
        target_on_device = {key: value.to(resolved_device) for key, value in target.items()}
        teacher_output = model.extract_teacher_roi_samples([image], [target_on_device])[0]

        keep = _select_teacher_roi_indices(
            teacher_labels=teacher_output["teacher_labels"],
            teacher_scores=teacher_output["teacher_scores"],
            max_positive_rois_per_image=max_positive_rois_per_image,
            max_background_rois_per_image=max_background_rois_per_image,
        )

        record = _build_symbolic_record(
            dataset=dataset,
            sample_index=index,
            target=target,
            teacher_output=teacher_output,
            keep=keep,
            torch_dtype=torch_dtype,
        )
        num_rois = int(record["teacher_labels"].shape[0])
        if num_rois > 0 and feature_memmap is None:
            feature_shape = tuple(int(value) for value in record["pooled_features"].shape[1:])
            feature_memmap = np.memmap(
                feature_path,
                dtype=numpy_dtype,
                mode="w+",
                shape=(max_rois, *feature_shape),
            )

        row_start = row_count
        row_stop = row_start + num_rois
        if num_rois > 0:
            if feature_memmap is None:
                raise RuntimeError("Feature memmap was not initialized.")
            feature_memmap[row_start:row_stop] = record["pooled_features"].numpy()
            _append_symbolic_metadata(record, metadata_parts)
            row_count = row_stop

        label_counts = torch.bincount(
            record["teacher_labels"],
            minlength=len(dataset.class_names) + 1,
        )
        manifest_records.append(
            {
                "image_id": int(record["image_id"]),
                "image_path": record["image_path"],
                "num_rois": int(record["teacher_labels"].shape[0]),
                "label_counts": label_counts.tolist(),
                "row_start": row_start,
                "row_stop": row_stop,
            }
        )

    if feature_memmap is not None:
        feature_memmap.flush()
        del feature_memmap
        if feature_shape is None:
            raise RuntimeError("Feature shape was not recorded.")
        _truncate_feature_storage(feature_path, row_count, feature_shape, numpy_dtype)

    metadata = _finalize_symbolic_metadata(metadata_parts)
    torch.save(metadata, metadata_path)

    payload = {
        "storage_format": "array_memmap_v1",
        "feature_cut": "roi_align_pooled_grid",
        "symbolic_feature": "pooled_features",
        "symbolic_target": "teacher_label",
        "proposal_source": "rpn_pre_detector_postprocess",
        "class_names": ("__background__", *tuple(dataset.class_names)),
        "feature_shape": feature_shape,
        "roi_sampling": {
            "max_positive_rois_per_image": max_positive_rois_per_image,
            "max_background_rois_per_image": max_background_rois_per_image,
        },
        "storage_dtype": storage_dtype,
        "storage_dir": str(storage_dir),
        "metadata_path": metadata_path.as_posix(),
        "feature_storage": {
            "path": feature_path.as_posix(),
            "dtype": storage_dtype,
            "shape": (row_count, *feature_shape) if feature_shape is not None else (0,),
        },
        "records": manifest_records,
    }
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)
    return output_path
