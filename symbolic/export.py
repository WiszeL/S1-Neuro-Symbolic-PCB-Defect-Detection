from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from util.device import select_device
from util.io import ensure_dir


_METADATA_TENSOR_FIELDS = [
    "teacher_labels",
    "teacher_scores",
    "proposal_boxes",
    "transformed_proposal_boxes",
    "matched_gt_boxes",
    "has_matched_gt",
    "gt_labels",
    "gt_iou",
]


def _storage_dtypes(storage_dtype: str) -> tuple[torch.dtype, np.dtype[Any]]:
    if storage_dtype == "float16":
        return torch.float16, np.dtype("float16")
    if storage_dtype == "float32":
        return torch.float32, np.dtype("float32")
    raise ValueError("storage_dtype must be either 'float16' or 'float32'.")


def _match_proposals_to_ground_truth(
    proposal_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_boxes = target_boxes.to(
        device=proposal_boxes.device, dtype=proposal_boxes.dtype
    )
    target_labels = target_labels.to(device=proposal_boxes.device, dtype=torch.int64)

    if proposal_boxes.numel() == 0:
        return {
            "matched_gt_boxes": torch.zeros(
                (0, 4), dtype=torch.float32, device=proposal_boxes.device
            ),
            "matched_gt_labels": torch.zeros(
                (0,), dtype=torch.int64, device=proposal_boxes.device
            ),
            "matched_gt_iou": torch.zeros(
                (0,), dtype=torch.float32, device=proposal_boxes.device
            ),
            "has_matched_gt": torch.zeros(
                (0,), dtype=torch.bool, device=proposal_boxes.device
            ),
        }

    if target_boxes.numel() == 0:
        num_boxes = proposal_boxes.shape[0]
        return {
            "matched_gt_boxes": torch.zeros(
                (num_boxes, 4), dtype=torch.float32, device=proposal_boxes.device
            ),
            "matched_gt_labels": torch.zeros(
                (num_boxes,), dtype=torch.int64, device=proposal_boxes.device
            ),
            "matched_gt_iou": torch.zeros(
                (num_boxes,), dtype=torch.float32, device=proposal_boxes.device
            ),
            "has_matched_gt": torch.zeros(
                (num_boxes,), dtype=torch.bool, device=proposal_boxes.device
            ),
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
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    num_rois = teacher_output["teacher_labels"].shape[0]
    proposal_boxes = teacher_output["proposal_boxes"]
    matched_ground_truth = _match_proposals_to_ground_truth(
        proposal_boxes=proposal_boxes,
        target_boxes=target["boxes"],
        target_labels=target["labels"],
    )

    record: dict[str, Any] = {
        "image_id": int(target["image_id"]),
        "image_path": str(dataset.samples[sample_index].image_path),
        "num_rois": num_rois,
        "proposal_boxes": proposal_boxes.detach().cpu(),
        "transformed_proposal_boxes": teacher_output["transformed_proposal_boxes"]
        .detach()
        .cpu(),
        "pooled_features": teacher_output["pooled_features"]
        .detach()
        .to(dtype=torch_dtype)
        .cpu(),
        "teacher_labels": teacher_output["teacher_labels"].detach().cpu(),
        "teacher_logits": teacher_output["teacher_logits"]
        .detach()
        .to(dtype=torch_dtype)
        .cpu(),
        "teacher_scores": teacher_output["teacher_scores"]
        .detach()
        .to(dtype=torch_dtype)
        .cpu(),
        "matched_gt_boxes": matched_ground_truth["matched_gt_boxes"].detach().cpu(),
        "has_matched_gt": matched_ground_truth["has_matched_gt"].detach().cpu(),
        "gt_labels": matched_ground_truth["matched_gt_labels"].detach().cpu(),
        "gt_iou": matched_ground_truth["matched_gt_iou"]
        .detach()
        .to(dtype=torch_dtype)
        .cpu(),
    }

    return record


def _new_metadata_parts() -> dict[str, list[Any]]:
    metadata_parts = {field: [] for field in _METADATA_TENSOR_FIELDS}
    metadata_parts["image_paths"] = []
    return metadata_parts


def _append_symbolic_metadata(
    record: dict[str, Any],
    metadata_parts: dict[str, list[Any]],
) -> None:
    metadata_parts["teacher_labels"].append(record["teacher_labels"])
    metadata_parts["teacher_scores"].append(record["teacher_scores"])
    metadata_parts["proposal_boxes"].append(record["proposal_boxes"])
    metadata_parts["transformed_proposal_boxes"].append(
        record["transformed_proposal_boxes"]
    )
    metadata_parts["matched_gt_boxes"].append(record["matched_gt_boxes"])
    metadata_parts["has_matched_gt"].append(record["has_matched_gt"])
    metadata_parts["gt_labels"].append(record["gt_labels"])
    metadata_parts["gt_iou"].append(record["gt_iou"])
    num_rois = record["teacher_labels"].shape[0]
    metadata_parts["image_paths"].extend([str(record["image_path"])] * num_rois)


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
    *,
    storage_dtype: str,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        print(
            f"Skipping teacher dataset extraction; found existing artifact at {output_path}."
        )
        return output_path

    resolved_device = select_device(device)
    storage_dir = ensure_dir(output_path.with_suffix(""))
    feature_path = storage_dir / "features.dat"
    metadata_path = storage_dir / "metadata.pt"
    if feature_path.exists():
        feature_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    model = model.to(resolved_device)
    model.eval()

    manifest_records: list[dict[str, Any]] = []
    metadata_parts = _new_metadata_parts()
    torch_dtype, numpy_dtype = _storage_dtypes(storage_dtype)
    feature_shape: tuple[int, int, int] | None = None
    row_count = 0

    for index in tqdm(
        range(len(dataset)), desc="Export symbolic teacher RoIs", leave=False
    ):
        image, target = dataset[index]
        image = image.to(resolved_device)
        target_on_device = {
            key: value.to(resolved_device) for key, value in target.items()
        }
        teacher_output = model.extract_teacher_roi_samples([image], [target_on_device])[
            0
        ]

        record = _build_symbolic_record(
            dataset=dataset,
            sample_index=index,
            target=target,
            teacher_output=teacher_output,
            torch_dtype=torch_dtype,
        )
        num_rois = record["teacher_labels"].shape[0]

        row_start = row_count
        row_stop = row_start + num_rois
        if num_rois > 0:
            record_feature_shape = tuple(
                int(value) for value in record["pooled_features"].shape[1:]
            )
            if feature_shape is None:
                feature_shape = record_feature_shape
            elif feature_shape != record_feature_shape:
                raise RuntimeError(
                    f"Inconsistent RoI feature shape: expected {feature_shape}, got {record_feature_shape}."
                )
            feature_array = (
                record["pooled_features"].numpy().astype(numpy_dtype, copy=False)
            )
            with feature_path.open("ab") as feature_file:
                feature_array.tofile(feature_file)
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
                "num_rois": record["teacher_labels"].shape[0],
                "label_counts": label_counts.tolist(),
                "row_start": row_start,
                "row_stop": row_stop,
            }
        )

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


def summarize_symbolic_export(export_path: Path | str, split_name: str) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    manifest = torch.load(export_path, map_location="cpu", weights_only=True)
    records = pd.DataFrame(manifest["records"])

    if records.empty:
        overview = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "images": 0,
                    "total_rois": 0,
                    "background_rois": 0,
                    "positive_rois": 0,
                    "feature_shape": None,
                    "feature_cut": manifest.get("feature_cut"),
                    "symbolic_target": manifest.get("symbolic_target"),
                    "proposal_source": manifest.get("proposal_source"),
                }
            ]
        )
        class_counts = pd.DataFrame(columns=["split", "class_name", "count"])
        return manifest, overview, class_counts

    class_names = list(manifest["class_names"])
    label_counts = records["label_counts"].apply(pd.Series).fillna(0).sum(axis=0)
    label_counts = label_counts.reindex(range(len(class_names)), fill_value=0).astype(
        int
    )

    feature_shape = manifest.get("feature_shape")
    overview = pd.DataFrame(
        [
            {
                "split": split_name,
                "images": len(records),
                "total_rois": int(records["num_rois"].sum()),
                "background_rois": int(label_counts.iloc[0]),
                "positive_rois": int(label_counts.iloc[1:].sum()),
                "feature_shape": "x".join(str(value) for value in feature_shape)
                if feature_shape
                else None,
                "feature_cut": manifest.get("feature_cut"),
                "symbolic_target": manifest.get("symbolic_target"),
                "proposal_source": manifest.get("proposal_source"),
            }
        ]
    )
    class_counts = pd.DataFrame(
        {
            "split": split_name,
            "class_name": class_names,
            "count": label_counts.tolist(),
        }
    )
    return manifest, overview, class_counts
