from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from util.device import select_device
from util.io import ensure_dir


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
    shard_dir = ensure_dir(output_path.with_suffix(""))

    model = model.to(resolved_device)
    model.eval()

    manifest_records: list[dict[str, Any]] = []
    dtype = torch.float16 if storage_dtype == "float16" else torch.float32
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

        selected_proposal_boxes = teacher_output["proposal_boxes"][keep]
        matched_ground_truth = _match_proposals_to_ground_truth(
            proposal_boxes=selected_proposal_boxes,
            target_boxes=target["boxes"],
            target_labels=target["labels"],
        )

        record = {
            "image_id": int(target["image_id"]),
            "image_path": str(dataset.samples[index].image_path),
            "proposal_boxes": selected_proposal_boxes.detach().cpu(),
            "transformed_proposal_boxes": teacher_output["transformed_proposal_boxes"][keep].detach().cpu(),
            "pooled_features": teacher_output["pooled_features"][keep].detach().to(dtype=dtype).cpu(),
            "teacher_labels": teacher_output["teacher_labels"][keep].detach().cpu(),
            "teacher_logits": teacher_output["teacher_logits"][keep].detach().to(dtype=dtype).cpu(),
            "teacher_scores": teacher_output["teacher_scores"][keep].detach().to(dtype=dtype).cpu(),
            "image_size": teacher_output["image_size"].detach().cpu(),
            "transformed_image_size": teacher_output["transformed_image_size"].detach().cpu(),
            "matched_gt_boxes": matched_ground_truth["matched_gt_boxes"].detach().cpu(),
            "has_matched_gt": matched_ground_truth["has_matched_gt"].detach().cpu(),
        }

        if "gt_labels" in teacher_output:
            record["gt_labels"] = teacher_output["gt_labels"][keep].detach().cpu()
        if "gt_iou" in teacher_output:
            record["gt_iou"] = teacher_output["gt_iou"][keep].detach().to(dtype=dtype).cpu()
        else:
            record["gt_iou"] = matched_ground_truth["matched_gt_iou"].detach().to(dtype=dtype).cpu()
        if "gt_labels" not in record:
            record["gt_labels"] = matched_ground_truth["matched_gt_labels"].detach().cpu()

        shard_path = shard_dir / f"image_{index:05d}.pt"
        torch.save(record, shard_path)
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
                "shard_path": shard_path.as_posix(),
            }
        )

    payload = {
        "storage_format": "sharded_pt_v1",
        "feature_cut": "roi_align_pooled_grid",
        "symbolic_feature": "pooled_features",
        "symbolic_target": "teacher_label",
        "proposal_source": "rpn_pre_detector_postprocess",
        "class_names": ("__background__", *tuple(dataset.class_names)),
        "feature_shape": tuple(int(value) for value in record["pooled_features"].shape[1:]) if manifest_records else None,
        "roi_sampling": {
            "max_positive_rois_per_image": max_positive_rois_per_image,
            "max_background_rois_per_image": max_background_rois_per_image,
        },
        "storage_dtype": storage_dtype,
        "shard_dir": str(shard_dir),
        "records": manifest_records,
    }
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)
    return output_path
