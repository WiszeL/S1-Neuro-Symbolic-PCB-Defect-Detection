from __future__ import annotations
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from .config import NeuroTrainConfig
from .utils import mean_dict, move_targets_to_device


def set_scheduled_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    epoch_index: int,
    global_step: int,
    train_config: NeuroTrainConfig,
) -> float:
    """Apply warmup plus step/milestone decay for the current train step."""

    train_settings = train_config["train"]
    milestones = train_settings["sch_milestones"]
    gamma = train_settings["sch_gamma"]

    if milestones:
        decay_factor = gamma ** sum(
            epoch_index >= milestone for milestone in milestones
        )
    else:
        step_size = train_settings["sch_step_size"]
        decay_factor = gamma ** (epoch_index // step_size) if step_size > 0 else 1.0

    warmup_iterations = train_settings["warmup_iterations"]
    if global_step < warmup_iterations:
        alpha = global_step / max(warmup_iterations, 1)
        warmup_factor = train_settings["warmup_ratio"] + alpha * (
            1.0 - train_settings["warmup_ratio"]
        )
    else:
        warmup_factor = 1.0

    current_lr = 0.0
    for parameter_group, base_lr in zip(optimizer.param_groups, base_lrs):
        parameter_group["lr"] = base_lr * decay_factor * warmup_factor
        current_lr = float(parameter_group["lr"])

    return current_lr


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    epoch_count: int,
    base_lrs: list[float],
    train_config: NeuroTrainConfig,
) -> dict[str, float]:
    """Train a Faster R-CNN model for exactly one epoch."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_history: list[dict[str, float]] = []
    train_settings = train_config["train"]
    grad_accumulation_steps = train_settings["grad_accumulation_steps"]
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=train_config["amp"] and device.type == "cuda",
    )

    progress = tqdm(
        data_loader,
        desc=f"Epoch {epoch_index + 1}/{epoch_count}",
        leave=True,
        mininterval=1.0,
    )
    for batch_index, (images, targets) in enumerate(progress):
        global_step = epoch_index * len(data_loader) + batch_index
        current_lr = set_scheduled_learning_rate(
            optimizer=optimizer,
            base_lrs=base_lrs,
            epoch_index=epoch_index,
            global_step=global_step,
            train_config=train_config,
        )

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            loss_for_backward = total_loss / grad_accumulation_steps

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch_index + 1}, batch {batch_index + 1}."
            )

        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (batch_index + 1) % grad_accumulation_steps == 0
        should_step = should_step or (batch_index + 1) == len(data_loader)
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_summary = {
            "loss_total": total_loss.detach().item(),
            "lr": current_lr,
        }
        batch_summary.update(
            {name: loss.detach().item() for name, loss in loss_dict.items()}
        )
        batch_history.append(batch_summary)

        progress.set_postfix(
            loss=f"{batch_summary['loss_total']:.4f}",
            cls=f"{batch_summary.get('loss_classifier', 0.0):.4f}",
            reg=f"{batch_summary.get('loss_box_reg', 0.0):.4f}",
            rpn=f"{batch_summary.get('loss_objectness', 0.0):.4f}",
            lr=f"{batch_summary.get('lr', 0.0):.2e}",
        )

    return mean_dict(batch_history)


def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: NeuroTrainConfig,
) -> list[dict[str, float]]:
    """Train for multiple epochs and return notebook-friendly history.

    The notebook still owns model/dataloader creation, checkpointing, and plots.
    This function only owns the repeated training calls and reads training
    hyperparameters from NeuroTrainConfig.
    """

    base_lrs = [
        float(parameter_group["lr"]) for parameter_group in optimizer.param_groups
    ]
    history: list[dict[str, float]] = []
    epoch_count = train_config["train"]["epochs"]

    print(
        "Starting Faster R-CNN training "
        f"on {device} for {epoch_count} epochs "
        f"({len(data_loader)} batches per epoch).",
        flush=True,
    )

    for epoch_index in range(epoch_count):
        epoch_summary = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch_index,
            epoch_count=epoch_count,
            base_lrs=base_lrs,
            train_config=train_config,
        )
        epoch_summary["epoch"] = epoch_index + 1
        history.append(epoch_summary)

    return history


def count_detection_matches(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[int, int, int]:
    """Count simple class-aware TP/FP/FN matches for precision and recall."""

    mask = prediction["scores"] >= score_threshold
    filtered_prediction = {
        "boxes": prediction["boxes"][mask],
        "scores": prediction["scores"][mask],
        "labels": prediction["labels"][mask],
    }

    labels = torch.unique(
        torch.cat([filtered_prediction["labels"], target["labels"]], dim=0)
    )
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for label in labels.tolist():
        prediction_indices = torch.where(filtered_prediction["labels"] == label)[0]
        target_indices = torch.where(target["labels"] == label)[0]

        predicted_boxes = filtered_prediction["boxes"][prediction_indices]
        target_boxes = target["boxes"][target_indices]

        if predicted_boxes.numel() == 0:
            false_negatives += target_boxes.shape[0]
            continue
        if target_boxes.numel() == 0:
            false_positives += predicted_boxes.shape[0]
            continue

        prediction_scores = filtered_prediction["scores"][prediction_indices]
        predicted_boxes = predicted_boxes[
            torch.argsort(prediction_scores, descending=True)
        ]
        ious = box_iou(predicted_boxes, target_boxes)
        matched_targets: set[int] = set()

        for row_index in range(predicted_boxes.shape[0]):
            best_iou, best_target_index = ious[row_index].max(dim=0)
            if (
                best_iou.item() >= iou_threshold
                and int(best_target_index) not in matched_targets
            ):
                matched_targets.add(int(best_target_index))
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += target_boxes.shape[0] - len(matched_targets)

    return true_positives, false_positives, false_negatives


def _confusion_and_per_class(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    iou_threshold: float,
    score_threshold: float,
    num_classes: int,
) -> dict[str, Any]:
    mask = prediction["scores"] >= score_threshold
    filtered = {k: v[mask] for k, v in prediction.items()}

    labels = torch.unique(torch.cat([filtered["labels"], target["labels"]], dim=0))

    conf = torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.int64)
    cp = torch.zeros(num_classes + 1, dtype=torch.int64)
    cfp = torch.zeros(num_classes + 1, dtype=torch.int64)
    cfn = torch.zeros(num_classes + 1, dtype=torch.int64)

    for label in labels.tolist():
        label = int(label)
        pred_idx = torch.where(filtered["labels"] == label)[0]
        tgt_idx = torch.where(target["labels"] == label)[0]
        pred_boxes = filtered["boxes"][pred_idx]
        tgt_boxes = target["boxes"][tgt_idx]

        if pred_boxes.numel() == 0:
            for _ in range(tgt_boxes.shape[0]):
                conf[label, 0] += 1
            cfn[label] = tgt_boxes.shape[0]
            continue

        if tgt_boxes.numel() == 0:
            for _ in range(pred_boxes.shape[0]):
                conf[0, label] += 1
            cfp[label] = pred_boxes.shape[0]
            continue

        scores = filtered["scores"][pred_idx]
        srt = torch.argsort(scores, descending=True)
        pred_boxes = pred_boxes[srt]
        ious = box_iou(pred_boxes, tgt_boxes)
        matched = set()

        for r in range(pred_boxes.shape[0]):
            best_iou, best_tgt = ious[r].max(dim=0)
            if best_iou.item() >= iou_threshold and best_tgt.item() not in matched:
                matched.add(best_tgt.item())
                conf[label, label] += 1
                cp[label] += 1
            else:
                conf[0, label] += 1
                cfp[label] += 1

        fn = tgt_boxes.shape[0] - len(matched)
        conf[label, 0] += fn
        cfn[label] = fn

    return {"confusion": conf, "class_tp": cp, "class_fp": cfp, "class_fn": cfn}


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    train_config: NeuroTrainConfig,
) -> dict[str, Any]:
    """Evaluate Faster R-CNN detections over an existing dataloader."""

    evaluation_config = train_config["evaluation"]
    class_names = train_config["dataset"]["class_names"]

    model.eval()
    coco_metric = MeanAveragePrecision(
        iou_type="bbox",
        backend="pycocotools",
        class_metrics=evaluation_config["class_metrics"],
    )
    paper_metric = MeanAveragePrecision(
        iou_type="bbox",
        backend="pycocotools",
        iou_thresholds=evaluation_config["iou_thresholds"],
    )

    num_classes = len(class_names) + 1
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    class_tp = torch.zeros(num_classes, dtype=torch.int64)
    class_fp = torch.zeros(num_classes, dtype=torch.int64)
    class_fn = torch.zeros(num_classes, dtype=torch.int64)

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    forward_times: list[float] = []

    for images, targets in tqdm(data_loader, desc="Evaluate detector", leave=False):
        t0 = time.perf_counter()
        outputs = model([image.to(device) for image in images])
        forward_times.append(time.perf_counter() - t0)

        predictions_for_metric: list[dict[str, Tensor]] = []
        targets_for_metric: list[dict[str, Tensor]] = []
        for output, target in zip(outputs, targets):
            prediction = {
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            }
            ground_truth = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }

            predictions_for_metric.append(prediction)
            targets_for_metric.append(ground_truth)

            tp, fp, fn = count_detection_matches(
                prediction,
                ground_truth,
                iou_threshold=evaluation_config["precision_iou"],
                score_threshold=evaluation_config["precision_score_threshold"],
            )
            total_true_positives += tp
            total_false_positives += fp
            total_false_negatives += fn

            pc = _confusion_and_per_class(
                prediction,
                ground_truth,
                iou_threshold=evaluation_config["precision_iou"],
                score_threshold=evaluation_config["precision_score_threshold"],
                num_classes=len(class_names),
            )
            conf_matrix += pc["confusion"]
            class_tp += pc["class_tp"]
            class_fp += pc["class_fp"]
            class_fn += pc["class_fn"]

        coco_metric.update(predictions_for_metric, targets_for_metric)
        paper_metric.update(predictions_for_metric, targets_for_metric)

    coco_output = coco_metric.compute()
    paper_output = paper_metric.compute()
    precision = total_true_positives / max(
        total_true_positives + total_false_positives, 1
    )
    recall = total_true_positives / max(total_true_positives + total_false_negatives, 1)

    summary: dict[str, Any] = {
        "mAP@0.5:0.95": coco_output["map"].item(),
        "mAP@0.5": coco_output["map_50"].item(),
        "AP75": coco_output["map_75"].item(),
        "AP@50:5:85": paper_output["map"].item(),
        "precision": precision,
        "recall": recall,
        "mar_100": coco_output["mar_100"].item(),
        "precision_recall_score_threshold": float(
            evaluation_config["precision_score_threshold"]
        ),
        "f1_score": 2 * precision * recall / max(precision + recall, 1e-10),
        "inference_time_ms": float(np.mean(forward_times)) * 1000 if forward_times else 0.0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    all_class_names: list[str] = ["__background__", *class_names]
    for c in range(num_classes):
        name = all_class_names[c]
        per_class_precision[name] = float(class_tp[c] / max(class_tp[c] + class_fp[c], 1))
        per_class_recall[name] = float(class_tp[c] / max(class_tp[c] + class_fn[c], 1))

    summary["confusion_matrix"] = conf_matrix.tolist()
    summary["confusion_matrix_labels"] = all_class_names
    summary["per_class_precision"] = per_class_precision
    summary["per_class_recall"] = per_class_recall

    if (
        evaluation_config["class_metrics"]
        and "map_per_class" in coco_output
        and "classes" in coco_output
    ):
        per_class_ap: dict[str, float] = {}
        for class_id, class_ap in zip(
            coco_output["classes"].tolist(),
            coco_output["map_per_class"].tolist(),
        ):
            per_class_ap[class_names[int(class_id) - 1]] = float(class_ap)
        summary["per_class_AP"] = per_class_ap

    return summary
