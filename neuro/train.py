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
    """Current LR for this step: warmup first, then decay."""

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
    """One training epoch, summarized for the notebook."""

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
    """Repeated epoch calls; the notebook still owns checkpoints and plots."""

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


def _greedy_match_within_label(
    predicted_boxes: Tensor,
    prediction_scores: Tensor,
    target_boxes: Tensor,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Best-score-first matching for one label; each target claimed once.

    Shared by the precision/recall counter and the confusion-matrix builder.
    """
    if predicted_boxes.numel() == 0:
        return 0, 0, target_boxes.shape[0]
    if target_boxes.numel() == 0:
        return 0, predicted_boxes.shape[0], 0

    predicted_boxes = predicted_boxes[
        torch.argsort(prediction_scores, descending=True)
    ]
    ious = box_iou(predicted_boxes, target_boxes)
    matched_targets: set[int] = set()
    true_positives = 0
    false_positives = 0

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

    false_negatives = target_boxes.shape[0] - len(matched_targets)
    return true_positives, false_positives, false_negatives


def count_detection_matches(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[int, int, int]:
    """TP/FP/FN per label, for precision and recall."""

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

        tp, fp, fn = _greedy_match_within_label(
            filtered_prediction["boxes"][prediction_indices],
            filtered_prediction["scores"][prediction_indices],
            target["boxes"][target_indices],
            iou_threshold,
        )
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    return true_positives, false_positives, false_negatives


def _global_confusion_matrix(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    iou_threshold: float,
    score_threshold: float,
    num_classes: int,
) -> Tensor:
    """Confusion matrix where wrong-class matches land off-diagonal.

    Row = truth, column = prediction, 0 = background.
    """
    mask = prediction["scores"] >= score_threshold
    pred_boxes = prediction["boxes"][mask]
    pred_labels = prediction["labels"][mask]
    pred_scores = prediction["scores"][mask]
    tgt_boxes = target["boxes"]
    tgt_labels = target["labels"]

    conf = torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.int64)

    if pred_boxes.numel() == 0:
        for label in tgt_labels.tolist():
            conf[int(label), 0] += 1
        return conf

    if tgt_boxes.numel() == 0:
        for label in pred_labels.tolist():
            conf[0, int(label)] += 1
        return conf

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    ious = box_iou(pred_boxes, tgt_boxes)
    matched_targets: set[int] = set()

    for row in range(pred_boxes.shape[0]):
        best_iou, best_target = ious[row].max(dim=0)
        pred_label = int(pred_labels[row])
        if best_iou.item() >= iou_threshold and int(best_target) not in matched_targets:
            matched_targets.add(int(best_target))
            conf[int(tgt_labels[int(best_target)]), pred_label] += 1
        else:
            conf[0, pred_label] += 1

    for target_index in range(tgt_boxes.shape[0]):
        if target_index not in matched_targets:
            conf[int(tgt_labels[target_index]), 0] += 1

    return conf


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

    cp = torch.zeros(num_classes + 1, dtype=torch.int64)
    cfp = torch.zeros(num_classes + 1, dtype=torch.int64)
    cfn = torch.zeros(num_classes + 1, dtype=torch.int64)

    for label in labels.tolist():
        label = int(label)
        pred_idx = torch.where(filtered["labels"] == label)[0]
        tgt_idx = torch.where(target["labels"] == label)[0]

        tp, fp, fn = _greedy_match_within_label(
            filtered["boxes"][pred_idx],
            filtered["scores"][pred_idx],
            target["boxes"][tgt_idx],
            iou_threshold,
        )
        cp[label] = tp
        cfp[label] = fp
        cfn[label] = fn

    conf = _global_confusion_matrix(
        prediction, target, iou_threshold, score_threshold, num_classes
    )
    return {"confusion": conf, "class_tp": cp, "class_fp": cfp, "class_fn": cfn}


# IoU grid every VOC number below is read from: 0.50, 0.55, ..., 0.95.
VOC_IOU_GRID = [round(0.5 + 0.05 * step, 2) for step in range(10)]


class VOCMeanAP:
    """mmdet `eval_map` (VOC style): greedy one-to-one matching, no fallback GT.

    Labels are 1..num_classes (0 is background). `legacy` adds +1 to box sides.
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: list[float],
        mode: str = "area",
        legacy: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.mode = mode
        self.legacy = legacy
        self.num_gts = np.zeros(num_classes, dtype=np.int64)
        self._scores: list[list[np.ndarray]] = [[] for _ in range(num_classes)]
        self._tp: list[list[np.ndarray]] = [[] for _ in range(num_classes)]
        self._fp: list[list[np.ndarray]] = [[] for _ in range(num_classes)]

    def _iou(self, det_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        extra = 1.0 if self.legacy else 0.0
        det_area = (det_boxes[:, 2] - det_boxes[:, 0] + extra) * (
            det_boxes[:, 3] - det_boxes[:, 1] + extra
        )
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + extra) * (
            gt_boxes[:, 3] - gt_boxes[:, 1] + extra
        )
        top_left = np.maximum(det_boxes[:, None, :2], gt_boxes[None, :, :2])
        bottom_right = np.minimum(det_boxes[:, None, 2:], gt_boxes[None, :, 2:])
        wh = np.clip(bottom_right - top_left + extra, 0, None)
        overlap = wh[..., 0] * wh[..., 1]
        union = np.maximum(det_area[:, None] + gt_area[None, :] - overlap, np.finfo(np.float32).eps)
        return overlap / union

    def update(
        self,
        predictions: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> None:
        for prediction, target in zip(predictions, targets):
            pred_boxes = prediction["boxes"].detach().cpu().numpy().astype(np.float64)
            pred_scores = prediction["scores"].detach().cpu().numpy().astype(np.float64)
            pred_labels = prediction["labels"].detach().cpu().numpy()
            gt_boxes = target["boxes"].detach().cpu().numpy().astype(np.float64)
            gt_labels = target["labels"].detach().cpu().numpy()

            for class_index in range(self.num_classes):
                label = class_index + 1
                class_gt = gt_boxes[gt_labels == label]
                self.num_gts[class_index] += class_gt.shape[0]

                det_mask = pred_labels == label
                det_count = int(det_mask.sum())
                if det_count == 0:
                    continue
                class_det = pred_boxes[det_mask]
                class_scores = pred_scores[det_mask]

                tp = np.zeros((len(self.iou_thresholds), det_count))
                fp = np.zeros_like(tp)
                if class_gt.shape[0] == 0:
                    fp[:] = 1.0
                else:
                    overlaps = self._iou(class_det, class_gt)
                    best_iou = overlaps.max(axis=1)
                    best_gt = overlaps.argmax(axis=1)
                    order = np.argsort(-class_scores, kind="stable")
                    for t, threshold in enumerate(self.iou_thresholds):
                        covered = np.zeros(class_gt.shape[0], dtype=bool)
                        for k in order:
                            if best_iou[k] >= threshold and not covered[best_gt[k]]:
                                covered[best_gt[k]] = True
                                tp[t, k] = 1.0
                            else:
                                fp[t, k] = 1.0

                self._scores[class_index].append(class_scores)
                self._tp[class_index].append(tp)
                self._fp[class_index].append(fp)

    def _average_precision(self, recalls: np.ndarray, precisions: np.ndarray) -> np.ndarray:
        rows = recalls.shape[0]
        if self.mode == "11points":
            ap = np.zeros(rows)
            for t in range(rows):
                for point in np.arange(0.0, 1.1, 0.1):
                    reached = precisions[t][recalls[t] >= point]
                    ap[t] += reached.max() if reached.size else 0.0
            return ap / 11.0

        mrec = np.hstack((np.zeros((rows, 1)), recalls, np.ones((rows, 1))))
        mpre = np.hstack((np.zeros((rows, 1)), precisions, np.zeros((rows, 1))))
        mpre = np.maximum.accumulate(mpre[:, ::-1], axis=1)[:, ::-1]
        ap = np.zeros(rows)
        for t in range(rows):
            changed = np.where(mrec[t, 1:] != mrec[t, :-1])[0]
            ap[t] = np.sum((mrec[t, changed + 1] - mrec[t, changed]) * mpre[t, changed + 1])
        return ap

    def compute(self) -> np.ndarray:
        """AP as [num_thresholds, num_classes]; NaN where a class has no GT."""

        ap = np.full((len(self.iou_thresholds), self.num_classes), np.nan)
        eps = np.finfo(np.float32).eps
        for class_index in range(self.num_classes):
            if self.num_gts[class_index] == 0:
                continue
            if not self._scores[class_index]:
                ap[:, class_index] = 0.0
                continue
            scores = np.concatenate(self._scores[class_index])
            order = np.argsort(-scores, kind="stable")
            tp = np.cumsum(np.concatenate(self._tp[class_index], axis=1)[:, order], axis=1)
            fp = np.cumsum(np.concatenate(self._fp[class_index], axis=1)[:, order], axis=1)
            recalls = tp / np.maximum(self.num_gts[class_index], eps)
            precisions = tp / np.maximum(tp + fp, eps)
            ap[:, class_index] = self._average_precision(recalls, precisions)
        return ap


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    train_config: NeuroTrainConfig,
) -> dict[str, Any]:
    """Detection scores over a notebook-built dataloader."""

    evaluation_config = train_config["evaluation"]
    class_names = train_config["dataset"]["class_names"]

    model.eval()
    # Headline numbers are mmdet-style VOC AP; COCO 101-point is kept as coco_*.
    coco_metric = MeanAveragePrecision(iou_type="bbox", backend="pycocotools")
    voc_metric = VOCMeanAP(len(class_names), VOC_IOU_GRID, mode="area", legacy=True)

    num_classes = len(class_names) + 1
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    class_tp = torch.zeros(num_classes, dtype=torch.int64)
    class_fp = torch.zeros(num_classes, dtype=torch.int64)
    class_fn = torch.zeros(num_classes, dtype=torch.int64)

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
        voc_metric.update(predictions_for_metric, targets_for_metric)

    coco_output = coco_metric.compute()
    voc_ap = voc_metric.compute()
    grid_rows = {threshold: row for row, threshold in enumerate(VOC_IOU_GRID)}
    paper_rows = [grid_rows[round(t, 2)] for t in evaluation_config["iou_thresholds"]]
    total_true_positives = int(class_tp.sum())
    total_false_positives = int(class_fp.sum())
    total_false_negatives = int(class_fn.sum())
    precision = total_true_positives / max(
        total_true_positives + total_false_positives, 1
    )
    recall = total_true_positives / max(total_true_positives + total_false_negatives, 1)

    summary: dict[str, Any] = {
        "mAP@0.5:0.95": float(np.nanmean(voc_ap)),
        "mAP@0.5": float(np.nanmean(voc_ap[grid_rows[0.5]])),
        "AP75": float(np.nanmean(voc_ap[grid_rows[0.75]])),
        "AP@50:5:85": float(np.nanmean(voc_ap[paper_rows])),
        "AP_by_iou": {
            f"AP{round(t * 100)}": float(np.nanmean(voc_ap[row]))
            for t, row in zip(evaluation_config["iou_thresholds"], paper_rows)
        },
        "coco_mAP@0.5": coco_output["map_50"].item(),
        "coco_mAP@0.5:0.95": coco_output["map"].item(),
        "eval_protocol": "voc_area_legacy",
        "precision": precision,
        "recall": recall,
        "mar_100": coco_output["mar_100"].item(),
        "precision_recall_score_threshold": float(
            evaluation_config["precision_score_threshold"]
        ),
        "f1_score": 2 * precision * recall / max(precision + recall, 1e-10),
        "inference_time_ms": float(np.mean(forward_times)) * 1000
        if forward_times
        else 0.0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    all_class_names: list[str] = ["__background__", *class_names]
    for c in range(num_classes):
        name = all_class_names[c]
        per_class_precision[name] = float(
            class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        )
        per_class_recall[name] = float(class_tp[c] / max(class_tp[c] + class_fn[c], 1))

    summary["confusion_matrix"] = conf_matrix.tolist()
    summary["confusion_matrix_labels"] = all_class_names
    summary["per_class_precision"] = per_class_precision
    summary["per_class_recall"] = per_class_recall

    if evaluation_config["class_metrics"]:
        summary["per_class_AP"] = {
            name: float(np.nanmean(voc_ap[:, index]))
            for index, name in enumerate(class_names)
        }
        summary["per_class_AP50"] = {
            name: float(voc_ap[grid_rows[0.5], index])
            for index, name in enumerate(class_names)
        }

    return summary
