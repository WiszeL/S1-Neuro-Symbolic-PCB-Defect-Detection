from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from .datasets import DeepPCBDataset
from .detector import build_detector
from .transforms import build_eval_transforms, build_train_transforms
from .utils import (
    detection_collate_fn,
    mean_dict,
    move_targets_to_device,
    next_run_artifacts,
    save_json,
    seed_everything,
    select_device,
)


def count_detection_matches(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[int, int, int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    prediction_mask = prediction["scores"] >= score_threshold
    filtered_prediction = {
        "boxes": prediction["boxes"][prediction_mask],
        "scores": prediction["scores"][prediction_mask],
        "labels": prediction["labels"][prediction_mask],
    }

    labels = torch.unique(torch.cat([filtered_prediction["labels"], target["labels"]], dim=0))

    for label in labels.tolist():
        predicted_indices = torch.where(filtered_prediction["labels"] == label)[0]
        target_indices = torch.where(target["labels"] == label)[0]

        predicted_boxes = filtered_prediction["boxes"][predicted_indices]
        target_boxes = target["boxes"][target_indices]

        if predicted_boxes.numel() == 0:
            false_negatives += target_boxes.shape[0]
            continue

        if target_boxes.numel() == 0:
            false_positives += predicted_boxes.shape[0]
            continue

        prediction_scores = filtered_prediction["scores"][predicted_indices]
        order = torch.argsort(prediction_scores, descending=True)
        predicted_boxes = predicted_boxes[order]

        ious = box_iou(predicted_boxes, target_boxes)
        matched_targets: set[int] = set()

        for row_index in range(predicted_boxes.shape[0]):
            best_iou, best_target_index = ious[row_index].max(dim=0)
            if best_iou.item() >= iou_threshold and int(best_target_index) not in matched_targets:
                matched_targets.add(int(best_target_index))
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += target_boxes.shape[0] - len(matched_targets)

    return true_positives, false_positives, false_negatives


def set_scheduled_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    epoch_index: int,
    global_step: int,
    warmup_iters: int,
    warmup_ratio: float,
    step_size: int,
    gamma: float,
    milestones: list[int] | None = None,
) -> float:
    if milestones:
        decay_factor = gamma ** sum(epoch_index >= milestone for milestone in milestones)
    else:
        decay_factor = gamma ** (epoch_index // step_size) if step_size > 0 else 1.0

    if global_step < warmup_iters:
        alpha = global_step / max(warmup_iters, 1)
        warmup_factor = warmup_ratio + alpha * (1.0 - warmup_ratio)
    else:
        warmup_factor = 1.0

    current_lr = 0.0
    for parameter_group, base_lr in zip(optimizer.param_groups, base_lrs):
        parameter_group["lr"] = base_lr * decay_factor * warmup_factor
        current_lr = parameter_group["lr"]

    return current_lr


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    epochs: int,
    base_lrs: list[float],
    warmup_iters: int,
    warmup_ratio: float,
    step_size: int,
    gamma: float,
    milestones: list[int] | None,
    amp_enabled: bool,
    grad_accumulation_steps: int,
) -> dict[str, float]:
    model.train()
    batch_history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_index + 1}/{epochs}", leave=False)
    for batch_index, (images, targets) in enumerate(progress_bar):
        global_step = epoch_index * len(data_loader) + batch_index
        current_lr = set_scheduled_learning_rate(
            optimizer=optimizer,
            base_lrs=base_lrs,
            epoch_index=epoch_index,
            global_step=global_step,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            step_size=step_size,
            gamma=gamma,
            milestones=milestones,
        )

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            loss_for_backward = total_loss / grad_accumulation_steps

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Encountered a non-finite loss at epoch {epoch_index + 1}, batch {batch_index + 1}.")

        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (batch_index + 1) % grad_accumulation_steps == 0 or (batch_index + 1) == len(data_loader)
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_summary = {"loss_total": float(total_loss.detach().item()), "lr": float(current_lr)}
        batch_summary.update({name: float(loss.detach().item()) for name, loss in loss_dict.items()})
        batch_history.append(batch_summary)

        progress_bar.set_postfix(
            loss=f"{batch_summary['loss_total']:.4f}",
            cls=f"{batch_summary.get('loss_classifier', 0.0):.4f}",
            reg=f"{batch_summary.get('loss_box_reg', 0.0):.4f}",
            rpn=f"{batch_summary.get('loss_objectness', 0.0):.4f}",
        )

    return mean_dict(batch_history)


@torch.inference_mode()
def evaluate_detector(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...],
    precision_iou: float = 0.5,
    precision_score_threshold: float = 0.5,
    class_metrics: bool = True,
    paper_iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    metric = MeanAveragePrecision(iou_type="bbox", backend="pycocotools", class_metrics=class_metrics)
    paper_metric = MeanAveragePrecision(
        iou_type="bbox",
        backend="pycocotools",
        iou_thresholds=paper_iou_thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    )
    model.eval()

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    progress_bar = tqdm(data_loader, desc="Final test evaluation", leave=False)
    for images, targets in progress_bar:
        images_on_device = [image.to(device) for image in images]
        outputs = model(images_on_device)

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

            true_positives, false_positives, false_negatives = count_detection_matches(
                prediction,
                ground_truth,
                iou_threshold=precision_iou,
                score_threshold=precision_score_threshold,
            )
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

        metric.update(predictions_for_metric, targets_for_metric)
        paper_metric.update(predictions_for_metric, targets_for_metric)

    metric_output = metric.compute()
    paper_metric_output = paper_metric.compute()
    precision = total_true_positives / max(total_true_positives + total_false_positives, 1)
    recall = total_true_positives / max(total_true_positives + total_false_negatives, 1)

    summary: dict[str, Any] = {
        "mAP@0.5:0.95": float(metric_output["map"].item()),
        "mAP@0.5": float(metric_output["map_50"].item()),
        "AP75": float(metric_output["map_75"].item()),
        "AP@50:5:85": float(paper_metric_output["map"].item()),
        "precision": float(precision),
        "recall": float(recall),
        "mar_100": float(metric_output["mar_100"].item()),
        "precision_recall_score_threshold": float(precision_score_threshold),
    }

    if class_metrics and "map_per_class" in metric_output and "classes" in metric_output:
        per_class_map: dict[str, float] = {}
        for class_id, class_ap in zip(metric_output["classes"].tolist(), metric_output["map_per_class"].tolist()):
            per_class_map[class_names[int(class_id) - 1]] = float(class_ap)
        summary["per_class_AP"] = per_class_map

    return summary


def build_optimizer(model: torch.nn.Module, optimizer_config: dict[str, Any]) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return SGD(
        parameters,
        lr=float(optimizer_config["lr"]),
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def run_training(
    model_config: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(int(train_config["seed"]), deterministic=bool(train_config["deterministic"]))
    device = select_device(train_config.get("device"))

    class_names = tuple(model_config["model"]["class_names"]) if "model" in model_config else tuple(model_config["class_names"])
    dataset_root = Path(train_config["dataset"]["root"])

    train_dataset = DeepPCBDataset(
        dataset_root=dataset_root,
        split_file=train_config["dataset"]["train_split"],
        transforms=build_train_transforms(
            horizontal_flip_prob=float(train_config["transforms"]["horizontal_flip_prob"]),
            vertical_flip_prob=float(train_config["transforms"]["vertical_flip_prob"]),
        ),
        class_names=class_names,
    )
    test_dataset = DeepPCBDataset(
        dataset_root=dataset_root,
        split_file=train_config["dataset"]["test_split"],
        transforms=build_eval_transforms(),
        class_names=class_names,
    )

    loader_config = train_config["loader"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_config["train_batch_size"]),
        shuffle=True,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(loader_config["test_batch_size"]),
        shuffle=False,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate_fn,
    )

    model = build_detector(model_config).to(device)
    optimizer = build_optimizer(model, train_config["optimizer"])
    base_lrs = [parameter_group["lr"] for parameter_group in optimizer.param_groups]
    amp_enabled = bool(train_config.get("precision", {}).get("amp", False))
    grad_accumulation_steps = int(train_config["train"].get("grad_accumulation_steps", 1))

    training_history: list[dict[str, float]] = []
    epochs = int(train_config["train"]["epochs"])
    for epoch_index in range(epochs):
        epoch_summary = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch_index,
            epochs=epochs,
            base_lrs=base_lrs,
            warmup_iters=int(train_config["warmup"]["iterations"]),
            warmup_ratio=float(train_config["warmup"]["ratio"]),
            step_size=int(train_config["scheduler"]["step_size"]),
            gamma=float(train_config["scheduler"]["gamma"]),
            milestones=[int(value) for value in train_config["scheduler"].get("milestones", [])],
            amp_enabled=amp_enabled,
            grad_accumulation_steps=grad_accumulation_steps,
        )
        epoch_summary["epoch"] = float(epoch_index + 1)
        training_history.append(epoch_summary)

        print(
            f"Epoch {epoch_index + 1}/{epochs} | "
            f"loss_total={epoch_summary['loss_total']:.4f} | "
            f"loss_classifier={epoch_summary.get('loss_classifier', 0.0):.4f} | "
            f"loss_box_reg={epoch_summary.get('loss_box_reg', 0.0):.4f} | "
            f"loss_objectness={epoch_summary.get('loss_objectness', 0.0):.4f} | "
            f"loss_rpn_box_reg={epoch_summary.get('loss_rpn_box_reg', 0.0):.4f}"
        )

    metrics = evaluate_detector(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=class_names,
        precision_iou=float(train_config["evaluation"]["precision_iou"]),
        precision_score_threshold=float(train_config["evaluation"]["precision_score_threshold"]),
        class_metrics=bool(train_config["evaluation"]["class_metrics"]),
        paper_iou_thresholds=[float(value) for value in train_config["evaluation"]["paper_iou_thresholds"]],
    )

    artifacts = next_run_artifacts(train_config["artifacts"]["checkpoint_dir"])
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": train_config,
        "class_names": class_names,
        "metrics": metrics,
    }
    torch.save(checkpoint_payload, artifacts.checkpoint_path)
    save_json(metrics, artifacts.metrics_path)
    save_json(training_history, artifacts.history_path)

    result = {
        "run_name": artifacts.run_name,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "metrics_path": str(artifacts.metrics_path),
        "history_path": str(artifacts.history_path),
        "metrics": metrics,
        "history": training_history,
    }
    return result
