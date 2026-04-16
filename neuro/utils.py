from __future__ import annotations

from typing import Any, Iterable

import torch


def detection_collate_fn(batch: list[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def move_targets_to_device(
    targets: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    moved_targets: list[dict[str, Any]] = []
    for target in targets:
        moved_target: dict[str, Any] = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if torch.is_tensor(value) else value
        moved_targets.append(moved_target)
    return moved_targets


def mean_dict(history: Iterable[dict[str, float]]) -> dict[str, float]:
    history = list(history)
    if not history:
        return {}

    keys = history[0].keys()
    return {
        key: float(sum(item[key] for item in history) / len(history)) for key in keys
    }
