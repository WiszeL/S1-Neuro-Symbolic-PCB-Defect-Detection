from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class RunArtifacts:
    run_name: str
    checkpoint_path: Path
    metrics_path: Path
    history_path: Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


def select_device(requested_device: str | None = None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(requested_device)


def next_run_artifacts(checkpoint_dir: str | Path) -> RunArtifacts:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    pattern = re.compile(r"run(\d+)\.pt$")
    run_indices: list[int] = []

    for path in checkpoint_dir.glob("run*.pt"):
        match = pattern.match(path.name)
        if match:
            run_indices.append(int(match.group(1)))

    next_index = max(run_indices, default=0) + 1
    run_name = f"run{next_index}"

    return RunArtifacts(
        run_name=run_name,
        checkpoint_path=checkpoint_dir / f"{run_name}.pt",
        metrics_path=checkpoint_dir / f"{run_name}_metrics.json",
        history_path=checkpoint_dir / f"{run_name}_train_history.json",
    )


def latest_run_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("run*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No run checkpoints were found in {checkpoint_dir}.")

    def run_index(path: Path) -> int:
        match = re.search(r"run(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=run_index)


def save_json(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]

    return value


def mean_dict(history: Iterable[dict[str, float]]) -> dict[str, float]:
    history = list(history)
    if not history:
        return {}

    keys = history[0].keys()
    return {
        key: float(sum(item[key] for item in history) / len(history)) for key in keys
    }
