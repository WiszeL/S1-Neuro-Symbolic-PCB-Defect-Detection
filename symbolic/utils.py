from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuro.utils import ensure_dir, save_json

BACKGROUND_CLASS_NAME = "__background__"


def build_symbolic_class_names(class_names: tuple[str, ...]) -> tuple[str, ...]:
    return (BACKGROUND_CLASS_NAME, *class_names)


def flatten_feature_grid(feature_grid: torch.Tensor) -> torch.Tensor:
    if feature_grid.ndim == 4:
        return feature_grid.flatten(start_dim=1)
    return feature_grid.reshape(-1)


def load_symbolic_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_symbolic_payload(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)
    return output_path


def save_symbolic_summary(summary: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    save_json(summary, output_path)
    return output_path
