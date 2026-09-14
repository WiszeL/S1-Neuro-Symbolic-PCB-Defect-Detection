from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io import ensure_dir


@dataclass(frozen=True)
class RunArtifacts:
    run_name: str
    checkpoint_path: Path
    metrics_path: Path
    history_path: Path


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
