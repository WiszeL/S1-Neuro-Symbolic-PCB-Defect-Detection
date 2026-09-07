"""Sanity check: the flattened grid still carries spatial layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_manifest_and_features(
    export_path: str | Path,
) -> tuple[dict[str, Any], np.memmap, dict[str, Any]]:
    """Open the export without loading features into RAM."""
    export_path = Path(export_path)
    manifest: dict[str, Any] = torch.load(
        export_path, map_location="cpu", weights_only=True
    )
    metadata_path = Path(manifest["metadata_path"])
    if not metadata_path.exists():
        storage_dir = Path(manifest.get("storage_dir", metadata_path.parent))
        metadata_path = storage_dir / metadata_path.name
    metadata: dict[str, Any] = torch.load(
        metadata_path, map_location="cpu", weights_only=True
    )

    feature_storage = manifest["feature_storage"]
    feature_path = Path(feature_storage["path"])
    if not feature_path.exists():
        # Paths move with the export dir, so retry relative to it.
        storage_dir = Path(manifest.get("storage_dir", feature_path.parent))
        feature_path = storage_dir / feature_path.name
    features = np.memmap(
        feature_path,
        dtype=feature_storage["dtype"],
        mode="r",
        shape=tuple(feature_storage["shape"]),
    )
    return manifest, features, metadata


def _select_random_samples(
    teacher_labels: torch.Tensor,
    teacher_scores: torch.Tensor,
    num_classes: int,
    num_samples: int,
    top_k_multiplier: int,
    rng: np.random.Generator,
) -> list[list[tuple[int, float]]]:
    """Confident but varied picks per class, for the sanity plot."""
    selections: list[list[tuple[int, float]]] = []
    for cls_idx in range(num_classes):
        mask = teacher_labels == cls_idx
        if mask.sum().item() == 0:
            selections.append([])
            continue

        cls_scores = teacher_scores[:, cls_idx]
        cls_scores_masked = cls_scores.clone()
        cls_scores_masked[~mask] = -1.0

        top_k = min(num_samples * top_k_multiplier, int(mask.sum().item()))
        top_indices = cls_scores_masked.topk(top_k).indices.numpy()

        chosen = rng.choice(
            top_indices, size=min(num_samples, len(top_indices)), replace=False
        )
        selections.append([(int(idx), float(cls_scores[idx].item())) for idx in chosen])
    return selections


def visualize_spatial_topology(
    export_path: str | Path,
    *,
    num_samples: int = 2,
    seed: int = 42,
    top_k_multiplier: int = 10,
) -> plt.Figure:
    """One plot proving the grid keeps spatial layout before flattening."""
    manifest, features, metadata = _load_manifest_and_features(export_path)
    class_names: tuple[str, ...] = tuple(manifest["class_names"])
    num_classes = len(class_names)
    grid_size = int(manifest["feature_shape"][-1])  # 7

    teacher_labels: torch.Tensor = metadata["teacher_labels"]
    teacher_scores: torch.Tensor = metadata["teacher_scores"]

    rng = np.random.default_rng(seed)
    selections = _select_random_samples(
        teacher_labels,
        teacher_scores,
        num_classes=num_classes,
        num_samples=num_samples,
        top_k_multiplier=top_k_multiplier,
        rng=rng,
    )

    # Layout
    fig, axes = plt.subplots(
        num_classes,
        num_samples,
        figsize=(3 * num_samples + 1, 3 * num_classes + 1),
        squeeze=False,
    )
    num_channels = manifest["feature_shape"][0]
    fig.suptitle(
        f"RoI Align Spatial Topology  (C\u2009=\u2009{num_channels} \u2192 mean pool \u2192 {grid_size}\u2009\u00d7\u2009{grid_size})",
        fontsize=14,
        fontweight="bold",
        y=1.0 - 0.01,
    )

    for row, (cls_name, cls_selections) in enumerate(zip(class_names, selections)):
        for col in range(num_samples):
            ax = axes[row, col]

            if col < len(cls_selections):
                roi_idx, score = cls_selections[col]
                spatial_map = features[roi_idx].mean(axis=0)

                ax.imshow(
                    spatial_map,
                    cmap="magma",
                    interpolation="nearest",
                    vmin=float(spatial_map.min()),
                    vmax=float(spatial_map.max()),
                )

                # Cell borders.
                for edge in range(grid_size + 1):
                    ax.axhline(edge - 0.5, color="white", linewidth=0.5)
                    ax.axvline(edge - 0.5, color="white", linewidth=0.5)

                if row == 0:
                    title_text = f"Sample {col + 1}\nscore={score:.4f}"
                else:
                    title_text = f"score={score:.4f}"

                ax.set_title(
                    title_text,
                    fontsize=8,
                    color="white",
                    backgroundcolor="#333333",
                    pad=6,
                )
            else:
                if row == 0:
                    ax.set_title(
                        f"Sample {col + 1}",
                        fontsize=10,
                        fontweight="bold",
                        pad=6,
                    )
                ax.text(
                    0.5,
                    0.5,
                    "no sample",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                    transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Labels
            if col == 0:
                ax.set_ylabel(
                    cls_name,
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=15,
                    ha="right",
                    va="center",
                )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    return fig
