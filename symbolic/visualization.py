"""RoI Align spatial-topology visualization for the symbolic export.

Visualizes the (C, 7, 7) RoI Align pooled tensors to demonstrate that
spatial geometry is preserved in the feature grid before being flattened
for SODT training.  Channel-wise mean pooling collapses the FPN channels
into a pure (7, 7) spatial activation map per sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_manifest_and_features(
    export_path: str | Path,
) -> tuple[dict[str, Any], np.memmap, dict[str, Any]]:
    """Load the teacher export manifest, metadata, and open features as memmap."""
    export_path = Path(export_path)
    manifest: dict[str, Any] = torch.load(
        export_path, map_location="cpu", weights_only=True
    )
    metadata_path = Path(manifest["metadata_path"])
    metadata: dict[str, Any] = torch.load(
        metadata_path, map_location="cpu", weights_only=True
    )

    feature_storage = manifest["feature_storage"]
    feature_path = Path(feature_storage["path"])
    if not feature_path.exists():
        # Fall back to relative resolution from the manifest's storage dir.
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
    """Pick *num_samples* random high-confidence RoIs per class.

    For each class the top ``num_samples * top_k_multiplier`` scoring RoIs
    (among those the teacher labelled as that class) are shortlisted, then
    *num_samples* are drawn uniformly at random from that pool.

    Returns a list of length *num_classes*, each containing a list of
    ``(roi_index, score)`` tuples.
    """
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

        chosen = rng.choice(top_indices, size=min(num_samples, len(top_indices)), replace=False)
        selections.append(
            [(int(idx), float(cls_scores[idx].item())) for idx in chosen]
        )
    return selections


def visualize_spatial_topology(
    export_path: str | Path,
    *,
    num_samples: int = 2,
    seed: int = 42,
    top_k_multiplier: int = 10,
) -> plt.Figure:
    """Visualize RoI Align spatial topology preservation.

    Mean-pools the 64 FPN channels of the exported ``(C, 7, 7)`` RoI Align
    tensors down to a ``(7, 7)`` spatial activation map.  Random high-confidence
    samples per class are shown to demonstrate that spatial geometry is
    preserved in the feature grid before being flattened for SODT training.

    Parameters
    ----------
    export_path:
        Path to the teacher manifest ``.pt`` file produced by
        :func:`symbolic.export.extract_dataset_from_teacher`.
    num_samples:
        Number of random picks per class (default 2).
    seed:
        Random seed for reproducibility.
    top_k_multiplier:
        The top ``num_samples * top_k_multiplier`` scoring RoIs per class
        are shortlisted before random selection (keeps picks confident but
        varied).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the spatial-topology heatmap grid.
    """
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

    # --- Build figure: one row per class, one column per sample ---
    fig, axes = plt.subplots(
        num_classes,
        num_samples,
        figsize=(3 * num_samples + 1, 3 * num_classes + 1),
        squeeze=False,
    )
    fig.suptitle(
        "RoI Align Spatial Topology  (C\u2009=\u200964 \u2192 mean pool \u2192 7\u2009\u00d7\u20097)",
        fontsize=14,
        fontweight="bold",
        y=1.0 - 0.01,
    )

    for row, (cls_name, cls_selections) in enumerate(
        zip(class_names, selections)
    ):
        for col in range(num_samples):
            ax = axes[row, col]

            if col < len(cls_selections):
                roi_idx, score = cls_selections[col]
                spatial_map = features[roi_idx].mean(axis=0)  # (64,7,7) -> (7,7)

                im = ax.imshow(
                    spatial_map,
                    cmap="magma",
                    interpolation="nearest",
                    vmin=float(spatial_map.min()),
                    vmax=float(spatial_map.max()),
                )

                # White grid lines at every cell boundary.
                for edge in range(grid_size + 1):
                    ax.axhline(edge - 0.5, color="white", linewidth=0.5)
                    ax.axvline(edge - 0.5, color="white", linewidth=0.5)

                ax.set_title(
                    f"score={score:.4f}",
                    fontsize=8,
                    color="white",
                    backgroundcolor="#333333",
                    pad=2,
                )
            else:
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

            # Row label on the first column only.
            if col == 0:
                ax.set_ylabel(
                    cls_name,
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=80,
                    ha="right",
                )

    # Column headers above the first row.
    for col in range(num_samples):
        axes[0, col].annotate(
            f"Sample {col + 1}",
            xy=(0.5, 1.08),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
