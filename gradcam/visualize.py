"""Visualization utilities for Grad-CAM baseline comparison.

Provides functions to:
1. Run Grad-CAM on Faster R-CNN detections and produce per-RoI heatmaps.
2. Plot side-by-side comparisons: Grad-CAM vs SODT local evidence heatmaps.
3. Run the full baseline pipeline from checkpoint + test images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch import Tensor

from neuro.inference import load_checkpoint_model
from util.visualization import image_to_array

from .gradcam import GradCAM

# A perceptually uniform "fire" colormap for heatmap overlays.
_FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire",
    [
        (0.0, "#000000"),
        (0.3, "#8B0000"),
        (0.6, "#FF4500"),
        (0.85, "#FFA500"),
        (1.0, "#FFFF00"),
    ],
)


def gradcam_for_detections(
    gradcam: GradCAM,
    image: Tensor,
    prediction: dict[str, Tensor],
    score_threshold: float = 0.3,
    output_size: tuple[int, int] = (7, 7),
) -> list[dict[str, Any]]:
    """Generate Grad-CAM heatmaps for all detections above the score threshold.

    Parameters
    ----------
    gradcam:
        Initialized GradCAM instance.
    image:
        Image tensor ``[3, H, W]`` in ``[0, 1]`` range.
    prediction:
        Faster R-CNN prediction dict with ``boxes``, ``labels``, ``scores``.
    score_threshold:
        Minimum detection score to generate a heatmap for.
    output_size:
        Spatial size of each output heatmap.

    Returns
    -------
    list[dict]
        Each dict contains ``box``, ``label``, ``score``, and ``heatmap``.
    """
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    # Filter by score.
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if boxes.shape[0] == 0:
        return []

    heatmaps = gradcam.generate(image, boxes, labels, output_size=output_size)

    results: list[dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        results.append(
            {
                "box": boxes[i].detach().cpu(),
                "label": int(labels[i]),
                "score": float(scores[i]),
                "heatmap": heatmaps[i],
            }
        )

    return results


def plot_gradcam_comparison(
    image: Tensor | Image.Image,
    gradcam_results: list[dict[str, Any]],
    class_names: tuple[str, ...],
    sodt_heatmaps: list[Tensor] | None = None,
    max_detections: int = 6,
    figsize_per_detection: tuple[float, float] = (5.5, 4.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Grad-CAM heatmaps (and optionally SODT heatmaps) for each detection.

    Parameters
    ----------
    image:
        Original image (PIL or ``[3, H, W]`` tensor).
    gradcam_results:
        Output from :func:`gradcam_for_detections`.
    class_names:
        Tuple of defect class names (excluding background).
    sodt_heatmaps:
        Optional SODT local evidence heatmaps, one per detection, matching
        the ordering of ``gradcam_results``.
    max_detections:
        Maximum number of detections to plot.
    figsize_per_detection:
        Figure width/height multiplier per row.
    save_path:
        If provided, save the figure to this path.
    """
    if isinstance(image, Tensor):
        image_np = image_to_array(image)
    else:
        image_np = np.array(image).astype(np.float32) / 255.0

    n = min(len(gradcam_results), max_detections)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image_np)
        ax.set_title("No detections above threshold", fontsize=12)
        ax.axis("off")
        return fig

    has_sodt = sodt_heatmaps is not None and len(sodt_heatmaps) >= n
    columns = 3 if has_sodt else 2  # image+box | gradcam | (sodt)

    fig, axes = plt.subplots(
        n,
        columns,
        figsize=(figsize_per_detection[0] * columns, figsize_per_detection[1] * n),
        squeeze=False,
    )
    fig.suptitle(
        "Grad-CAM Baseline" + (" vs SODT" if has_sodt else ""),
        fontsize=16,
        fontweight="bold",
        x=0.01,
        y=0.995,
        ha="left",
    )

    palette = [
        "#d62828",
        "#0077b6",
        "#2a9d8f",
        "#f4a261",
        "#6a4c93",
        "#5f6f52",
    ]

    for row_idx in range(n):
        result = gradcam_results[row_idx]
        box = result["box"]
        label = result["label"]
        score = result["score"]
        heatmap = result["heatmap"]

        class_name = (
            class_names[label - 1]
            if 0 < label <= len(class_names)
            else f"class {label}"
        )
        color = palette[(label - 1) % len(palette)]

        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        # --- Column 0: Detection on image ---
        ax_det = axes[row_idx][0]
        ax_det.imshow(image_np)
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2.0, edgecolor=color, facecolor="none"
        )
        ax_det.add_patch(rect)
        ax_det.set_title(
            f"{class_name} ({score:.2f})",
            fontsize=11,
            loc="left",
            color=color,
            fontweight="bold",
        )
        ax_det.axis("off")

        # --- Column 1: Grad-CAM heatmap cropped to detection ---
        ax_cam = axes[row_idx][1]
        # Crop the original image to the detection region.
        crop_x1 = max(0, int(x1))
        crop_y1 = max(0, int(y1))
        crop_x2 = min(image_np.shape[1], int(x2))
        crop_y2 = min(image_np.shape[0], int(y2))
        crop = image_np[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            ax_cam.set_title("Grad-CAM (empty crop)", fontsize=10)
            ax_cam.axis("off")
            continue

        ax_cam.imshow(crop)
        # Resize heatmap to match the crop.
        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_resized = (
            np.array(
                Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(
                    (crop.shape[1], crop.shape[0]),
                    Image.BILINEAR,
                )
            ).astype(np.float32)
            / 255.0
        )
        ax_cam.imshow(heatmap_resized, cmap=_FIRE_CMAP, alpha=0.55, vmin=0, vmax=1)
        ax_cam.set_title("Grad-CAM", fontsize=11, loc="left", fontweight="bold")
        ax_cam.axis("off")

        # --- Column 2: SODT heatmap (if available) ---
        if has_sodt:
            ax_sodt = axes[row_idx][2]
            sodt_hm = sodt_heatmaps[row_idx].detach().cpu().numpy()
            sodt_resized = (
                np.array(
                    Image.fromarray((sodt_hm * 255).astype(np.uint8)).resize(
                        (crop.shape[1], crop.shape[0]),
                        Image.BILINEAR,
                    )
                ).astype(np.float32)
                / 255.0
            )
            ax_sodt.imshow(crop)
            ax_sodt.imshow(sodt_resized, cmap=_FIRE_CMAP, alpha=0.55, vmin=0, vmax=1)
            ax_sodt.set_title(
                "SODT Evidence", fontsize=11, loc="left", fontweight="bold"
            )
            ax_sodt.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def run_gradcam_baseline(
    checkpoint_path: str | Path,
    model_config_path: str | Path = "neuro.yaml",
    train_config_path: str | Path = "neuro_train.yaml",
    image_paths: list[str | Path] | None = None,
    score_threshold: float = 0.3,
    output_size: tuple[int, int] = (7, 7),
    device: str | None = None,
    save_dir: str | Path | None = None,
) -> list[list[dict[str, Any]]]:
    """End-to-end Grad-CAM baseline: load model, run on images, plot results.

    Parameters
    ----------
    checkpoint_path:
        Path to the Faster R-CNN checkpoint (``.pt``).
    model_config_path:
        Path to ``neuro.yaml`` (relative to ``configs/``).
    train_config_path:
        Path to ``neuro_train.yaml`` (relative to ``configs/``).
    image_paths:
        List of image file paths to process.
    score_threshold:
        Minimum detection score.
    output_size:
        Per-RoI heatmap spatial size.
    device:
        Compute device.
    save_dir:
        If provided, save figures here.

    Returns
    -------
    list[list[dict]]
        Grad-CAM results per image.
    """
    from neuro.preprocess_dataset import test_preprocess
    from util.config import load_yaml
    from neuro.config import NeuroTrainConfig

    model, _ = load_checkpoint_model(
        checkpoint_path, model_config_path, train_config_path, device
    )
    train_config = load_yaml(train_config_path, NeuroTrainConfig)
    class_names = tuple(train_config["dataset"]["class_names"])
    preprocess = test_preprocess()

    gradcam = GradCAM(model, device=str(next(model.parameters()).device))

    all_results: list[list[dict[str, Any]]] = []
    if image_paths is None:
        image_paths = []

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        pil_image = Image.open(str(img_path)).convert("RGB")
        image_tensor = preprocess(pil_image, {})[0]

        # Run standard Faster R-CNN inference (no grad) for detection boxes.
        with torch.inference_mode():
            predictions = model([image_tensor.to(next(model.parameters()).device)])
            prediction = {k: v.detach().cpu() for k, v in predictions[0].items()}

        # Generate Grad-CAM heatmaps.
        results = gradcam_for_detections(
            gradcam, image_tensor, prediction, score_threshold, output_size
        )
        all_results.append(results)

        # Plot.
        fig = plot_gradcam_comparison(
            image_tensor,
            results,
            class_names,
            save_path=(
                Path(save_dir) / f"{Path(img_path).stem}_gradcam.png"
                if save_dir
                else None
            ),
        )
        plt.close(fig)

    gradcam.release()
    return all_results
