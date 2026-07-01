from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from util.device import select_device

from util.config import load_yaml
from util.visualization import image_to_array
from .config import NeuroConfig, NeuroTrainConfig
from .faster_rcnn import NeuroFasterRCNN


def load_checkpoint_model(
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    device: str | None = None,
) -> tuple[NeuroFasterRCNN, dict[str, Any]]:
    resolved_device = select_device(device)
    model_config = load_yaml(model_config_path, NeuroConfig)
    train_config = load_yaml(train_config_path, NeuroTrainConfig)
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=True
    )

    model = NeuroFasterRCNN(
        neuro_config=model_config,
        train_config=train_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def run_inference(
    model: NeuroFasterRCNN,
    images: list[Tensor],
    device: str | torch.device,
) -> list[dict[str, Tensor]]:
    resolved_device = select_device(str(device))
    model.eval()
    outputs = model([image.to(resolved_device) for image in images])
    return [
        {key: value.detach().cpu() for key, value in output.items()}
        for output in outputs
    ]


def visualize_prediction(
    image: Tensor | Image.Image,
    prediction: dict[str, Tensor],
    class_names: tuple[str, ...],
    score_threshold: float = 0.3,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image_to_array(image)

    ax.imshow(image_array, cmap="gray" if image_array.shape[-1] == 1 else None)
    ax.axis("off")

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1

        rectangle = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rectangle)
        class_name = class_names[int(label) - 1]
        ax.text(
            x1,
            max(y1 - 4, 0),
            f"{class_name} {float(score):.2f}",
            color="yellow",
            fontsize=9,
            backgroundcolor="black",
        )

    return ax
