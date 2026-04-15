from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2

from .config import NeuroTrainConfig
from .prepare_dataset import PCBDataset, PCBTarget

PCBPreprocess = Callable[
    [Image.Image, PCBTarget],
    tuple[Tensor, PCBTarget],
]
Palette = list[str]
PREVIEW_PALETTE = [
    "#d62828",
    "#0077b6",
    "#2a9d8f",
    "#f4a261",
    "#6a4c93",
    "#5f6f52",
    "#bc6c25",
    "#3a86ff",
]


def train_preprocess(horizontal_flip_prob: float = 0.5) -> PCBPreprocess:
    return v2.Compose(
        [
            # Pixel standardization: convert PIL image to tensor and scale 0-255 to 0-1.
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # Training augmentation: updates image and BoundingBoxes
            v2.RandomHorizontalFlip(p=horizontal_flip_prob),
        ]
    )


def test_preprocess() -> PCBPreprocess:
    return v2.Compose(
        [
            # Pixel standardization: convert PIL image to tensor and scale 0-255 to 0-1.
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def _image_for_plot(image: Image.Image | Tensor) -> Image.Image | Tensor:
    if isinstance(image, Image.Image):
        return image

    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.clamp(0.0, 1.0)


def _image_size(image: Image.Image | Tensor) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size

    height, width = image.shape[-2:]
    return int(width), int(height)


def _tensor_range(image: Tensor) -> str:
    image = image.detach()
    return f"{float(image.min()):.2f}..{float(image.max()):.2f}"


def _boxes_changed(before: Tensor, after: Tensor) -> bool:
    return before.shape == after.shape and not torch.equal(before.cpu(), after.cpu())


def _draw_target(
    ax,
    image: Image.Image | Tensor,
    target: PCBTarget,
    class_names: tuple[str, ...],
    palette: Palette,
    title: str,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    image_width, image_height = _image_size(image)
    ax.imshow(
        _image_for_plot(image),
        extent=(0, image_width, image_height, 0),
    )
    ax.set_title(title, fontsize=11, loc="left")
    ax.axis("off")

    if canvas_size is not None:
        canvas_width, canvas_height = canvas_size
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(canvas_height, 0)

    for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
        x1, y1, x2, y2 = box
        class_index = int(label) - 1
        color = palette[class_index % len(palette)]
        label_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"class {int(label)}"
        )

        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.4,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x1,
            max(y1 - 3, 0),
            label_name,
            color="white",
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.5},
        )


def visualize_preprocessing(
    dataset: PCBDataset,
    n: int = 3,
    repeats: int = 2,
) -> None:
    if len(dataset.samples) == 0:
        raise ValueError("Cannot visualize preprocessing for an empty dataset.")

    n = min(max(n, 1), len(dataset.samples))
    repeats = max(repeats, 1)
    records = random.sample(dataset.samples, k=n)

    columns = repeats + 1
    fig, axes = plt.subplots(
        n,
        columns,
        figsize=(5.0 * columns, 4.2 * n),
        squeeze=False,
    )
    fig.suptitle(
        "Dataset Preprocessing Preview",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=16,
        weight="bold",
    )

    for row, record in enumerate(records):
        image = Image.open(Path(record.image_path)).convert("RGB")
        area = (record.boxes[:, 2] - record.boxes[:, 0]) * (
            record.boxes[:, 3] - record.boxes[:, 1]
        )
        target: PCBTarget = {
            "boxes": record.boxes,
            "labels": record.labels,
            "image_id": torch.tensor(row, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((record.boxes.shape[0],), dtype=torch.int64),
        }

        _draw_target(
            axes[row][0],
            image,
            target,
            tuple(dataset.class_names),
            PREVIEW_PALETTE,
            f"{record.image_path.stem} | original",
        )

        for column in range(1, columns):
            processed_image, processed_target = dataset.preprocess(image, target)
            flipped = _boxes_changed(target["boxes"], processed_target["boxes"])
            _draw_target(
                axes[row][column],
                processed_image,
                processed_target,
                tuple(dataset.class_names),
                PREVIEW_PALETTE,
                f"preprocess pass {column} | flipped: {'yes' if flipped else 'no'}",
            )

    fig.tight_layout()
    plt.show()


def visualize_train_rcnn_preprocessing(
    rcnn_preprocess: "RCNNPreprocessing",
    dataset: PCBDataset,
    n: int = 3,
    repeats: int = 2,
) -> None:
    if len(dataset.samples) == 0:
        raise ValueError("Cannot visualize RCNN preprocessing for an empty dataset.")

    was_training = rcnn_preprocess.training
    rcnn_preprocess.train()

    try:
        n = min(max(n, 1), len(dataset))
        repeats = max(repeats, 1)
        indices = random.sample(range(len(dataset)), k=n)

        columns = repeats + 1
        fig, axes = plt.subplots(
            n,
            columns,
            figsize=(5.0 * columns, 4.2 * n),
            squeeze=False,
        )
        fig.suptitle(
            "Train RCNN Preprocessing Preview",
            x=0.01,
            y=0.995,
            ha="left",
            fontsize=16,
            weight="bold",
        )

        for row, index in enumerate(indices):
            image, target = dataset[index]
            record = dataset.samples[index]
            rcnn_outputs = []

            for _ in range(repeats):
                image_list, transformed_targets = rcnn_preprocess([image], [target])
                transformed_image = image_list.tensors[0]
                transformed_target = transformed_targets[0]
                image_height, image_width = image_list.image_sizes[0]
                rcnn_outputs.append(
                    (
                        transformed_image,
                        transformed_target,
                        int(image_width),
                        int(image_height),
                        int(transformed_image.shape[-1]),
                        int(transformed_image.shape[-2]),
                    )
                )

            row_canvas_width = max(_image_size(image)[0], *(output[4] for output in rcnn_outputs))
            row_canvas_height = max(_image_size(image)[1], *(output[5] for output in rcnn_outputs))

            _draw_target(
                axes[row][0],
                image,
                target,
                tuple(dataset.class_names),
                PREVIEW_PALETTE,
                f"{record.image_path.stem} | dataset preprocess",
                canvas_size=(row_canvas_width, row_canvas_height),
            )

            for column, output in enumerate(rcnn_outputs, start=1):
                (
                    transformed_image,
                    transformed_target,
                    image_width,
                    image_height,
                    padded_width,
                    padded_height,
                ) = output
                _draw_target(
                    axes[row][column],
                    transformed_image,
                    transformed_target,
                    tuple(dataset.class_names),
                    PREVIEW_PALETTE,
                    (
                        f"RCNN pass {column} | "
                        f"resized {image_width}x{image_height} | "
                        f"padded {padded_width}x{padded_height} | "
                        f"norm {_tensor_range(transformed_image)}"
                    ),
                    canvas_size=(row_canvas_width, row_canvas_height),
                )

        fig.tight_layout()
        plt.show()
    finally:
        rcnn_preprocess.train(was_training)


class RCNNPreprocessing(GeneralizedRCNNTransform):
    def __init__(self, train_config: NeuroTrainConfig) -> None:
        preprocessing_config = train_config["preprocessing"]
        train_min_sizes = tuple(int(size) for size in preprocessing_config["train_min_sizes"])
        eval_min_size = int(preprocessing_config["test_min_size"])

        # Mean/std normalization and max-size limiting are handled by
        # torchvision's GeneralizedRCNNTransform during the model forward pass.
        super().__init__(
            min_size=train_min_sizes,
            max_size=int(preprocessing_config["max_size"]),
            image_mean=list(preprocessing_config["mean"]),
            image_std=list(preprocessing_config["std"]),
        )
        self.train_min_sizes = train_min_sizes
        self.eval_min_size = eval_min_size

    def resize(
        self,
        image: Tensor,
        target: PCBTarget | None = None,
    ) -> tuple[Tensor, PCBTarget | None]:
        original_min_size = self.min_size
        # Multi-scale resizing: training samples one configured min size,
        # while evaluation uses a fixed min size.
        self.min_size = self.train_min_sizes if self.training else (self.eval_min_size,)
        image, target = super().resize(image, target)
        self.min_size = original_min_size

        return image, target
