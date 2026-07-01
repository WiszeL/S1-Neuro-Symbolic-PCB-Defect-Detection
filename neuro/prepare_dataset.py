import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors


from .config import NeuroTrainConfig


class PCBTarget(TypedDict):
    boxes: tv_tensors.BoundingBoxes
    labels: Tensor
    image_id: Tensor
    area: Tensor
    iscrowd: Tensor


PCBItem = tuple[Tensor, PCBTarget]
PCBPreprocess = Callable[[Image.Image, PCBTarget], tuple[Tensor, PCBTarget]]
InspectionSummary = dict[str, Any]

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


@dataclass(frozen=True)
class PCBRecord:
    image_path: Path
    boxes: tv_tensors.BoundingBoxes
    labels: Tensor


def _parse_annotation(annotation_path: Path) -> tuple[Tensor, Tensor]:
    raw_lines = annotation_path.read_text(encoding="utf-8").splitlines()

    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 values per annotation row in {annotation_path}, got {len(parts)} on line {line_number}."
            )

        x1, y1, x2, y2, label = map(int, parts)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid box in {annotation_path} on line {line_number}: {line}"
            )

        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        labels.append(label)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.int64
        )

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.int64
    )


def _build_dataset_manifest(
    dataset_root: Path,
    split_file: str,
    image_size: tuple[int, int],
) -> list[PCBRecord]:
    samples: list[PCBRecord] = []

    # Get split file
    split_path = dataset_root / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    for line_number, line in enumerate(
        split_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        # Get the image and annots reference from the split file
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Expected two columns in {split_path} on line {line_number}, got {len(parts)}."
            )
        image_reference, annotation_reference = parts

        # Image (lazy)
        image_reference = Path(image_reference)
        image_path = (
            dataset_root / image_reference.parent / f"{image_reference.stem}_test.jpg"
        )
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Annotations
        annotation_path = dataset_root / annotation_reference
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        boxes, labels = _parse_annotation(annotation_path)
        bounding_boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=image_size,
        )

        # Add
        samples.append(
            PCBRecord(image_path=image_path, boxes=bounding_boxes, labels=labels)
        )

    return samples


def build_pcb_target(index: int, boxes: Tensor, labels: Tensor) -> PCBTarget:
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor(index, dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
    }


class PCBDataset(Dataset[PCBItem]):
    def __init__(
        self,
        train_config: NeuroTrainConfig,
        split_file: str,
    ) -> None:
        self.split_file = split_file
        self.samples = _build_dataset_manifest(
            Path(train_config["dataset"]["path"]),
            self.split_file,
            image_size=tuple(train_config["dataset"]["size"]),
        )
        self.class_names = tuple(train_config["dataset"]["class_names"])
        self.preprocess: PCBPreprocess | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PCBItem:
        if self.preprocess is None:
            raise RuntimeError(
                "PCBDataset.preprocess is not set; call add_preprocess() first."
            )
        sample = self.samples[index]

        # Load Image
        image = Image.open(sample.image_path).convert("RGB")

        target = build_pcb_target(index, sample.boxes, sample.labels)

        return self.preprocess(image, target)

    def add_preprocess(self, preprocess: PCBPreprocess):
        self.preprocess = preprocess

    def inspect(self, n: int = 6) -> None:
        if len(self.samples) == 0:
            raise ValueError("Cannot inspect an empty dataset.")

        n = min(max(n, 1), len(self.samples))
        inspected_samples = random.sample(self.samples, k=n)

        inspection = _build_inspection(
            self.samples,
            self.class_names,
            inspected_samples,
        )
        _print_inspection_summary(self.split_file, inspection)
        _plot_class_counts(inspection)
        _plot_random_samples(inspected_samples, self.class_names)


def _build_inspection(
    samples: list[PCBRecord],
    class_names: tuple[str, ...],
    inspected_samples: list[PCBRecord],
) -> InspectionSummary:
    # Full-split statistics use cached annotations, while image sizes use only
    # the sampled files chosen by inspect().
    annotation_counts = torch.tensor(
        [sample.boxes.shape[0] for sample in samples],
        dtype=torch.int64,
    )
    non_empty_labels = [
        sample.labels for sample in samples if sample.labels.numel() > 0
    ]

    if non_empty_labels:
        labels = torch.cat(non_empty_labels, dim=0)
        unique_labels, label_counts = torch.unique(
            labels, sorted=True, return_counts=True
        )
        label_count_by_id = {
            int(label): int(count)
            for label, count in zip(unique_labels.tolist(), label_counts.tolist())
        }
        present_class_count = int(unique_labels.numel())
    else:
        label_count_by_id = {}
        present_class_count = 0

    class_ids = list(range(1, len(class_names) + 1))
    class_labels = [class_names[class_id - 1] for class_id in class_ids]
    class_counts = [label_count_by_id.get(class_id, 0) for class_id in class_ids]

    return {
        "sample_count": len(samples),
        "class_labels": class_labels,
        "class_counts": class_counts,
        "present_class_count": present_class_count,
        "total_defect_boxes": annotation_counts.sum().item(),
        "empty_image_count": (annotation_counts == 0).sum().item(),
        "min_boxes_per_image": annotation_counts.min().item(),
        "mean_boxes_per_image": annotation_counts.float().mean().item(),
        "max_boxes_per_image": annotation_counts.max().item(),
        "image_size_checked_count": len(inspected_samples),
    }


def _print_inspection_summary(split_file: str, inspection: InspectionSummary) -> None:
    # Keep scalar stats as text so the plots can focus on visual distributions.
    print(f"========== Inspect {split_file} ==========")
    print("DeepPCB dataset inspection")
    print(f"Files: {inspection['sample_count']:,}")
    print(f"Configured classes: {len(inspection['class_labels']):,}")
    print(f"Classes present in annotations: {inspection['present_class_count']:,}")
    print(f"Empty images: {inspection['empty_image_count']:,}")
    print(f"Total defect boxes: {inspection['total_defect_boxes']:,}")
    print(
        "Defect boxes per image: "
        f"min {inspection['min_boxes_per_image']}, "
        f"mean {inspection['mean_boxes_per_image']:.2f}, "
        f"max {inspection['max_boxes_per_image']}"
    )


def _plot_class_counts(inspection: InspectionSummary) -> None:
    # Class imbalance is the main dataset-level signal worth plotting.
    class_fig, class_ax = plt.subplots(figsize=(10, 4.8))
    bars = class_ax.bar(
        inspection["class_labels"], inspection["class_counts"], color="#2f6f73"
    )
    class_ax.set_title("Annotations per Class", loc="left", fontsize=14, weight="bold")
    class_ax.set_ylabel("Annotations")
    class_ax.tick_params(axis="x", rotation=30)
    class_ax.grid(axis="y", alpha=0.25)
    class_ax.bar_label(bars, padding=3, fontsize=9)
    class_fig.tight_layout()
    plt.show()


def _draw_annotated_sample(
    ax,
    sample: PCBRecord,
    class_names: tuple[str, ...],
    palette: list[str],
) -> None:
    # Draw directly from the cached manifest record to avoid applying training transforms.
    image = Image.open(sample.image_path).convert("RGB")
    ax.imshow(image)
    ax.set_title(
        f"{sample.image_path.stem} | {sample.boxes.shape[0]} annotations",
        fontsize=11,
        loc="left",
    )
    ax.axis("off")

    for box, label in zip(sample.boxes.tolist(), sample.labels.tolist()):
        x1, y1, x2, y2 = box
        class_index = int(label) - 1
        color = palette[class_index % len(palette)]
        label_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"class {int(label)}"
        )

        rectangle = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.6,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rectangle)
        ax.text(
            x1,
            max(y1 - 3, 0),
            label_name,
            color="white",
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={
                "facecolor": color,
                "edgecolor": color,
                "pad": 1.5,
            },
        )


def _plot_random_samples(
    samples: list[PCBRecord],
    class_names: tuple[str, ...],
) -> None:
    # The caller already chose the random subset so size reporting and plotting match.
    n = len(samples)
    plot_columns = 3
    sample_rows = math.ceil(n / plot_columns)

    sample_fig, axes = plt.subplots(
        sample_rows,
        plot_columns,
        figsize=(5.2 * plot_columns, 4.4 * sample_rows),
        squeeze=False,
    )
    sample_fig.suptitle(
        f"Random Samples ({n})",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=16,
        weight="bold",
    )

    for plot_index, sample in enumerate(samples):
        row = plot_index // plot_columns
        column = plot_index % plot_columns
        _draw_annotated_sample(
            axes[row][column],
            sample,
            class_names,
            PREVIEW_PALETTE,
        )

    for empty_index in range(n, sample_rows * plot_columns):
        row = empty_index // plot_columns
        column = empty_index % plot_columns
        axes[row][column].axis("off")

    sample_fig.tight_layout()
    plt.show()
