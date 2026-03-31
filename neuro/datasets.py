from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


DEFAULT_DEEPPCB_CLASS_NAMES = (
    "open",
    "short",
    "mouse_bite",
    "spur",
    "pinhole",
    "spurious_copper",
)


@dataclass(frozen=True)
class DeepPCBSample:
    image_path: Path
    annotation_path: Path
    sample_key: str
    group_name: str


def resolve_test_image_path(
    dataset_root: str | Path, image_reference: str | Path
) -> Path:
    dataset_root = Path(dataset_root)
    image_reference = Path(image_reference)
    preferred = (
        dataset_root / image_reference.parent / f"{image_reference.stem}_test.jpg"
    )

    if preferred.exists():
        return preferred

    direct = dataset_root / image_reference
    if direct.exists():
        return direct

    raise FileNotFoundError(
        f"Unable to resolve the non-referential DeepPCB image for {image_reference} under {dataset_root}."
    )


def parse_deeppcb_annotation(annotation_path: str | Path) -> tuple[Tensor, Tensor]:
    annotation_path = Path(annotation_path)
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


def build_deeppcb_manifest(
    dataset_root: str | Path, split_file: str | Path
) -> list[DeepPCBSample]:
    dataset_root = Path(dataset_root)
    split_path = (
        split_file if Path(split_file).is_absolute() else dataset_root / split_file
    )

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    samples: list[DeepPCBSample] = []

    # ---------------------------------------------------------------------
    # Resolve the raw split manifest into lazy image/annotation references
    # ---------------------------------------------------------------------
    for line_number, line in enumerate(
        split_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Expected two columns in {split_path} on line {line_number}, got {len(parts)}."
            )

        image_reference, annotation_reference = parts
        image_path = resolve_test_image_path(dataset_root, image_reference)
        annotation_path = dataset_root / annotation_reference

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        sample_key = Path(image_reference).stem
        samples.append(
            DeepPCBSample(
                image_path=image_path,
                annotation_path=annotation_path,
                sample_key=sample_key,
                group_name=Path(image_reference).parts[0],
            )
        )

    return samples


def summarize_deeppcb_split(
    dataset_root: str | Path,
    split_file: str | Path,
    class_names: tuple[str, ...] = DEFAULT_DEEPPCB_CLASS_NAMES,
) -> dict[str, object]:
    samples = build_deeppcb_manifest(dataset_root, split_file)
    class_counts = {name: 0 for name in class_names}
    total_boxes = 0

    for sample in samples:
        _, labels = parse_deeppcb_annotation(sample.annotation_path)
        total_boxes += int(labels.numel())
        for label in labels.tolist():
            class_counts[class_names[label - 1]] += 1

    return {
        "num_images": len(samples),
        "num_boxes": total_boxes,
        "class_counts": class_counts,
    }


class DeepPCBDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    def __init__(
        self,
        dataset_root: str | Path,
        split_file: str | Path,
        transforms: (
            Callable[[Image.Image, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]
            | None
        ) = None,
        class_names: tuple[str, ...] = DEFAULT_DEEPPCB_CLASS_NAMES,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples = build_deeppcb_manifest(self.dataset_root, split_file)
        self.transforms = transforms
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.samples[index]

        # ---------------------------------------------------------------------
        # Load the grayscale PCB image and expand it to three channels
        # ---------------------------------------------------------------------
        image = Image.open(sample.image_path).convert("RGB")
        boxes, labels = parse_deeppcb_annotation(sample.annotation_path)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image, target)
        else:
            image_tensor = F.to_tensor(image)

        return image_tensor, target
