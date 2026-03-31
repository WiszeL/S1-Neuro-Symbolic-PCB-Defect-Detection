from __future__ import annotations

import random
from typing import Callable

from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: list[Callable[[Image.Image | Tensor, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]]):
        self.transforms = transforms

    def __call__(self, image: Image.Image | Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image | Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, image: Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if random.random() >= self.probability:
            return image, target

        image = F.hflip(image)
        _, _, width = image.shape
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, probability: float = 0.0) -> None:
        self.probability = probability

    def __call__(self, image: Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if random.random() >= self.probability:
            return image, target

        image = F.vflip(image)
        _, height, _ = image.shape
        boxes = target["boxes"].clone()
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        target["boxes"] = boxes
        return image, target


def build_train_transforms(horizontal_flip_prob: float = 0.5, vertical_flip_prob: float = 0.0) -> Compose:
    return Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(horizontal_flip_prob),
            RandomVerticalFlip(vertical_flip_prob),
        ]
    )


def build_eval_transforms() -> Compose:
    return Compose([ToTensor()])
