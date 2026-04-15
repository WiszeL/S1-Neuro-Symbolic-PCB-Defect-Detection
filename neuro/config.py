from pathlib import Path
from typing import Literal, TypedDict, cast

import yaml


# Model config


class NetConfig(TypedDict):
    backbone_pretrained: bool
    backbone_freeze_batch_norm: bool
    neck_attention_reduction: int


class ImageConfig(TypedDict):
    train_min_sizes: list[int]
    test_min_size: int
    max_size: int
    mean: list[float]
    std: list[float]


class AnchorConfig(TypedDict):
    sizes: list[list[int]]
    aspect_ratios: list[list[float]]


class ProposalConfig(TypedDict):
    positive_iou: float
    negative_iou: float


class SoftNmsConfig(TypedDict):
    enabled: bool
    method: Literal["linear", "gaussian", "hard"]
    iou_thresh: float
    sigma: float
    score_thresh: float


class NeuroConfig(TypedDict):
    net: NetConfig
    image: ImageConfig
    anchors: AnchorConfig
    proposal: ProposalConfig
    soft_nms: SoftNmsConfig


# Training config


class DatasetConfig(TypedDict):
    path: str
    class_names: list[str]
    batch_size: int
    num_workers: int
    horizontal_flip_prob: float


class TrainConfig(TypedDict):
    epochs: int
    grad_accumulation_steps: int
    opt_lr: float
    opt_momentum: float
    opt_weight_decay: float
    sch_step_size: int
    sch_gamma: float
    sch_milestones: list[int]
    warmup_iterations: int
    warmup_ratio: float


class EvaluationConfig(TypedDict):
    precision_iou: float
    precision_score_threshold: float
    class_metrics: bool
    iou_thresholds: list[float]


class NeuroTrainConfig(TypedDict):
    seed: int
    deterministic: bool
    device: str
    amp: bool
    dataset: DatasetConfig
    train: TrainConfig
    evaluation: EvaluationConfig


# Loader


def load_yaml[T](path: Path, _: type[T]) -> T:
    config_path = Path("configs") / path

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping in {config_path}, but received {type(data).__name__}."
        )

    return cast(T, data)
