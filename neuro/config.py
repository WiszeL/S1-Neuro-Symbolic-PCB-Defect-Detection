from pathlib import Path
from typing import Literal, TypedDict

from util.config import load_yaml


# Model config


class NetConfig(TypedDict):
    backbone_pretrained: bool
    backbone_freeze_batch_norm: bool
    neck_attention_reduction: int


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
    anchors: AnchorConfig
    proposal: ProposalConfig
    soft_nms: SoftNmsConfig


# Training config


class DatasetConfig(TypedDict):
    path: str
    size: list[int]
    class_names: list[str]
    batch_size: int
    num_workers: int


class PreprocessingConfig(TypedDict):
    horizontal_flip_prob: float
    train_min_sizes: list[int]
    test_min_size: int
    max_size: int
    mean: list[float]
    std: list[float]


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
    preprocessing: PreprocessingConfig
    train: TrainConfig
    evaluation: EvaluationConfig
