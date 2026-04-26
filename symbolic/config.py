from __future__ import annotations

from typing import TypedDict


class SymbolicDataConfig(TypedDict):
    include_background: bool
    min_teacher_score: float | None
    max_samples_total: int | None


class SymbolicSearchConfig(TypedDict):
    max_depth: int
    depth_values: int | list[int] | None
    iterations: int
    lambda_values: float | list[float] | None
    alpha_values: float | list[float] | None
    logistic_max_iter: int
    tolerance: float
    zero_threshold: float
    random_state: int


class SymbolicTrainConfig(TypedDict):
    data: SymbolicDataConfig
    search: SymbolicSearchConfig
