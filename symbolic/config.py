from __future__ import annotations

from typing import TypedDict


class SymbolicDataConfig(TypedDict):
    include_background: bool
    min_teacher_score: float | None
    max_samples_total: int | None


class SymbolicSearchConfig(TypedDict):
    tree_depth: int
    iterations: int
    l1_lambda: float
    sparsity_alpha: float
    logistic_max_iter: int
    tolerance: float
    zero_threshold: float
    random_state: int


class SymbolicTrainConfig(TypedDict):
    data: SymbolicDataConfig
    search: SymbolicSearchConfig
