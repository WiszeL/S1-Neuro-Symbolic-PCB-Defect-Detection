from __future__ import annotations

from typing import TypedDict


class SymbolicDataConfig(TypedDict):
    neg_ratio: float
    storage_dtype: str


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
