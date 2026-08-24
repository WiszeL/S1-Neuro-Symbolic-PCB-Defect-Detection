from __future__ import annotations

from typing import NotRequired, TypedDict


class SymbolicDataConfig(TypedDict):
    neg_ratio: float
    storage_dtype: str
    # Image-level model-selection split of the trainval export (see
    # symbolic/dataset.py::open_exported_symbolic_array_payload). None/absent
    # uses the full dump — the setting for the final promoted training run.
    split: NotRequired[str | None]
    val_fraction: NotRequired[float]
    split_seed: NotRequired[int]


class SymbolicSearchConfig(TypedDict):
    tree_depth: int
    iterations: int
    l1_lambda: float
    sparsity_alpha: float
    logistic_max_iter: int
    tolerance: float
    zero_threshold: float
    random_state: int
    class_weights: NotRequired[dict[str, float] | None]


class SymbolicTrainConfig(TypedDict):
    data: SymbolicDataConfig
    search: SymbolicSearchConfig
