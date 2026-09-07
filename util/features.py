from __future__ import annotations

import numpy as np


def ensure_float32(features: np.ndarray) -> np.ndarray:
    """Float32 view that keeps big feature files on disk, not in RAM."""
    if features.dtype == np.float32:
        return features
    return np.asarray(features, dtype=np.float32)
