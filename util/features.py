from __future__ import annotations

import numpy as np


def ensure_float32(features: np.ndarray) -> np.ndarray:
    """Coerce to float32, preserving memmap arrays when the dtype already matches.

    Unlike ``np.asarray(features, dtype=np.float32)``, which always copies a
    memmap into a regular ndarray, this returns the original object when the
    dtype already matches, keeping large RoI feature arrays disk-backed.
    """
    if features.dtype == np.float32:
        return features
    return np.asarray(features, dtype=np.float32)
