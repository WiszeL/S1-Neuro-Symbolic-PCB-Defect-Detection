from __future__ import annotations

import numpy as np
from torch import Tensor


def image_to_array(image_tensor: Tensor) -> np.ndarray:
    """CHW float tensor -> HWC array in [0, 1] for matplotlib imshow."""
    return image_tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
