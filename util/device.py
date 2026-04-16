from __future__ import annotations

import torch


def select_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(requested_device)
