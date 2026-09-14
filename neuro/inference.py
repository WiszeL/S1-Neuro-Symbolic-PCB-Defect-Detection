from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from util.device import select_device

from util.config import load_yaml
from .config import NeuroConfig, NeuroTrainConfig
from .faster_rcnn import NeuroFasterRCNN


def load_checkpoint_model(
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    device: str | None = None,
) -> tuple[NeuroFasterRCNN, dict[str, Any]]:
    resolved_device = select_device(device)
    model_config = load_yaml(model_config_path, NeuroConfig)
    train_config = load_yaml(train_config_path, NeuroTrainConfig)
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=True
    )

    model = NeuroFasterRCNN(
        neuro_config=model_config,
        train_config=train_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def run_inference(
    model: NeuroFasterRCNN,
    images: list[Tensor],
    device: str | torch.device,
) -> list[dict[str, Tensor]]:
    resolved_device = select_device(str(device))
    model.eval()
    outputs = model([image.to(resolved_device) for image in images])
    return [
        {key: value.detach().cpu() for key, value in output.items()}
        for output in outputs
    ]
