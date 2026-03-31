from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .utils import ensure_dir, select_device


@torch.inference_mode()
def export_dataset_roi_features(
    model: torch.nn.Module,
    dataset: Any,
    output_path: str | Path,
    device: str | None = None,
    proposal_source: str = "detections",
) -> Path:
    if proposal_source not in {"detections", "targets"}:
        raise ValueError("proposal_source must be either 'detections' or 'targets'.")

    resolved_device = select_device(device)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    model = model.to(resolved_device)
    model.eval()

    records: list[dict[str, Any]] = []
    for index in tqdm(range(len(dataset)), desc=f"Export RoI features ({proposal_source})", leave=False):
        image, target = dataset[index]
        image = image.to(resolved_device)
        target_on_device = {key: value.to(resolved_device) for key, value in target.items()}

        # ---------------------------------------------------------------------
        # Keep the original baseline export paths for notebook 02 compatibility
        # ---------------------------------------------------------------------
        if proposal_source == "detections":
            output = model([image])[0]
            record = {
                "image_id": int(target["image_id"]),
                "image_path": str(dataset.samples[index].image_path),
                "boxes": output["boxes"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "roi_features": output["roi_features"].detach().cpu(),
            }
        else:
            output = model.extract_target_roi_features([image], [target_on_device])[0]
            record = {
                "image_id": int(target["image_id"]),
                "image_path": str(dataset.samples[index].image_path),
                "boxes": output["boxes"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
                "pooled_features": output["pooled_features"].detach().cpu(),
                "roi_features": output["roi_features"].detach().cpu(),
            }

        records.append(record)

    payload = {
        "proposal_source": proposal_source,
        "class_names": tuple(dataset.class_names),
        "records": records,
    }
    torch.save(payload, output_path)
    return output_path
