from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml


def load_yaml[T](path: str | Path, _: type[T]) -> T:
    config_path = Path("configs") / Path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping in {config_path}, but received {type(data).__name__}."
        )

    return cast(T, data)
