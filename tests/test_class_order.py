"""Class names must follow DeepPCB's annotation ids, or every per-class number lies."""

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "neuro_train.yaml"


def test_class_names_follow_deeppcb_ids():
    # DeepPCB README: 1 open, 2 short, 3 mousebite, 4 spur, 5 copper, 6 pin-hole.
    names = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["dataset"]["class_names"]
    assert names == ["open", "short", "mouse_bite", "spur", "spurious_copper", "pinhole"]


if __name__ == "__main__":
    test_class_names_follow_deeppcb_ids()
    print("OK")
