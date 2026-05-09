from pathlib import Path
import os
import sys


def resolve_root() -> Path:
    root = Path(__file__).resolve().parents[1]

    os.chdir(root)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root
