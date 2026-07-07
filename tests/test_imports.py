"""Smoke check: every top-level package imports cleanly."""


def test_imports():
    import gradcam  # noqa: F401
    import neuro  # noqa: F401
    import neurosym  # noqa: F401
    import symbolic  # noqa: F401
    import util  # noqa: F401


if __name__ == "__main__":
    test_imports()
    print("OK")
