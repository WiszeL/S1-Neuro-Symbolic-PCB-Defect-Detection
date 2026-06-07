"""Grad-CAM baseline for qualitative comparison against SODT explanations."""

from .gradcam import GradCAM
from .visualize import (
    gradcam_for_detections,
    plot_gradcam_comparison,
    run_gradcam_baseline,
)

__all__ = [
    "GradCAM",
    "gradcam_for_detections",
    "plot_gradcam_comparison",
    "run_gradcam_baseline",
]
