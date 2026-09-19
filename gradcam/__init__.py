"""Grad-CAM baseline for qualitative comparison against SODT explanations."""

from .evaluation import evaluate_gradcam_faithfulness, evaluate_shared_localization
from .gradcam import GradCAM
from .visualize import gradcam_for_detections, plot_gradcam_comparison

__all__ = [
    "GradCAM",
    "evaluate_gradcam_faithfulness",
    "evaluate_shared_localization",
    "gradcam_for_detections",
    "plot_gradcam_comparison",
]
