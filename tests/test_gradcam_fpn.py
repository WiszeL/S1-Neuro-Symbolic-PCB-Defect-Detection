"""Grad-CAM explains the decision the detector actually made: at the proposal, on its FPN level."""

from pathlib import Path

import torch

from gradcam.gradcam import GradCAM
from neuro.faster_rcnn import NeuroFasterRCNN
from util.config import load_yaml


def _untrained_detector() -> NeuroFasterRCNN:
    neuro_config = load_yaml(Path("neuro.yaml"), dict)
    neuro_config["net"]["backbone_pretrained"] = False
    train_config = load_yaml(Path("neuro_train.yaml"), dict)
    torch.manual_seed(0)
    return NeuroFasterRCNN(neuro_config, train_config).eval()


def test_detections_carry_the_proposal_they_were_classified_from():
    model = _untrained_detector()
    seen: dict[str, torch.Tensor] = {}
    postprocess = model.postprocess_detections

    def spy(class_logits, box_regression, proposals, image_shapes):
        seen["proposals"] = proposals[0].clone()
        return postprocess(class_logits, box_regression, proposals, image_shapes)

    model.postprocess_detections = spy
    with torch.inference_mode():
        detection = model([torch.rand(3, 128, 128)])[0]

    carried = detection["proposal_boxes_processed"]
    assert carried.shape[0] == detection["boxes"].shape[0]
    assert (carried[:, None, :] == seen["proposals"][None]).all(-1).any(1).all()


def test_cam_lives_on_the_rois_own_fpn_level():
    model = _untrained_detector()
    gradcam = GradCAM(model, device="cpu")
    image = torch.rand(3, 128, 128)
    boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0], [5.0, 5.0, 600.0, 600.0]])

    maps = gradcam.fpn_maps(image, boxes, torch.tensor([1, 2]))
    names = list(model.roi_align.pool.featmap_names)
    levels = model.roi_align.pool.map_levels([boxes])
    for row, cam in enumerate(maps["cams"]):
        level_name = names[int(levels[row])]
        assert maps["level_names"][row] == level_name
        assert cam.shape == maps["fpn_features"][level_name].shape[-2:]
        assert (cam >= 0).all()  # ReLU'd

    heatmaps = gradcam.generate(image, boxes, torch.tensor([1, 2]))
    assert all(h.shape == (7, 7) for h in heatmaps)
