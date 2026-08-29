"""Standalone Grad-CAM explanation-quality evaluation.

Computes the same five explanation metrics used for the SODT, but adapted
for Grad-CAM heatmaps:

1. **Sufficiency** (prediction preservation) — keeping only the spatial
   positions with the highest Grad-CAM activation preserves the neural
   classifier's predicted class.
2. **Necessity** (prediction flip rate) — zeroing out the high-activation
   positions causes the classifier to change its prediction.
3. **Deletion AUC** — as features are removed in order of decreasing
   Grad-CAM importance, the classifier confidence for the predicted class
   should drop quickly (lower AUC = better).
4. **Insertion AUC** — as features are added in order of decreasing
   importance, confidence should rise quickly (higher AUC = better).
5. **Pointing Game** — the peak of the Grad-CAM heatmap falls inside the
   ground-truth defect region.
6. **IoU Heatmap** (box-grounded ROI overlap) — the top-k heatmap cells
   (k = number of GT cells) overlap with the ground-truth region.

All feature-level metrics use the *neural* classifier's probabilities on
masked/unmasked pooled features, ensuring the metrics measure how well the
Grad-CAM heatmap explains the *same* model that produced it.

Perturbation protocol — shared with symbolic/evaluation.py so the two sides
are directly comparable: one 7x7 grid cell (all channels) per unit, ranked
by each method's own heatmap via util.heatmap_metrics.importance_ranking,
identical cell budget (top-50% for sufficiency/necessity) and step schedule
(k = s/steps * num_cells for deletion/insertion). Each side probes its own
model — this is a faithfulness check, not a cross-model comparison. See
symbolic/evaluation.py's module docstring for why that split matters.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

from neuro.faster_rcnn import NeuroFasterRCNN
from util.geometry import project_gt_box_to_roi_grid
from util.heatmap_metrics import (
    importance_ranking,
    normalize_heatmap,
    pointing_score,
    stratified_spatial_result,
    topk_region_overlap,
)

from .gradcam import GradCAM

_NAN = float("nan")
_ROI_GRID = (7, 7)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _neural_probs(
    model: NeuroFasterRCNN,
    pooled_features: Tensor,
) -> Tensor:
    """Run pooled features through the neural classifier head.

    Returns softmax probabilities ``[N, num_classes]``.
    """
    with torch.no_grad():
        roi_repr = model.box_head(pooled_features.to(next(model.parameters()).device))
        logits = model.box_predictor.classifier(roi_repr)
    return F.softmax(logits, dim=-1).detach().cpu()


def _match_proposals_to_gt(
    proposal_boxes: Tensor,
    gt_boxes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match each proposal to its best-overlapping GT box.

    Returns ``(matched_gt_boxes, has_matched, matched_iou)``.
    """
    if gt_boxes.numel() == 0 or proposal_boxes.numel() == 0:
        n = proposal_boxes.shape[0]
        return (
            torch.zeros((n, 4)),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n),
        )
    overlaps = box_iou(proposal_boxes, gt_boxes)
    matched_iou, matched_idx = overlaps.max(dim=1)
    matched_boxes = gt_boxes[matched_idx]
    has_match = matched_iou > 0
    matched_boxes = torch.where(
        has_match[:, None], matched_boxes, torch.zeros_like(matched_boxes)
    )
    return matched_boxes, has_match, matched_iou


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def evaluate_gradcam(
    model: NeuroFasterRCNN,
    gradcam: GradCAM,
    images: list[Tensor],
    targets: list[dict[str, Tensor]],
    score_threshold: float = 0.3,
    num_images: int | None = None,
    num_samples: int = 2000,
    deletion_insertion_steps: int = 10,
    random_state: int = 42,
    min_proposal_iou: float = 0.0,
    max_proposal_iou: float = 1.0,
) -> dict[str, Any]:
    """Evaluate Grad-CAM explanation quality over a set of images.

    Parameters
    ----------
    model:
        Trained ``NeuroFasterRCNN`` in eval mode.
    gradcam:
        Initialized ``GradCAM`` instance.
    images:
        List of preprocessed image tensors ``[3, H, W]``.
    targets:
        Corresponding annotation dicts with ``boxes`` and ``labels``.
    score_threshold:
        Minimum detection score to include a detection in evaluation.
    num_images:
        Maximum number of images to evaluate. ``None`` uses all images.
        50 is a good speed/accuracy trade-off (~500 ROIs).
    num_samples:
        Maximum detections to use for deletion/insertion AUC.
    deletion_insertion_steps:
        Number of intermediate steps for the deletion/insertion curves.
    random_state:
        Seed for the subsampling RNG.
    min_proposal_iou:
        Minimum GT-match IoU for a detection to enter the spatial metrics.
        Pass the same value given to evaluate_symbolic_spatial_metrics so
        the two sides' spatial numbers are computed over comparably
        GT-aligned populations, not one filtered and one not.

    Returns
    -------
    dict
        Keys: ``sufficiency_prediction_preservation``,
        ``necessity_prediction_flip_rate``,
        ``deletion_auc``, ``insertion_auc``,
        ``pointing_score``, ``box_grounded_roi_overlap``,
        ``evaluated_roi_count``, ``total_detections``,
        ``low_gt_coverage`` (same three spatial keys, restricted to RoIs
        whose GT box covers less of the grid — see stratified_spatial_result).
    """
    rng_img = np.random.default_rng(random_state)
    if num_images is not None and num_images < len(images):
        total_images = len(images)
        chosen = rng_img.choice(total_images, size=num_images, replace=False)
        images = [images[i] for i in chosen]
        targets = [targets[i] for i in chosen]
        print(f"  Subsampled {num_images}/{total_images} images for GradCAM eval")

    device = next(model.parameters()).device

    # ── Collect detections, heatmaps, and pooled features ────────
    all_pooled: list[Tensor] = []
    all_heatmaps: list[Tensor] = []
    all_det_boxes: list[Tensor] = []
    all_matched_gt: list[Tensor] = []
    all_has_gt: list[Tensor] = []
    all_matched_iou: list[Tensor] = []
    per_image_times: list[float] = []

    n_images = len(images)
    for img_idx in tqdm(range(n_images), desc="Grad-CAM collection"):
        t_img = time.perf_counter()
        image = images[img_idx]
        target = targets[img_idx]

        # ── Step 1: real FRCNN detections (box decode + NMS) ─────────────
        try:
            with torch.inference_mode():
                preds = model([image.to(device)])
                det = preds[0]  # {boxes, scores, labels} in original coords

            keep = det["scores"] >= score_threshold
            det_boxes = det["boxes"][keep].detach().cpu()
            det_labels = det["labels"][keep].detach().cpu()

            n_kept = det_boxes.shape[0]
            if n_kept == 0:
                continue
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  [WARN] OOM on image {img_idx} (detection), skipping")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        # ── Step 2: backbone (enable_grad) + RoI Align + GradCAM ─────────
        try:
            img_tensor = image.to(device).unsqueeze(0)
            images_list, _ = model.transform(img_tensor, None)

            with torch.inference_mode(False), torch.enable_grad():
                # Clone input tensor inside normal autograd mode to ensure it's not an inference tensor.
                input_tensor = images_list.tensors.clone()
                features = model.backbone(input_tensor)

                # Scale detection boxes from original to transformed coords
                original_h, original_w = image.shape[-2:]
                processed_h, processed_w = images_list.image_sizes[0]
                scale_x = processed_w / original_w
                scale_y = processed_h / original_h

                scaled_boxes = det_boxes.clone().float().to(device)
                scaled_boxes[:, [0, 2]] *= scale_x
                scaled_boxes[:, [1, 3]] *= scale_y

                # RoI Align on detection boxes → pooled features (for faithfulness)
                pooled = model.roi_align(
                    features, [scaled_boxes], images_list.image_sizes
                )
                roi_repr = model.box_head(pooled)
                class_logits = model.box_predictor.classifier(roi_repr)

                # ── Grad-CAM backward per detection ───────────────────────────
                activations = gradcam._activations  # captured by forward hook
                heatmaps: list[Tensor] = []
                padded_h, padded_w = images_list.tensors.shape[-2:]

                for i in range(n_kept):
                    model.zero_grad()
                    target_class = int(det_labels[i])
                    score = class_logits[i, target_class]
                    score.backward(retain_graph=True)

                    gradients = gradcam._gradients
                    if gradients is None or activations is None:
                        heatmaps.append(torch.ones(_ROI_GRID, device=device))
                        continue

                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam)

                    cam_up = F.interpolate(
                        cam,
                        size=(padded_h, padded_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    cam_crop = cam_up[:, :, :processed_h, :processed_w]

                    x1 = max(0, int(scaled_boxes[i, 0]))
                    y1 = max(0, int(scaled_boxes[i, 1]))
                    x2 = min(processed_w, int(scaled_boxes[i, 2]) + 1)
                    y2 = min(processed_h, int(scaled_boxes[i, 3]) + 1)

                    if x2 <= x1 or y2 <= y1:
                        heatmaps.append(torch.ones(_ROI_GRID, device=device))
                        continue

                    roi_cam = cam_crop[:, :, y1:y2, x1:x2]
                    roi_cam = (
                        F.interpolate(
                            roi_cam,
                            size=_ROI_GRID,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                    cam_min = roi_cam.min()
                    cam_max = roi_cam.max()
                    if cam_max - cam_min > 1e-8:
                        roi_cam = (roi_cam - cam_min) / (cam_max - cam_min)
                    else:
                        roi_cam = torch.ones_like(roi_cam)

                    heatmaps.append(roi_cam.detach().cpu())

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"  [WARN] OOM on image {img_idx} ({n_kept} detections), skipping"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        # Match detection boxes to GT.
        gt_boxes = target["boxes"]
        matched_gt, has_gt, matched_iou = _match_proposals_to_gt(det_boxes, gt_boxes)

        all_pooled.append(pooled.detach().cpu())
        all_matched_iou.append(matched_iou)
        all_heatmaps.extend(heatmaps)
        all_det_boxes.append(det_boxes)
        all_matched_gt.append(matched_gt)
        all_has_gt.append(has_gt)
        per_image_times.append(time.perf_counter() - t_img)

        # Free GPU memory between images.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_pooled:
        empty_spatial = stratified_spatial_result([], [], [])
        return {
            "sufficiency_prediction_preservation": _NAN,
            "necessity_prediction_flip_rate": _NAN,
            "deletion_auc": _NAN,
            "insertion_auc": _NAN,
            **empty_spatial,
            "total_detections": 0,
            "inference_time_ms_avg": 0.0,
            "inference_time_ms_min": 0.0,
            "inference_time_ms_max": 0.0,
        }

    pooled_all = torch.cat(all_pooled, dim=0)
    det_boxes_all = torch.cat(all_det_boxes, dim=0)
    matched_gt_all = torch.cat(all_matched_gt, dim=0)
    has_gt_all = torch.cat(all_has_gt, dim=0)
    matched_iou_all = torch.cat(all_matched_iou, dim=0)
    N = pooled_all.shape[0]

    # Full (unmasked) neural classifier predictions.
    full_probs = _neural_probs(model, pooled_all)
    full_preds = full_probs.argmax(dim=1)
    full_conf = full_probs[torch.arange(N), full_preds]

    # ── Faithfulness: sufficiency & necessity ─────────────────────
    suf_preservation = np.empty(N, dtype=np.float64)
    nec_flip = np.empty(N, dtype=np.float64)

    for i in range(N):
        heatmap = all_heatmaps[i]
        ranking = importance_ranking(heatmap)
        grid_h, grid_w = _ROI_GRID
        # Threshold: top half of spatial positions are "important".
        threshold = max(int(grid_h * grid_w * 0.5), 1)
        important_positions = ranking[:threshold]

        # Necessity: zero out important positions.
        nec_feat = pooled_all[i].clone()
        for pos in important_positions:
            r, col = pos.item() // grid_w, pos.item() % grid_w
            nec_feat[:, r, col] = 0.0

        # Sufficiency: keep only important positions.
        suf_feat = torch.zeros_like(pooled_all[i])
        for pos in important_positions:
            r, col = pos.item() // grid_w, pos.item() % grid_w
            suf_feat[:, r, col] = pooled_all[i, :, r, col]

        nec_prob = _neural_probs(model, nec_feat.unsqueeze(0))[0]
        suf_prob = _neural_probs(model, suf_feat.unsqueeze(0))[0]

        nec_pred = nec_prob.argmax().item()
        suf_pred = suf_prob.argmax().item()

        nec_flip[i] = float(nec_pred != full_preds[i].item())
        suf_preservation[i] = float(suf_pred == full_preds[i].item())

    # ── Faithfulness: deletion / insertion AUC ────────────────────
    steps = deletion_insertion_steps
    rng = np.random.default_rng(random_state)
    auc_indices = (
        rng.choice(N, size=min(num_samples, N), replace=False)
        if N > num_samples
        else np.arange(N)
    )

    n_auc = len(auc_indices)
    deletion_curves = np.ones((n_auc, steps + 1))
    insertion_curves = np.zeros((n_auc, steps + 1))
    num_positions = _ROI_GRID[0] * _ROI_GRID[1]

    # Insertion step 0 starts from an all-zero input — its real confidence,
    # not a hardcoded 0.0 (that floor biased insertion AUC low regardless of
    # heatmap quality). The all-zero input is the same for every row, so one
    # forward pass suffices; only the readout class (pred_cls) varies.
    empty_probs = _neural_probs(model, torch.zeros((1, *pooled_all.shape[1:])))[0]

    for ai, idx in enumerate(auc_indices):
        heatmap = all_heatmaps[idx]
        ranking = importance_ranking(heatmap)
        pred_cls = full_preds[idx].item()

        # Step 0: full confidence (deletion) / empty-input confidence (insertion).
        deletion_curves[ai, 0] = full_conf[idx].item()
        insertion_curves[ai, 0] = empty_probs[pred_cls].item()

        for s in range(1, steps + 1):
            k = int(s / steps * num_positions)
            top_k = ranking[:k]

            # Deletion: start from full, remove top-k.
            del_feat = pooled_all[idx].clone()
            for pos in top_k:
                r, col = pos.item() // _ROI_GRID[1], pos.item() % _ROI_GRID[1]
                del_feat[:, r, col] = 0.0

            # Insertion: start from zero, add top-k.
            ins_feat = torch.zeros_like(pooled_all[idx])
            for pos in top_k:
                r, col = pos.item() // _ROI_GRID[1], pos.item() % _ROI_GRID[1]
                ins_feat[:, r, col] = pooled_all[idx, :, r, col]

            del_prob = _neural_probs(model, del_feat.unsqueeze(0))[0]
            ins_prob = _neural_probs(model, ins_feat.unsqueeze(0))[0]

            deletion_curves[ai, s] = del_prob[pred_cls].item()
            insertion_curves[ai, s] = ins_prob[pred_cls].item()

    dx = 1.0 / steps
    deletion_auc = float(np.trapz(deletion_curves, dx=dx, axis=1).mean())
    insertion_auc = float(np.trapz(insertion_curves, dx=dx, axis=1).mean())

    # ── Spatial: pointing game & IoU heatmap overlap ──────────────
    # min_proposal_iou filter matches evaluate_symbolic_spatial_metrics's
    # gt_iou filter, so the two sides are computed over comparably
    # GT-aligned populations rather than one filtered and one not.
    overlap_scores: list[float] = []
    pointing_scores: list[float] = []
    gt_coverage: list[float] = []

    for i in range(N):
        if not bool(has_gt_all[i]):
            continue
        if not (min_proposal_iou <= float(matched_iou_all[i]) <= max_proposal_iou):
            continue

        heatmap = all_heatmaps[i]
        norm_hm = normalize_heatmap(heatmap)
        gt_mask = project_gt_box_to_roi_grid(
            proposal_box=det_boxes_all[i],
            matched_gt_box=matched_gt_all[i],
            grid_shape=_ROI_GRID,
        )
        if gt_mask.sum().item() == 0:
            continue

        overlap_scores.append(topk_region_overlap(norm_hm, gt_mask))
        pointing_scores.append(pointing_score(norm_hm, gt_mask))
        gt_coverage.append(float(gt_mask.float().mean().item()))

    spatial_metrics = stratified_spatial_result(
        overlap_scores, pointing_scores, gt_coverage
    )

    times_arr = np.array(per_image_times) if per_image_times else np.array([0.0])
    return {
        "sufficiency_prediction_preservation": float(suf_preservation.mean()),
        "necessity_prediction_flip_rate": float(nec_flip.mean()),
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        **spatial_metrics,
        "total_detections": N,
        "inference_time_ms_avg": float(times_arr.mean()) * 1000,
        "inference_time_ms_min": float(times_arr.min()) * 1000,
        "inference_time_ms_max": float(times_arr.max()) * 1000,
    }
