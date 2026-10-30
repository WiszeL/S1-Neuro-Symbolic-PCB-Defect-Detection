"""Figures for Bab 4: Gambar 4.10 (deteksi 4 PCB), 4.11/4.12 (bukti FP/FN), 4.13 (keyakinan root), 4.14 (confusion matrix).

Run from anywhere: python papers/make_figures_bab4.py
Inference is cached in figures_bab4_v2/_cache_detections.pt (detections, NeSy per-node certainties) and
_cache_root_split.pt (N0's left/right counts); delete them to re-run the models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm

from neuro.config import NeuroConfig, NeuroTrainConfig
from neuro.prepare_dataset import PCBDataset
from neuro.preprocess_dataset import test_preprocess
from neuro.train import _global_confusion_matrix
from neurosym.evaluation import breakdown_nesy_misses
from neurosym.inference import load_neurosymbolic_detector
from util.config import load_yaml
from util.device import select_device
from util.visualization import image_to_array

OUT = ROOT / "papers" / "figures_bab4_v2"
CACHE = OUT / "_cache_detections.pt"
ROOT_CACHE = OUT / "_cache_root_split.pt"
FRCNN_THR, NESY_THR, IOU_THR = 0.5, 0.35, 0.5
CLASSES = ("open", "short", "mouse_bite", "spur", "spurious_copper", "pinhole")
MODELS = ("frcnn", "nesy", "nomargin")
THRESHOLD = {"frcnn": FRCNN_THR, "nesy": NESY_THR, "nomargin": NESY_THR}


def load_models() -> tuple[object, dict, torch.device]:
    neuro_config = load_yaml("neuro.yaml", NeuroConfig)
    train_config = load_yaml("neuro_train.yaml", NeuroTrainConfig)
    device = select_device(train_config["device"])
    hybrid, _ = load_neurosymbolic_detector(
        detector_checkpoint_path=ROOT / "checkpoints/neuro/neuro_chosen.pt",
        neuro_config=neuro_config,
        train_config=train_config,
        symbolic_checkpoint_path=ROOT / "checkpoints/symbolic/L0,01;S0.85.pt",
        device=str(device),
    )
    return hybrid, train_config, device


def load_test_set(train_config: dict) -> PCBDataset:
    dataset = PCBDataset(train_config, "test.txt")
    dataset.add_preprocess(test_preprocess())
    return dataset


def _keep(output: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: output[k].detach().cpu() for k in ("boxes", "scores", "labels")}


def _log_certainties(hybrid, pooled: Tensor) -> Tensor:
    """log σ(|margin|) of every node on each detection's path, [detections, nodes] (0 off the path)."""
    if pooled.shape[0] == 0:
        return torch.zeros(0, hybrid.symbolic_tree.num_internal_nodes)
    _, log_certainties = hybrid.symbolic_tree.path_log_certainties(pooled.flatten(start_dim=1).numpy())
    return torch.from_numpy(log_certainties)


@torch.inference_mode()
def run_all(hybrid, dataset: PCBDataset, device: torch.device) -> list[dict]:
    records = []
    for i in tqdm(range(len(dataset)), desc="inference"):
        image, target = dataset[i]
        record = {"gt_boxes": target["boxes"].cpu(), "gt_labels": target["labels"].cpu()}
        record["frcnn"] = _keep(hybrid.detector([image.to(device)])[0])
        hybrid.use_routing_margin = True
        nesy = hybrid([image])[0]
        record["nesy"] = _keep(nesy)
        record["nesy"]["log_certainties"] = _log_certainties(hybrid, nesy["pooled_features"])
        hybrid.use_routing_margin = False  # every detection scores 1.0 (Subbab 4.6.3 ablation)
        record["nomargin"] = _keep(hybrid([image])[0])
        records.append(record)
    hybrid.use_routing_margin = True
    return records


def load_records(hybrid, dataset, device) -> list[dict]:
    if CACHE.exists():
        return torch.load(CACHE, weights_only=True)
    records = run_all(hybrid, dataset, device)
    torch.save(records, CACHE)
    return records


def match(pred: dict[str, Tensor], gt_boxes: Tensor, gt_labels: Tensor, threshold: float):
    """Per-label greedy matching, same rule as neuro.train._greedy_match_within_label but per box.

    Returns (detections by descending score, unmatched GT indices). A detection is
    {"box", "label", "score", "status": "TP"|"FP", "gt": matched GT index or None, "row": index in `pred`}.
    """
    mask = pred["scores"] >= threshold
    boxes, scores, labels = pred["boxes"][mask], pred["scores"][mask], pred["labels"][mask]
    rows = torch.where(mask)[0]
    order = torch.argsort(scores, descending=True, stable=True)
    detections = [
        {"box": boxes[j], "label": int(labels[j]), "score": float(scores[j]), "status": "FP", "gt": None, "row": int(rows[j])}
        for j in order.tolist()
    ]
    for label in set(d["label"] for d in detections):
        mine = [d for d in detections if d["label"] == label]
        gt_idx = torch.where(gt_labels == label)[0]
        if gt_idx.numel() == 0:
            continue
        ious = box_iou(torch.stack([d["box"] for d in mine]), gt_boxes[gt_idx])
        claimed: set[int] = set()
        for row, det in enumerate(mine):
            best_iou, best = ious[row].max(dim=0)
            if best_iou.item() >= IOU_THR and int(best) not in claimed:
                claimed.add(int(best))
                det["status"], det["gt"] = "TP", int(gt_idx[int(best)])
    # Bab 4 counts a defect as missed (FN) only when no detection of ANY class covers it;
    # a wrong-class cover is a false positive instead (same rule as the confusion matrix).
    covered: set[int] = set()
    if detections and gt_boxes.numel():
        ious = box_iou(torch.stack([d["box"] for d in detections]), gt_boxes)
        for row in range(len(detections)):
            best_iou, best = ious[row].max(dim=0)
            if best_iou.item() >= IOU_THR and int(best) not in covered:
                covered.add(int(best))
    unmatched = [g for g in range(gt_boxes.shape[0]) if g not in covered]
    return detections, unmatched


def confusion(records: list[dict], model: str) -> np.ndarray:
    total = torch.zeros(7, 7, dtype=torch.int64)
    for r in records:
        target = {"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
        total += _global_confusion_matrix(r[model], target, IOU_THR, THRESHOLD[model], len(CLASSES))
    return total.numpy()


def check_against_text(records: list[dict]) -> dict[str, np.ndarray]:
    """The numbers quoted in Bab 4 must come back out of the cached detections."""
    cms = {m: confusion(records, m) for m in MODELS}
    # conf[row=truth, col=prediction]; row 0 = background. Classes: 1 open 2 short 3 mouse_bite 4 spur 5 spurious_copper 6 pinhole
    expect_bg = {"nesy": {1: 24, 2: 71, 5: 51, 6: 89}, "nomargin": {1: 152, 2: 264, 5: 143, 6: 218}}
    for model, cells in expect_bg.items():
        for col, value in cells.items():
            assert cms[model][0, col] == value, (model, "background ->", CLASSES[col - 1], cms[model][0, col], value)
    assert cms["nomargin"][3, 6] == 25 and cms["nomargin"][5, 6] == 16, "mouse_bite/spurious_copper -> pinhole"
    totals = {"frcnn": (331, 38), "nesy": (298, 47), "nomargin": (1036, None)}
    for model, (fp_expected, fn_expected) in totals.items():
        fp = fn = 0
        for r in records:
            detections, unmatched = match(r[model], r["gt_boxes"], r["gt_labels"], THRESHOLD[model])
            fp += sum(d["status"] == "FP" for d in detections)
            fn += len(unmatched)
        assert fp == fp_expected, (model, "FP", fp, fp_expected)
        assert fn_expected is None or fn == fn_expected, (model, "FN", fn, fn_expected)
    return cms


TP_GREEN, FP_RED, FN_YELLOW, REJECT_BLUE = "#2ca044", "#d62728", "#f0b400", "#1f77b4"
CLASS_COLORS = ("#d62728", "#0072b2", "#2a9d8f", "#f4a261", "#7b5ea7", "#6b705c")  # as the GT column always had
# Test-set indices of the PCBs shown (found by their detection scores; PCB 1 is the one reused in Subbab 4.7.2).
DETECTION_PCBS = (47, 406, 345, 126)
# (test index, PCB number in the text, FRCNN class, FRCNN score, NeSy candidate score)
EVIDENCE_FP = ((303, 5, "pinhole", 0.62, 0.29), (198, 6, "short", 0.78, 0.25), (13, 7, "spurious_copper", 0.76, 0.29), (56, 8, "mouse_bite", 0.62, 0.21))
EVIDENCE_FN = ((126, 4, "short", 0.89, 0.32), (86, 9, "mouse_bite", 0.53, 0.23), (410, 10, "open", 0.89, 0.27), (41, 11, "pinhole", 0.77, 0.30))


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def fmt_threshold(value: float) -> str:
    return f"{value:g}".replace(".", ",")  # 0,5 and 0,35, as in the tables


def draw_box(ax, box, color, linestyle="-", linewidth=1.3, pad=0.0) -> None:
    x1, y1, x2, y2 = (float(v) for v in box)
    ax.add_patch(plt.Rectangle((x1 - pad, y1 - pad), x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth))


def place_label(ax, box, text, background, placed: list, fontsize=5.5) -> None:
    """Text tag next to a box: first of a few anchors that stays inside the panel and hides no earlier tag."""
    renderer = ax.figure.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    x1, y1, x2, y2 = (float(v) for v in box)
    anchors = ((x1, y1, "left", "bottom"), (x1, y2, "left", "top"), (x2, y1, "right", "bottom"), (x2, y2, "right", "top"),
               (x1, y1 - 14, "left", "bottom"), (x1, y2 + 14, "left", "top"), (x2, y1 - 14, "right", "bottom"), (x2, y2 + 14, "right", "top"))
    fallback = None
    for x, y, ha, va in anchors:
        tag = ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color="white", fontweight="bold", clip_on=False,
                      bbox=dict(facecolor=background, edgecolor="none", pad=0.7))
        bb = tag.get_window_extent(renderer)
        inside = panel.x0 <= bb.x0 and bb.x1 <= panel.x1 and panel.y0 <= bb.y0 and bb.y1 <= panel.y1
        if inside and not any(bb.overlaps(p) for p in placed):
            placed.append(bb)
            return
        if inside and fallback is None:
            fallback = (x, y, ha, va)
        tag.remove()
    x, y, ha, va = fallback or anchors[0]
    tag = ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color="white", fontweight="bold", clip_on=False,
                  bbox=dict(facecolor=background, edgecolor="none", pad=0.7))
    placed.append(tag.get_window_extent(renderer))


def blank_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def detection_figure(records, dataset) -> None:
    """Gambar 4.10: GT | Faster R-CNN | NeSy for four PCBs. Every model box carries its score
    (a missed defect shows its best same-class candidate's score, or – if the model has none)."""
    fig, axes = plt.subplots(len(DETECTION_PCBS), 3, figsize=(9.0, 12.6), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.04, wspace=0.03, hspace=0.20)
    for row, index in enumerate(DETECTION_PCBS):
        image = image_to_array(dataset[index][0])
        record = records[index]
        gt_boxes, gt_labels = record["gt_boxes"], record["gt_labels"]
        ax = axes[row, 0]
        ax.imshow(image)
        placed: list = []
        for box, label in zip(gt_boxes, gt_labels.tolist()):
            draw_box(ax, box, CLASS_COLORS[label - 1])
            place_label(ax, box, CLASSES[label - 1], CLASS_COLORS[label - 1], placed, fontsize=5.5)
        ax.set_title(f"PCB {row + 1} – ground truth ({len(gt_labels)} cacat)", fontsize=8, fontweight="bold")
        for col, (model, name) in enumerate((("frcnn", "Faster R-CNN"), ("nesy", "Neuro-Symbolic")), start=1):
            ax = axes[row, col]
            ax.imshow(image)
            detections, missed = match(record[model], gt_boxes, gt_labels, THRESHOLD[model])
            placed = []
            # FP and FN first so their longer tags get the free spots.
            for number, det in sorted(enumerate(detections, start=1), key=lambda item: item[1]["status"] == "TP"):
                if det["status"] == "TP":
                    draw_box(ax, det["box"], TP_GREEN)
                    place_label(ax, det["box"], f"#{number} {fmt(det['score'])}", TP_GREEN, placed)
                else:
                    draw_box(ax, det["box"], FP_RED, linewidth=1.5)
                    place_label(ax, det["box"], f"#{number} {CLASSES[det['label'] - 1]} {fmt(det['score'])}", FP_RED, placed)
            for g in missed:
                draw_box(ax, gt_boxes[g], FN_YELLOW, linestyle="--", linewidth=1.5)
                pred = record[model]
                j = best_overlap(pred["boxes"], pred["scores"], pred["labels"], gt_boxes[g], int(gt_labels[g]))
                candidate = "–" if j is None else fmt(float(pred["scores"][j]))
                place_label(ax, gt_boxes[g], f"terlewat: {CLASSES[int(gt_labels[g]) - 1]} {candidate}", FN_YELLOW, placed)
            tp = sum(d["status"] == "TP" for d in detections)
            ax.set_title(f"PCB {row + 1} – {name} (ambang {fmt_threshold(THRESHOLD[model])})\n{tp} TP · {len(detections) - tp} FP · {len(missed)} FN", fontsize=8, fontweight="bold")
    for ax in axes.ravel():
        blank_axis(ax)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=TP_GREEN, label="deteksi benar (TP)"),
                        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=FP_RED, label="deteksi salah (FP)"),
                        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=FN_YELLOW, linestyle="--", label="cacat terlewat (FN)")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.savefig(OUT / "fig_deteksi_4pcb.png", facecolor="white")
    plt.close(fig)


def crop_window(box, size=640, minimum=96, factor=4.0):
    """Square window around a box, shifted (not shrunk) to stay inside the image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    side = min(max(factor * max(x2 - x1, y2 - y1), minimum), size)
    left = min(max((x1 + x2) / 2 - side / 2, 0), size - side)
    top = min(max((y1 + y2) / 2 - side / 2, 0), size - side)
    return left, top, left + side, top + side


def best_overlap(boxes: Tensor, scores: Tensor, labels: Tensor, box: Tensor, label: int | None):
    """Highest-scoring output overlapping `box` (IoU >= 0.5), optionally of one class."""
    if boxes.numel() == 0:
        return None
    ok = box_iou(boxes, box[None])[:, 0] >= IOU_THR
    if label is not None:
        ok &= labels == label
    if not bool(ok.any()):
        return None
    j = int(torch.where(ok, scores, torch.full_like(scores, -1.0)).argmax())
    return j


def weakest_node(hybrid, dataset, device, index: int, label: int, score: float) -> tuple[int, int]:
    """(node index, depth counted from 1) with the lowest log sigma(|margin|) on the candidate's decision path."""
    with torch.inference_mode():
        out = hybrid([dataset[index][0]])[0]
    j = [k for k in range(len(out["scores"])) if int(out["labels"][k]) == label and abs(float(out["scores"][k]) - score) < 1e-5]
    assert len(j) == 1, (index, label, score, j)
    pooled = out["pooled_features"][j[0]].detach().cpu().numpy().reshape(1, -1)
    _, log_certainties = hybrid.symbolic_tree.path_log_certainties(pooled)
    node = int(np.argmin(log_certainties[0]))
    return node, hybrid.symbolic_tree.depth_of(node) + 1


def evidence_rows(records, hybrid, dataset, device) -> tuple[list[dict], list[dict]]:
    """Resolve the eight examples against the cached detections; asserts they are the ones the text describes."""
    rows_fp, rows_fn = [], []
    for index, pcb, cls, frcnn_score, nesy_score in EVIDENCE_FP:
        r = records[index]
        label = CLASSES.index(cls) + 1
        detections, _ = match(r["frcnn"], r["gt_boxes"], r["gt_labels"], FRCNN_THR)
        hit = [d for d in detections if d["status"] == "FP" and d["label"] == label and abs(d["score"] - frcnn_score) < 0.006]
        assert len(hit) == 1, ("FP example", index, cls)
        n = r["nesy"]
        j = best_overlap(n["boxes"], n["scores"], n["labels"], hit[0]["box"], None)
        assert j is not None and abs(float(n["scores"][j]) - nesy_score) < 0.006 and float(n["scores"][j]) < NESY_THR, ("FP example NeSy", index)
        rows_fp.append(dict(index=index, pcb=pcb, cls=cls, frcnn=hit[0], nesy_box=n["boxes"][j], nesy_score=float(n["scores"][j])))
    for index, pcb, cls, frcnn_score, nesy_score in EVIDENCE_FN:
        r = records[index]
        label = CLASSES.index(cls) + 1
        detections, _ = match(r["frcnn"], r["gt_boxes"], r["gt_labels"], FRCNN_THR)
        hit = [d for d in detections if d["status"] == "TP" and d["label"] == label and abs(d["score"] - frcnn_score) < 0.006]
        assert len(hit) == 1, ("FN example", index, cls)
        gt_box = r["gt_boxes"][hit[0]["gt"]]
        n = r["nesy"]
        j = best_overlap(n["boxes"], n["scores"], n["labels"], gt_box, label)
        assert j is not None and abs(float(n["scores"][j]) - nesy_score) < 0.006 and float(n["scores"][j]) < NESY_THR, ("FN example NeSy", index)
        node, depth = weakest_node(hybrid, dataset, device, index, label, float(n["scores"][j]))
        rows_fn.append(dict(index=index, pcb=pcb, cls=cls, frcnn=hit[0], gt_box=gt_box, nesy_box=n["boxes"][j], nesy_score=float(n["scores"][j]), node=node, depth=depth))
    return rows_fp, rows_fn


def evidence_figure(rows: list[dict], group: str, dataset, filename: str) -> None:
    """Gambar 4.11 ("fp") / 4.12 ("fn"): PCB (one marked location) | Faster R-CNN | NeSy, one example per row."""
    fig = plt.figure(figsize=(7.6, 2.8 * len(rows) + 0.8), dpi=200)
    grid = fig.add_gridspec(len(rows), 3, hspace=0.5, wspace=0.04, left=0.01, right=0.99, top=0.95, bottom=0.07)
    marker_color = FP_RED if group == "fp" else FN_YELLOW
    all_axes = []
    for k, row in enumerate(rows):
        image = image_to_array(dataset[row["index"]][0])
        axes = [fig.add_subplot(grid[k, c]) for c in range(3)]
        all_axes += axes
        marker = row["frcnn"]["box"] if group == "fp" else row["gt_box"]
        axes[0].imshow(image)
        draw_box(axes[0], marker, marker_color, linewidth=1.8, pad=6)
        axes[0].set_title(f"PCB {row['pcb']} (citra uji #{row['index']})\n{row['cls']}", fontsize=7.5, fontweight="bold")
        left, top, right, bottom = crop_window(marker)
        for ax in axes[1:]:
            ax.imshow(image)
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
        draw_box(axes[1], row["frcnn"]["box"], FP_RED if group == "fp" else TP_GREEN, linewidth=1.8)
        draw_box(axes[2], row["nesy_box"], REJECT_BLUE, linestyle="--", linewidth=1.8)
        nesy_score = f"skor {fmt(row['nesy_score'])} < {fmt_threshold(NESY_THR)}"
        if group == "fp":
            axes[1].set_title(f"Faster R-CNN: gagal (FP)\nskor {fmt(row['frcnn']['score'])}", fontsize=7.5)
            axes[2].set_title(f"Neuro-Symbolic: berhasil (ditolak)\n{nesy_score}", fontsize=7.5)
        else:
            axes[1].set_title(f"Faster R-CNN: berhasil (TP)\nskor {fmt(row['frcnn']['score'])}", fontsize=7.5)
            axes[2].set_title(f"Neuro-Symbolic: gagal (terlewat)\n{nesy_score}\nnode terlemah N{row['node']} (kedalaman {row['depth']})", fontsize=7.5)
    for ax in all_axes:
        blank_axis(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REJECT_BLUE, linestyle="--", label="kandidat Neuro-Symbolic")]
    if group == "fp":
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=FP_RED, label="deteksi salah (FP) Faster R-CNN"))
    else:
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=TP_GREEN, label="deteksi benar (TP) Faster R-CNN"))
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=FN_YELLOW, label="ground truth"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=False)
    fig.savefig(OUT / filename, facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def root_split_counts(hybrid, dataset, device) -> np.ndarray:
    """[7, 2]: test-set RPN proposals by teacher label (0 = background) that N0 sends left / right."""
    if ROOT_CACHE.exists():
        return torch.load(ROOT_CACHE, weights_only=True).numpy()
    tree = hybrid.symbolic_tree
    w0, b0 = tree.node_weights[0], tree.node_bias[0]
    counts = np.zeros((len(CLASSES) + 1, 2), dtype=np.int64)
    for i in tqdm(range(len(dataset)), desc="root split"):
        image, target = dataset[i]
        sample = hybrid.detector.extract_teacher_roi_samples(
            [image.to(device)], [{k: v.to(device) for k, v in target.items()}]
        )[0]
        goes_left = (sample["pooled_features"].flatten(start_dim=1).cpu().numpy() @ w0 + b0) >= 0.0  # as sodt.decision_path
        np.add.at(counts, (sample["teacher_labels"].cpu().numpy(), (~goes_left).astype(int)), 1)
    torch.save(torch.from_numpy(counts), ROOT_CACHE)
    return counts


def root_confidence(records: list[dict], tree) -> dict[str, object]:
    """Root σ (Persamaan 3.3) of NeSy's true positives and of the detections lost to a score under the cutoff."""
    tp = np.array([float(np.exp(r["nesy"]["log_certainties"][d["row"], 0]))
                   for r in records
                   for d in match(r["nesy"], r["gt_boxes"], r["gt_labels"], NESY_THR)[0] if d["status"] == "TP"])
    targets = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]} for r in records]
    misses = breakdown_nesy_misses([r["frcnn"] for r in records], [r["nesy"] for r in records], targets,
                                   tree, FRCNN_THR, IOU_THR, nesy_score_threshold=NESY_THR)["score_misses"]
    rows = [dict(label=m["label"], sigma=float(np.exp(m["log_certainties"][0])), root=int(np.argmin(m["log_certainties"])) == 0)
            for m in misses]
    assert len(rows) == 14 and sum(m["root"] for m in rows) == 9, "Subbab 4.6.2: 14 score misses, root weakest in 9"
    return {"tp": tp, "misses": rows}


def root_figure(confidence: dict[str, object], counts: np.ndarray) -> None:
    """Gambar 4.13: (a) root σ of detected vs missed defects, (b) share of RoIs N0 sends left / right per class."""
    tp, misses = confidence["tp"], confidence["misses"]
    names = ["background", "open", "short", "mousebite", "spur", "spurious copper", "pinhole"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200, gridspec_kw=dict(width_ratios=(1, 1.25), wspace=0.32))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.27)

    rng = np.random.default_rng(0)
    ax_a.scatter(rng.uniform(-0.3, 0.3, tp.size), tp, s=6, color="#9aa5b1", alpha=0.25, linewidths=0, rasterized=True)
    for m in misses:
        color = CLASS_COLORS[m["label"] - 1]
        ax_a.scatter(1 + rng.uniform(-0.15, 0.15), m["sigma"], s=46, facecolors=color if m["root"] else "none", edgecolors=color, linewidths=1.6, zorder=3)
    for x, values in ((0, tp), (1, np.array([m["sigma"] for m in misses]))):
        ax_a.hlines(np.median(values), x - 0.38, x + 0.38, color="black", linewidth=1.6, zorder=4)
    ax_a.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax_a.text(-0.42, 0.505, "tidak yakin", fontsize=8, color="#555555", ha="left", va="bottom")
    ax_a.set_xlim(-0.45, 1.45)
    ax_a.set_ylim(0.48, 1.02)
    ax_a.set_xticks([0, 1], [f"Terdeteksi (TP)\nn = {tp.size}", f"Terlewat (skor < {fmt_threshold(NESY_THR)})\nn = {len(misses)}"], fontsize=8.5)
    ax_a.set_ylabel("Keyakinan root", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=CLASS_COLORS[c], markeredgecolor=CLASS_COLORS[c], markersize=6, label=names[c + 1])
               for c in range(len(CLASSES)) if any(m["label"] == c + 1 for m in misses)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="#555555", markeredgecolor="#555555", markersize=6, label="node terlemah: root"),
                plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#555555", markersize=6, label="node terlemah: lebih dalam"),
                plt.Line2D([], [], color="black", linewidth=1.6, label="median")]
    ax_a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7, frameon=False)

    share = counts / counts.sum(axis=1, keepdims=True) * 100
    y = np.arange(len(names))[::-1]
    ax_b.barh(y, share[:, 0], color="#4c78a8", label="kiri (f₀ ≥ 0)")
    ax_b.barh(y, share[:, 1], left=share[:, 0], color="#f58518", label="kanan (f₀ < 0)")
    for yi, (left, right) in zip(y, share):
        for x0, width in ((0, left), (left, right)):
            if width >= 6:
                ax_b.text(x0 + width / 2, yi, f"{width:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax_b.set_yticks(y, names, fontsize=8.5)
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Proporsi RoI menurut label teacher (%)", fontsize=9)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.03, letter, ha="center", va="center", fontsize=12, family="serif")
    fig.savefig(OUT / "fig_root_keyakinan.png", facecolor="white")
    plt.close(fig)

    sigma = np.array([m["sigma"] for m in misses])
    at_root = np.array([m["sigma"] for m in misses if m["root"]])
    deeper = np.array([m["sigma"] for m in misses if not m["root"]])
    print(f"root σ median: TP {np.median(tp):.3f} (n={tp.size}, IQR {np.percentile(tp, 25):.3f}-{np.percentile(tp, 75):.3f}); "
          f"14 misses {np.median(sigma):.3f} (min {sigma.min():.3f}, max {sigma.max():.3f}); "
          f"9 root {np.median(at_root):.3f} (max {at_root.max():.3f}); 5 deeper {np.median(deeper):.3f} (min {deeper.min():.3f}); "
          f"TP with σ <= max root case: {(tp <= at_root.max()).sum()} of {tp.size}")
    print("N0 left share % by teacher label:", {n: round(float(s), 1) for n, s in zip(names, share[:, 0])}, "counts", counts.tolist())


def confusion_figure(cms: dict[str, np.ndarray]) -> None:
    """Gambar 4.14: confusion matrix without (a) and with (b) routing margin, one shared color scale."""
    names = ["background", "open", "short", "mousebite", "spur", "spurious copper", "pinhole"]
    vmax = max(cms["nomargin"].max(), cms["nesy"].max())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.97, bottom=0.27, wspace=0.42)
    for ax, key, letter in zip(axes, ("nomargin", "nesy"), ("(a)", "(b)")):
        cm = cms[key]
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(7), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(7), names, fontsize=8)
        ax.set_xlabel("Kelas prediksi", fontsize=9)
        for r in range(7):
            for c in range(7):
                ax.text(c, r, f"{int(cm[r, c])}", ha="center", va="center", fontsize=8, color="white" if cm[r, c] > 0.5 * vmax else "black")
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.06, letter, ha="center", va="center", fontsize=12, family="serif")
    axes[0].set_ylabel("Kelas sebenarnya (ground truth)", fontsize=9)  # shared by both panels
    cax = fig.add_axes((0.925, 0.27, 0.015, 0.70))
    fig.colorbar(image, cax=cax).set_label("Jumlah deteksi", fontsize=9)
    fig.savefig(OUT / "cm_nesy_margin_compare.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    hybrid, train_config, device = load_models()
    dataset = load_test_set(train_config)
    records = load_records(hybrid, dataset, device)
    cms = check_against_text(records)
    print("text numbers reproduced")
    detection_figure(records, dataset)
    rows_fp, rows_fn = evidence_rows(records, hybrid, dataset, device)
    print("weakest nodes (b):", [(r["index"], f"N{r['node']}", r["depth"]) for r in rows_fn])
    evidence_figure(rows_fp, "fp", dataset, "fig_fp_ditolak.png")
    evidence_figure(rows_fn, "fn", dataset, "fig_fn_terlewat.png")
    root_figure(root_confidence(records, hybrid.symbolic_tree), root_split_counts(hybrid, dataset, device))
    confusion_figure(cms)


if __name__ == "__main__":
    main()
