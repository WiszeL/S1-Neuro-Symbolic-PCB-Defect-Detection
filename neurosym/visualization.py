from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from util.visualization import image_to_array


def heatmap_to_array(heatmap: torch.Tensor) -> np.ndarray:
    arr = heatmap.detach().cpu().numpy()
    arr_norm = np.clip(arr, 0.0, 1.0)
    if arr_norm.max() > 0:
        arr_norm = arr_norm / arr_norm.max()

    rgba = plt.get_cmap("jet")(arr_norm)

    # Hide noise, keep hot spots bright.
    alpha = np.where(arr_norm > 0.15, arr_norm, 0.0)
    rgba[..., 3] = alpha
    return rgba


def zoom_axis_to_box(
    axis: plt.Axes,
    box: torch.Tensor,
    image_shape: tuple[int, int],
    padding_ratio: float = 0.2,
    minimum_crop_size: int = 48,
) -> None:
    image_height, image_width = image_shape[0], image_shape[1]
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    crop_width = max(box_width * (1.0 + 2.0 * padding_ratio), float(minimum_crop_size))
    crop_height = max(
        box_height * (1.0 + 2.0 * padding_ratio), float(minimum_crop_size)
    )
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    left = max(center_x - crop_width / 2.0, 0.0)
    right = min(center_x + crop_width / 2.0, float(image_width))
    top = max(center_y - crop_height / 2.0, 0.0)
    bottom = min(center_y + crop_height / 2.0, float(image_height))

    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)


def draw_numbered_detections(
    axis: plt.Axes,
    image_tensor: torch.Tensor,
    detection_result: dict[str, torch.Tensor],
    detection_indices: list[int],
    class_names: tuple[str, ...],
    selected_index: int | None = None,
    display_numbers: list[int] | None = None,
) -> None:
    axis.imshow(image_to_array(image_tensor))
    # Dimmer on its own layer so heat colors stay true.
    dimmer = np.zeros(
        (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
    )
    dimmer[..., 3] = 0.35
    axis.imshow(dimmer)
    axis.axis("off")

    if display_numbers is None:
        display_numbers = list(range(1, len(detection_indices) + 1))

    for display_number, detection_index in zip(display_numbers, detection_indices):
        box = detection_result["boxes"][detection_index].detach().cpu()
        label = int(detection_result["labels"][detection_index])
        score = float(detection_result["scores"][detection_index])
        x1, y1, x2, y2 = box.tolist()
        edge_color = "lime" if detection_index == selected_index else "red"
        line_width = 3 if detection_index == selected_index else 2
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=line_width,
                edgecolor=edge_color,
                facecolor="none",
                clip_on=True,
            )
        )
        label_name = class_names[label - 1]
        axis.text(
            x1,
            max(y1 - 5, 0),
            f"#{display_number} {label_name} {score:.2f}",
            color="black",
            fontsize=9,
            weight="bold",
            clip_on=True,
            bbox={"facecolor": "yellow", "edgecolor": edge_color, "pad": 2},
        )


def draw_ground_truth_boxes(
    axis: plt.Axes,
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: tuple[str, ...],
) -> None:
    """Truth boxes in green dashes, unlike detections."""
    axis.imshow(image_to_array(image_tensor))
    # Match detection panels.
    dimmer = np.zeros(
        (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
    )
    dimmer[..., 3] = 0.15
    axis.imshow(dimmer)
    axis.axis("off")

    palette = [
        "#2a9d8f",
        "#e76f51",
        "#264653",
        "#f4a261",
        "#6a4c93",
        "#bc6c25",
        "#3a86ff",
        "#d62828",
    ]

    for i, (box, label) in enumerate(zip(gt_boxes.tolist(), gt_labels.tolist())):
        x1, y1, x2, y2 = box
        class_index = int(label) - 1
        color = palette[class_index % len(palette)]
        label_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"class {int(label)}"
        )
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                clip_on=True,
            )
        )
        axis.text(
            x1,
            max(y1 - 5, 0),
            f"GT: {label_name}",
            color="white",
            fontsize=9,
            weight="bold",
            clip_on=True,
            bbox={"facecolor": color, "edgecolor": color, "pad": 2, "alpha": 0.85},
        )


def lookup_ground_truth(
    dataset: Any,
    image_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Truth boxes for one image, by filename; None if missing."""
    # Match on filename stem, not extension.
    search_stem = Path(image_name).stem

    for sample in dataset.samples:
        if sample.image_path.stem == search_stem:
            return sample.boxes, sample.labels

    return None


_SODT_HEATMAP_FOOTNOTE = (
    "Positions are exact (RoI-Align's own coefficients); each cell's value pools a receptive "
    "field far larger than the box (~360x350px vs ~33x29px mean proposal) — region-level, "
    "not pixel-level.  Color: red=dominant, yellow=large, green=~half, cyan=small, "
    "dark=no contribution (maps are non-negative, each panel normalized to its own peak — "
    "don't compare brightness across panels).  Sign is not shown — a node's LEFT/RIGHT "
    "direction is in its title, not the color."
)


def _is_ancestor(ancestor: int, descendant: int) -> bool:
    """Heap-indexed tree: ancestry is a parent walk up from `descendant`."""
    while descendant > ancestor:
        descendant = (descendant - 1) // 2
    return descendant == ancestor


class _TreeLayout(NamedTuple):
    active_nodes: dict[int, dict[str, Any]]
    num_internal: int
    leaf_node: int
    visible_edges: list[tuple[int, int, str]]
    leaf_override: dict[int, str]
    coords: dict[int, tuple[float, float]]


_TREE_Y_STEP = 2.0  # row height


def _pruned_tree_layout(
    explanation: dict[str, Any],
    symbolic_tree: Any,
    class_names: tuple[str, ...],
) -> _TreeLayout:
    """The whole tree, top to bottom. Dead subtrees collapse into one leaf so only live branches take room."""
    active_nodes = {n["node_index"]: n for n in explanation["node_explanations"]}
    num_internal = (2**symbolic_tree.max_depth) - 1
    leaf_node = explanation["symbolic_leaf_index"] + num_internal

    # Dead subtree: nothing in it can change the answer, so draw it as one leaf.
    pruned_nodes: set[int] = set()
    for idx in range(num_internal - 1, -1, -1):
        left, right = idx * 2 + 1, idx * 2 + 2
        if (
            np.all(symbolic_tree.node_weights[idx] == 0.0)
            and symbolic_tree.node_bias[idx] == 0.0
            and (left >= num_internal or left in pruned_nodes)
            and (right >= num_internal or right in pruned_nodes)
        ):
            pruned_nodes.add(idx)

    visible_edges: list[tuple[int, int, str]] = []
    leaf_override: dict[int, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    next_slot = 0

    def build(node: int, depth: int) -> float:
        nonlocal next_slot
        if node in pruned_nodes:
            # Collapsed pure subtree shows one label.
            curr = node
            while curr < num_internal:
                curr = curr * 2 + 1
            label_idx = int(symbolic_tree.leaf_labels[curr - num_internal])
            # Tree names include background (7); detector names don't (6) — index carefully.
            if symbolic_tree.class_names is not None:
                leaf_override[node] = symbolic_tree.class_names[label_idx]
            else:
                leaf_override[node] = class_names[label_idx - 1]
        if node >= num_internal or node in pruned_nodes:
            x = float(next_slot)
            next_slot += 1
        else:
            left, right = node * 2 + 1, node * 2 + 2
            visible_edges.append((node, left, "left"))
            visible_edges.append((node, right, "right"))
            x = (build(left, depth + 1) + build(right, depth + 1)) / 2.0
        coords[node] = (x, -depth * _TREE_Y_STEP)
        return x

    build(0, 0)

    return _TreeLayout(
        active_nodes=active_nodes,
        num_internal=num_internal,
        leaf_node=leaf_node,
        visible_edges=visible_edges,
        leaf_override=leaf_override,
        coords=coords,
    )


def _union_box(proposal_box: torch.Tensor, detection_box: torch.Tensor) -> torch.Tensor:
    """Smallest box containing both — frames the zoom so edge heat survives."""
    return torch.tensor(
        [
            min(proposal_box[0], detection_box[0]),
            min(proposal_box[1], detection_box[1]),
            max(proposal_box[2], detection_box[2]),
            max(proposal_box[3], detection_box[3]),
        ]
    )


def _draw_heatmap_panel(
    axis: plt.Axes,
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    proposal_box: torch.Tensor,
    detection_box: torch.Tensor,
) -> None:
    """Image + dimmer + heatmap + both boxes + zoom, off — one node panel's worth.

    Shared by the matplotlib per-node loop, the path panel, and the HTML
    exporter, so the three can't quietly drift out of matching styles.
    """
    axis.imshow(image_to_array(image_tensor))
    dimmer = np.zeros((image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32)
    dimmer[..., 3] = 0.4
    axis.imshow(dimmer)
    axis.imshow(heatmap_to_array(heatmap), cmap="jet", vmin=0.0, vmax=1.0)

    x1, y1, x2, y2 = proposal_box.tolist()
    axis.add_patch(
        patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="cyan", facecolor="none"
        )
    )
    dx1, dy1, dx2, dy2 = detection_box.tolist()
    axis.add_patch(
        patches.Rectangle(
            (dx1, dy1), dx2 - dx1, dy2 - dy1,
            linewidth=1.5, edgecolor="lime", linestyle="--", facecolor="none",
        )
    )
    zoom_axis_to_box(axis, _union_box(proposal_box, detection_box), tuple(image_tensor.shape[-2:]))
    axis.axis("off")


def _node_margin(score: float) -> float:
    """How sure a node's call was: 0.5 = coin flip, 1.0 = certain."""
    return float(1.0 / (1.0 + np.exp(-abs(score))))


def _draw_full_tree(
    axis: plt.Axes, layout: _TreeLayout, label_name: str
) -> None:
    active_nodes, num_internal, leaf_node = layout.active_nodes, layout.num_internal, layout.leaf_node
    coords, leaf_override = layout.coords, layout.leaf_override

    for parent, child, side in layout.visible_edges:
        child_active = parent in active_nodes and _is_ancestor(child, leaf_node)
        color = (
            "#388e3c" if (child_active and side == "left")
            else ("#d32f2f" if (child_active and side == "right") else "#e0e0e0")
        )
        axis.plot(
            [coords[parent][0], coords[child][0]],
            [coords[parent][1], coords[child][1]],
            color=color, lw=3 if child_active else 1, zorder=2 if child_active else 1,
        )

    for node in layout.coords:
        x, y = coords[node]
        is_leaf = (node >= num_internal) or (node in leaf_override)
        is_active = (
            (node in active_nodes)
            or (node == leaf_node)
            or (node in leaf_override and _is_ancestor(node, leaf_node))
        )
        ec = "black" if is_active else "#9e9e9e"
        lw = 2 if is_active else 1
        font_size = 10 if is_active else 8
        weight = "bold" if is_active else "normal"

        if is_leaf:
            if node in leaf_override:
                label = leaf_override[node].replace("__background__", "bg")
            else:
                label = label_name if node == leaf_node else f"L{node - num_internal}"
            # A leaf is never on `active_nodes` (only internal path nodes carry a
            # score), so "landed on this leaf" and `is_active` agree.
            fc = "#4caf50" if is_active else "#f5f5f5"
            if is_active:
                ec = "#1b5e20"
            axis.text(
                x, y, label, ha="center", va="center", fontsize=font_size, weight=weight,
                bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw, alpha=1.0 if is_active else 0.6),
                zorder=3, rotation=90, rotation_mode="anchor",
            )
        else:
            label = f"{active_nodes[node]['score']:.2f}" if node in active_nodes else f"N{node}"
            axis.text(
                x, y, label, ha="center", va="center", fontsize=font_size, weight=weight,
                bbox=dict(
                    boxstyle="circle,pad=0.2", fc="#e1f5fe" if is_active else "#f5f5f5",
                    ec=ec, lw=lw, alpha=1.0 if is_active else 0.6,
                ),
                zorder=3,
            )

    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    axis.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    axis.set_ylim(min(ys) - 1.5, 1.5)
    axis.axis("off")


def _draw_node_split_panel(
    axis: plt.Axes, node: dict[str, Any], symbolic_tree: Any, feature_vector: np.ndarray
) -> float:
    """One node's real split plane; returns its margin factor σ.

    Axes are the node's own score split in two: a = evidence for LEFT (Σ w>0 · x),
    c = evidence for RIGHT (-Σ w<0 · x), so f(x) = a - c + b exactly. The split
    f = 0 is the line c = a + b; the dot is the RoI's real (a, c); its
    perpendicular distance to the line is |f| / √2. Shading = σ(|f|) at each point.
    """
    index = node["node_index"]
    weights = symbolic_tree.node_weights[index]
    axis.set_title(f"N{index} {node['decision'].upper()}", fontsize=9, weight="bold", pad=3)
    if not np.any(weights != 0.0):
        axis.text(0.5, 0.5, "pruned, skipped", ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        return 1.0

    bias = float(symbolic_tree.node_bias[index])
    contributions = weights * feature_vector
    a = float(contributions[weights > 0].sum())
    c = float(-contributions[weights < 0].sum())
    score = float(node["score"])
    margin = _node_margin(score)

    lo = min(0.0, a, c)
    hi = 1.15 * max(a, c, abs(bias), 1e-6)
    grid = np.linspace(lo, hi, 200)
    f_grid = grid[None, :] - grid[:, None] + bias  # rows = c, cols = a
    sigma = 1.0 / (1.0 + np.exp(-np.abs(f_grid)))
    # Each half in its branch color (matches the tree edges): light at the split, saturated far from it.
    for cmap, half in (("Greens", f_grid >= 0.0), ("Reds", f_grid < 0.0)):
        axis.imshow(
            np.ma.masked_where(~half, sigma), extent=(lo, hi, lo, hi), origin="lower",
            cmap=cmap, vmin=0.5, vmax=1.15, aspect="equal", zorder=0,
        )
    span = hi - lo
    # Two candidate corners per label; take the one farther from the dot so it never hides behind it.
    for text, spots in (
        ("LEFT branch", ((0.5, 0.06), (0.3, 0.06))),
        ("RIGHT branch", ((0.06, 0.9), (0.06, 0.6))),
    ):
        x, y = max(spots, key=lambda sp: (lo + sp[0] * span - a) ** 2 + (lo + sp[1] * span - c) ** 2)
        axis.text(lo + x * span, lo + y * span, text, fontsize=7, weight="bold", color="white", zorder=5)
    axis.plot(grid, grid + bias, color="black", lw=1.5, zorder=2)

    foot = ((a + c - bias) / 2.0, (a + c + bias) / 2.0)  # closest point on the split line
    axis.plot([a, foot[0]], [c, foot[1]], color="#f57c00", lw=1.5, ls="--", zorder=3)
    axis.plot(a, c, "o", color="#fdd835", mec="black", ms=8, zorder=4)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=7)
    axis.set_xlabel(f"evidence LEFT   |f|={abs(score):.2f}  σ={margin:.4f}", fontsize=8, labelpad=2)
    axis.set_ylabel("evidence RIGHT", fontsize=8, labelpad=2)
    return margin


def _draw_routing_margin_panels(
    fig: plt.Figure,
    spec: Any,
    explanation: dict[str, Any],
    symbolic_tree: Any,
    feature_vector: np.ndarray,
) -> None:
    """One 2D split plane per visited node, in tree-depth rows. Πσ is the detection score."""
    rows = gridspec.GridSpecFromSubplotSpec(symbolic_tree.max_depth, 1, subplot_spec=spec, hspace=0.55)
    confidence = 1.0
    for node in explanation["node_explanations"]:
        confidence *= _draw_node_split_panel(
            fig.add_subplot(rows[node["depth"]]), node, symbolic_tree, feature_vector
        )
    box = spec.get_position(fig)
    fig.text(
        (box.x0 + box.x1) / 2, box.y1 + 0.025,
        f"Routing margin per node\n"
        f"x: evidence LEFT (Σw⁺x)   y: evidence RIGHT (−Σw⁻x)\n"
        f"Πσ = {confidence:.4f}   detection score = {explanation['score']:.4f}",
        ha="center", va="bottom", fontsize=10, weight="bold",
    )


def draw_neurosymbolic_explanation(
    image_tensor: torch.Tensor,
    detection_result: dict[str, torch.Tensor],
    detection_index: int,
    explanation: dict[str, Any],
    class_names: tuple[str, ...],
    symbolic_tree: Any,
    selected_number: int = 1,
    extra_panel_func: Any = None,
    show_path_panel: bool = False,
) -> None:
    """Two figures: the full tree with per-node routing margins, then the heatmaps."""
    label_name = class_names[explanation["label"] - 1]
    node_count = len(explanation["node_explanations"])
    layout = _pruned_tree_layout(explanation, symbolic_tree, class_names)

    # ── Figure 1: full tree (path highlighted) + routing margin per node, rows shared ──
    slots = max(x for x, _ in layout.coords.values()) + 1.0
    tree_width = max(9.0, 0.45 * slots)
    fig_tree = plt.figure(figsize=(tree_width + 4.5, 2.4 * symbolic_tree.max_depth + 1.5))
    gs_tree = gridspec.GridSpec(1, 2, width_ratios=[tree_width, 4.5], wspace=0.04, left=0.02, right=0.98, bottom=0.03, top=0.93, figure=fig_tree)
    ax_tree = fig_tree.add_subplot(gs_tree[0])
    _draw_full_tree(ax_tree, layout, label_name)
    ax_tree.set_title(
        f"SODT tree — #{selected_number} {label_name} {explanation['score']:.2f}   "
        "(green=went left, red=went right, node label = its score, grey = not visited)",
        fontsize=13, weight="bold",
    )
    # Same vector explain_hybrid_detection routed on, so each node's (a, c) is its real score.
    feature_vector = detection_result["pooled_features"][detection_index].detach().cpu().numpy().reshape(-1)
    _draw_routing_margin_panels(fig_tree, gs_tree[1], explanation, symbolic_tree, feature_vector)

    # ── Figure 2: per-node heatmaps (+ Grad-CAM) ──
    has_exact = "projected_exact_attribution_on_proposal_box" in explanation["node_explanations"][0]
    show_path_panel = show_path_panel and has_exact
    columns = 2 if extra_panel_func is not None else 1
    fig_maps = plt.figure(figsize=(7 * columns, max(8, 4 * (node_count + (1 if show_path_panel else 0)))))
    gs = gridspec.GridSpec(1, columns, wspace=0.1, figure=fig_maps)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        node_count + (1 if show_path_panel else 0), 1, subplot_spec=gs[0], hspace=0.4
    )

    proposal_box = explanation["proposal_box"]
    detection_box = explanation["detection_box"]

    panel_offset = 0
    if show_path_panel:
        axis = fig_maps.add_subplot(gs_right[0])
        _draw_heatmap_panel(
            axis, image_tensor,
            explanation["projected_path_exact_attribution_on_proposal_box"],
            proposal_box, detection_box,
        )
        axis.set_title(
            f"SODT heatmap — per-node maps stacked, Σ|node map| ({node_count} nodes, "
            f"{explanation.get('exact_attribution_level', '?')})\n"
            "cyan=proposal (heat frame)  lime=detection box  "
            "— validated in Table A, not the per-step story"
        )
        panel_offset = 1

    heatmap_key = (
        "projected_exact_attribution_on_proposal_box" if has_exact
        else "projected_node_heatmap_on_proposal_box"
    )
    title_prefix = "SODT heatmap" if has_exact else "SODT heatmap (pooled-grid, no FPN context)"
    for axis_index, node in enumerate(explanation["node_explanations"]):
        axis = fig_maps.add_subplot(gs_right[axis_index + panel_offset])
        _draw_heatmap_panel(axis, image_tensor, node[heatmap_key], proposal_box, detection_box)
        axis.set_title(
            f"{title_prefix} — Depth {node['depth'] + 1} | Node {node['node_index']}\n"
            f"Score: {node['score']:.2f} -> Went {node['decision'].upper()}  "
            f"margin σ={_node_margin(node['score']):.4f} "
            f"(evidence {node.get('positive_evidence_sum', 0.0):.2f})"
        )

    if extra_panel_func is not None:
        extra_panel_func(fig_maps.add_subplot(gs[1]))

    if has_exact:
        fig_maps.text(0.5, 0.0, _SODT_HEATMAP_FOOTNOTE, ha="center", va="bottom", fontsize=7, wrap=True)

    plt.show()
    plt.close("all")
