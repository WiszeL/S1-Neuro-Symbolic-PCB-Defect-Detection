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
    visible_nodes: set[int]
    visible_edges: list[tuple[int, int, str]]
    leaf_override: dict[int, str]
    coords: dict[int, tuple[float, float]]


def _pruned_tree_layout(
    explanation: dict[str, Any],
    symbolic_tree: Any,
    class_names: tuple[str, ...],
) -> _TreeLayout:
    """Path + immediate off-path siblings, laid out top-to-bottom.

    Path stays in a fixed column, one dangling leaf per level at a fixed
    offset — spacing never shrinks with depth (see `coords` below).
    """
    active_nodes = {n["node_index"]: n for n in explanation["node_explanations"]}
    tree_depth = symbolic_tree.max_depth
    num_internal = (2**tree_depth) - 1
    leaf_node = explanation["symbolic_leaf_index"] + num_internal

    # Pruned = all-zero weights and bias.
    pruned_nodes = set()
    for idx in range(num_internal):
        if (
            np.all(symbolic_tree.node_weights[idx] == 0.0)
            and symbolic_tree.node_bias[idx] == 0.0
        ):
            pruned_nodes.add(idx)

    visible_nodes = set()
    visible_edges = []
    leaf_override = {}
    path_node_indices = set(active_nodes.keys())

    def build_pruned_tree(node):
        visible_nodes.add(node)
        if node >= num_internal:
            return
        if node in pruned_nodes:
            # Collapsed pure subtree shows one label.
            curr = node
            while curr < num_internal:
                curr = curr * 2 + 1
            leaf_offset = curr - num_internal
            label_idx = int(symbolic_tree.leaf_labels[leaf_offset])
            # Tree names include background (7); detector names don't (6) — index carefully.
            if symbolic_tree.class_names is not None:
                leaf_override[node] = symbolic_tree.class_names[label_idx]
            else:
                leaf_override[node] = class_names[label_idx - 1]
            return
        if node not in path_node_indices:
            # Off-path branch, not pruned but not walked either — collapse
            # without recursing, so it doesn't drag its own subtree in.
            leaf_override[node] = "…"
            return

        left = node * 2 + 1
        right = node * 2 + 2
        visible_edges.append((node, left, "left"))
        visible_edges.append((node, right, "right"))
        build_pruned_tree(left)
        build_pruned_tree(right)

    build_pruned_tree(0)

    # This tree is always a straight line with one dangling leaf per level
    # (never two live subtrees) — a generic "x = average of children" layout
    # squeezes that shape, since a deep node's position keeps averaging back
    # toward its own narrowing subtree. Keep the path in a fixed column
    # instead and hang each level's dangling side at a fixed offset, so
    # spacing never shrinks no matter how deep the path goes.
    coords: dict[int, tuple[float, float]] = {}
    y_step, x_step = 2.5, 40.0
    node, depth = 0, 0
    while node < num_internal:
        coords[node] = (0.0, -depth * y_step)
        left, right = node * 2 + 1, node * 2 + 2
        taken = left if _is_ancestor(left, leaf_node) else right
        dangling = right if taken == left else left
        coords[dangling] = (
            -x_step if dangling == left else x_step,
            -(depth + 1) * y_step,
        )
        node, depth = taken, depth + 1
    coords[node] = (0.0, -depth * y_step)  # final leaf, still on the path

    return _TreeLayout(
        active_nodes=active_nodes,
        num_internal=num_internal,
        leaf_node=leaf_node,
        visible_nodes=visible_nodes,
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
    label_name = class_names[explanation["label"] - 1]

    node_count = len(explanation["node_explanations"])

    if extra_panel_func is not None:
        plt.figure(figsize=(18, max(8, 4 * node_count)))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.1)
    else:
        plt.figure(figsize=(15, max(8, 4 * node_count)))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)

    # Tree panel
    ax_tree = plt.subplot(gs[0])
    ax_tree.axis("off")
    ax_tree.set_title("Pruned SODT Tree", fontsize=16, weight="bold")

    layout = _pruned_tree_layout(explanation, symbolic_tree, class_names)
    active_nodes = layout.active_nodes
    num_internal = layout.num_internal
    leaf_node = layout.leaf_node
    visible_nodes = layout.visible_nodes
    visible_edges = layout.visible_edges
    leaf_override = layout.leaf_override
    coords = layout.coords

    for parent, child, side in visible_edges:
        is_active_parent = parent in active_nodes
        child_active = is_active_parent and _is_ancestor(child, leaf_node)

        color = (
            "#388e3c"
            if (child_active and side == "left")
            else ("#d32f2f" if (child_active and side == "right") else "#e0e0e0")
        )
        lw = 3 if child_active else 1
        zorder = 2 if child_active else 1
        ax_tree.plot(
            [coords[parent][0], coords[child][0]],
            [coords[parent][1], coords[child][1]],
            color=color,
            lw=lw,
            zorder=zorder,
        )

    for node in visible_nodes:
        x, y = coords[node]
        is_leaf = (node >= num_internal) or (node in leaf_override)
        is_active = (
            (node in active_nodes)
            or (node == leaf_node)
            or (node in leaf_override and _is_ancestor(node, leaf_node))
        )

        alpha = 1.0 if is_active else 0.5
        ec = "black" if is_active else "#9e9e9e"
        lw = 2 if is_active else 1
        font_size = 11 if is_active else 9

        if is_leaf:
            if node in leaf_override:
                label = leaf_override[node]
            else:
                label = (
                    f"{label_name}" if node == leaf_node else f"L{node - num_internal}"
                )

            # A leaf is never on `active_nodes` (only internal path nodes
            # carry a score), so "landed on this leaf" and `is_active` agree.
            fc = "#4caf50" if is_active else "#f5f5f5"
            if is_active:
                ec = "#1b5e20"
            bbox = dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw, alpha=alpha)
            ax_tree.text(
                x, y, label, ha="center", va="center",
                fontsize=font_size, weight="bold" if is_active else "normal",
                bbox=bbox, zorder=3, rotation=90, rotation_mode="anchor",
            )
        else:
            label = (
                f"{active_nodes[node]['score']:.2f}"
                if node in active_nodes
                else f"N{node}"
            )
            fc = "#e1f5fe" if is_active else "#f5f5f5"
            bbox = dict(boxstyle="circle,pad=0.2", fc=fc, ec=ec, lw=lw, alpha=alpha)
            ax_tree.text(
                x, y, label, ha="center", va="center",
                fontsize=font_size, weight="bold" if is_active else "normal",
                bbox=bbox, zorder=3,
            )

    # Fit to the tree actually drawn, not a fixed depth.
    min_final_x = min(x for x, y in coords.values())
    max_final_x = max(x for x, y in coords.values())
    min_final_y = min(y for x, y in coords.values())
    ax_tree.set_xlim(min_final_x - 8, max_final_x + 8)
    ax_tree.set_ylim(min_final_y - 1.0, 1.0)

    # Heatmap panel
    has_exact = (
        "projected_exact_attribution_on_proposal_box"
        in explanation["node_explanations"][0]
    )
    show_path_panel = show_path_panel and has_exact
    panel_count = node_count + (1 if show_path_panel else 0)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        panel_count, 1, subplot_spec=gs[1], hspace=0.4
    )

    proposal_box = explanation["proposal_box"]
    detection_box = explanation["detection_box"]

    panel_offset = 0
    if show_path_panel:
        axis = plt.subplot(gs_right[0])
        _draw_heatmap_panel(
            axis, image_tensor,
            explanation["projected_path_exact_attribution_on_proposal_box"],
            proposal_box, detection_box,
        )
        axis.set_title(
            f"SODT heatmap — full path total ({node_count} nodes, "
            f"{explanation.get('exact_attribution_level', '?')})\n"
            "cyan=proposal (heat frame)  lime=detection box  "
            "— validated in Table A, not the per-step story"
        )
        panel_offset = 1

    for axis_index, node in enumerate(explanation["node_explanations"]):
        axis = plt.subplot(gs_right[axis_index + panel_offset])
        heatmap_key = (
            "projected_exact_attribution_on_proposal_box"
            if has_exact
            else "projected_node_heatmap_on_proposal_box"
        )
        _draw_heatmap_panel(axis, image_tensor, node[heatmap_key], proposal_box, detection_box)
        evidence = node.get("positive_evidence_sum", 0.0)
        title_prefix = "SODT heatmap" if has_exact else "SODT heatmap (pooled-grid, no FPN context)"
        axis.set_title(
            f"{title_prefix} — Depth {node['depth'] + 1} | Node {node['node_index']}\n"
            f"Score: {node['score']:.2f} -> Went {node['decision'].upper()} "
            f"(evidence {evidence:.2f})"
        )

    if extra_panel_func is not None:
        ax_extra = plt.subplot(gs[2])
        extra_panel_func(ax_extra)

    if has_exact:
        plt.figtext(0.5, 0.0, _SODT_HEATMAP_FOOTNOTE, ha="center", va="bottom", fontsize=7, wrap=True)

    plt.show()
    plt.close("all")

