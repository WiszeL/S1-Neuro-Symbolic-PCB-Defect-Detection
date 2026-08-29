from __future__ import annotations

from pathlib import Path
from typing import Any

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

    # Remove noise entirely, but make the active area VERY opaque and bright
    # arr_norm ** 2.0 strongly suppresses background noise to yield sharp heatmaps.
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
    # Add a separate black dimmer layer over the image so it doesn't corrupt the heatmap colors
    dimmer = np.zeros(
        (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
    )
    dimmer[..., 3] = 0.35  # 40% perfect black dimmer
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
    """Draw ground truth bounding boxes on the given axes.

    Uses a distinctive green-dashed style so they are clearly
    distinguishable from model detections.
    """
    axis.imshow(image_to_array(image_tensor))
    # Add a slight dimmer to keep consistency with detection panels
    dimmer = np.zeros(
        (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
    )
    dimmer[..., 3] = 0.15  # Light dimmer for ground truth panel
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
    """Look up ground truth boxes and labels from a PCBDataset by image filename.

    Parameters
    ----------
    dataset : PCBDataset
        The dataset whose ``.samples`` list will be searched.
    image_name : str
        The uploaded image filename (e.g. ``"00041200_test.jpg"``).

    Returns
    -------
    tuple[Tensor, Tensor] | None
        ``(boxes, labels)`` if a match is found, otherwise ``None``.
    """
    # Normalise the search key: strip extension, compare stems
    search_stem = Path(image_name).stem

    for sample in dataset.samples:
        if sample.image_path.stem == search_stem:
            return sample.boxes, sample.labels

    return None


def draw_neurosymbolic_explanation(
    image_tensor: torch.Tensor,
    detection_result: dict[str, torch.Tensor],
    detection_index: int,
    explanation: dict[str, Any],
    class_names: tuple[str, ...],
    symbolic_tree: Any,
    selected_number: int = 1,
    extra_panel_func: Any = None,
) -> None:
    label_name = class_names[explanation["label"] - 1]

    node_count = len(explanation["node_explanations"])

    if extra_panel_func is not None:
        plt.figure(figsize=(18, max(8, 4 * node_count)))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.1)
    else:
        plt.figure(figsize=(15, max(8, 4 * node_count)))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)

    # 1. LEFT SIDE: Pruned SODT Tree
    ax_tree = plt.subplot(gs[0])
    ax_tree.axis("off")
    ax_tree.set_title("Pruned SODT Tree", fontsize=16, weight="bold")

    active_nodes = {n["node_index"]: n for n in explanation["node_explanations"]}
    tree_depth = symbolic_tree.max_depth
    num_internal = (2**tree_depth) - 1

    leaf_node = explanation["symbolic_leaf_index"] + num_internal

    # Identify structurally pruned internal nodes (weights and bias are zero)
    pruned_nodes = set()
    for idx in range(num_internal):
        if (
            np.all(symbolic_tree.node_weights[idx] == 0.0)
            and symbolic_tree.node_bias[idx] == 0.0
        ):
            pruned_nodes.add(idx)

    # Build the structurally visible pruned tree
    visible_nodes = set()
    visible_edges = []
    leaf_override = {}

    def build_pruned_tree(node, depth):
        visible_nodes.add(node)
        if node >= num_internal:
            return
        if node in pruned_nodes:
            # Find leftmost leaf class index in the pure collapsed subtree
            curr = node
            while curr < num_internal:
                curr = curr * 2 + 1
            leaf_offset = curr - num_internal
            label_idx = int(symbolic_tree.leaf_labels[leaf_offset])
            # leaf_labels are 0-indexed INCLUDING background — index the
            # tree's own (7-name) class_names directly when available.
            # class_names passed in here is the detector's defect-only
            # (6-name) tuple, 1-indexed against label_idx; using it directly
            # sends label_idx=0 (background) to class_names[-1].
            if symbolic_tree.class_names is not None:
                leaf_override[node] = symbolic_tree.class_names[label_idx]
            else:
                leaf_override[node] = class_names[label_idx - 1]
            return

        left = node * 2 + 1
        right = node * 2 + 2
        visible_edges.append((node, left, "left"))
        visible_edges.append((node, right, "right"))
        build_pruned_tree(left, depth + 1)
        build_pruned_tree(right, depth + 1)

    build_pruned_tree(0, 0)

    # --- NEW PRETTY LAYOUT ALGORITHM ---
    # 1. Build parent-child relationships
    children_map = {node: [] for node in visible_nodes}
    for p, c, _ in visible_edges:
        children_map[p].append(c)

    coords = {}
    leaf_x_counter = [0]

    def assign_coords(node, depth):
        # Sort children to maintain left-to-right order (left child index < right child index)
        children = sorted(children_map[node])
        if not children:
            # Terminal node (spaced evenly)
            coords[node] = (leaf_x_counter[0], -depth * 2.5)
            leaf_x_counter[0] += 40  # Give plenty of horizontal spacing between leaves
        else:
            child_x_sum = 0
            for c in children:
                assign_coords(c, depth + 1)
                child_x_sum += coords[c][0]
            coords[node] = (child_x_sum / len(children), -depth * 2.5)

    assign_coords(0, 0)

    # 2. Center the tree around x=0
    min_x_coord = min(x for x, y in coords.values())
    max_x_coord = max(x for x, y in coords.values())
    center_offset = (max_x_coord + min_x_coord) / 2.0
    for node in coords:
        x, y = coords[node]
        coords[node] = (x - center_offset, y)

    # Helper to check if ancestor reaches leaf_node
    def is_ancestor(ancestor, descendent):
        if ancestor == descendent:
            return True
        if ancestor >= num_internal:
            return False
        return is_ancestor(ancestor * 2 + 1, descendent) or is_ancestor(
            ancestor * 2 + 2, descendent
        )

    for parent, child, side in visible_edges:
        is_active_parent = parent in active_nodes
        child_active = is_active_parent and is_ancestor(child, leaf_node)

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
            or (node in leaf_override and is_ancestor(node, leaf_node))
        )

        alpha = 1.0 if is_active else 0.5  # Slightly more opaque for unselected nodes
        fc = "#e1f5fe" if not is_leaf else "#c8e6c9"
        ec = "black" if is_active else "#9e9e9e"  # Slightly darker grey edge
        if not is_active:
            fc = "#f5f5f5"
        lw = 2 if is_active else 1

        # --- MAKE UNSELECTED NODES BIGGER ---
        internal_font_active = 11
        internal_font_inactive = 9
        leaf_font_active = 11
        leaf_font_inactive = 9

        if is_leaf:
            if node in leaf_override:
                label = leaf_override[node]
            else:
                label = (
                    f"{label_name}" if node == leaf_node else f"L{node - num_internal}"
                )

            leaf_active = (node == leaf_node) or (
                node in leaf_override and is_ancestor(node, leaf_node)
            )
            if leaf_active:
                fc = "#4caf50"
                ec = "#1b5e20"
            bbox = dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=lw, alpha=alpha)
            ax_tree.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=leaf_font_active if is_active else leaf_font_inactive,
                weight="bold" if is_active else "normal",
                bbox=bbox,
                zorder=3,
                rotation=90,
                rotation_mode="anchor",
            )
        else:
            label = (
                f"{active_nodes[node]['score']:.2f}"
                if node in active_nodes
                else f"N{node}"
            )
            bbox = dict(boxstyle="circle,pad=0.2", fc=fc, ec=ec, lw=lw, alpha=alpha)
            ax_tree.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=internal_font_active if is_active else internal_font_inactive,
                weight="bold" if is_active else "normal",
                bbox=bbox,
                zorder=3,
            )

    # Dynamically scale axis limits so the text never overlaps the edge
    min_final_x = min(x for x, y in coords.values())
    max_final_x = max(x for x, y in coords.values())
    ax_tree.set_xlim(min_final_x - 8, max_final_x + 8)
    ax_tree.set_ylim(-tree_depth * 2.5 - 1.0, 1.0)

    # 2. RIGHT SIDE: Vertical Heatmaps
    has_exact = (
        "projected_exact_attribution_on_proposal_box"
        in explanation["node_explanations"][0]
    )
    panel_count = node_count + (1 if has_exact else 0)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        panel_count, 1, subplot_spec=gs[1], hspace=0.4
    )

    proposal_box = explanation["proposal_box"]
    detection_box = explanation["detection_box"]
    # The heat is projected onto proposal_box (RoI-Align pooled from it) — draw
    # that as the primary frame and zoom to the union of both boxes, so heat
    # near a proposal edge is never cropped out and the refinement is still
    # visible. See neurosym/inference.py::explain_hybrid_detection.
    union_box = torch.tensor(
        [
            min(proposal_box[0], detection_box[0]),
            min(proposal_box[1], detection_box[1]),
            max(proposal_box[2], detection_box[2]),
            max(proposal_box[3], detection_box[3]),
        ]
    )

    def _draw_boxes(axis) -> None:
        x1, y1, x2, y2 = proposal_box.tolist()
        axis.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="cyan", facecolor="none",
            )
        )
        dx1, dy1, dx2, dy2 = detection_box.tolist()
        axis.add_patch(
            patches.Rectangle(
                (dx1, dy1), dx2 - dx1, dy2 - dy1,
                linewidth=1.5, edgecolor="lime", linestyle="--", facecolor="none",
            )
        )

    panel_offset = 0
    if has_exact:
        axis = plt.subplot(gs_right[0])
        axis.imshow(image_to_array(image_tensor))
        dimmer = np.zeros(
            (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
        )
        dimmer[..., 3] = 0.4
        axis.imshow(dimmer)
        axis.imshow(
            heatmap_to_array(
                explanation["projected_path_exact_attribution_on_proposal_box"]
            ),
            cmap="jet", vmin=0.0, vmax=1.0,
        )
        _draw_boxes(axis)
        zoom_axis_to_box(axis, union_box, tuple(image_tensor.shape[-2:]))
        axis.set_title(
            f"Exact path attribution ({node_count} nodes, "
            f"{explanation.get('exact_attribution_level', '?')})\n"
            "cyan=proposal (heat frame)  lime=detection box"
        )
        axis.axis("off")
        panel_offset = 1

    for axis_index, node in enumerate(explanation["node_explanations"]):
        axis = plt.subplot(gs_right[axis_index + panel_offset])
        axis.imshow(image_to_array(image_tensor))
        # Add a separate black dimmer layer over the image so it doesn't corrupt the heatmap colors
        dimmer = np.zeros(
            (image_tensor.shape[-2], image_tensor.shape[-1], 4), dtype=np.float32
        )
        dimmer[..., 3] = 0.4  # 40% perfect black dimmer
        axis.imshow(dimmer)
        heatmap_key = (
            "projected_exact_attribution_on_proposal_box"
            if has_exact
            else "projected_node_heatmap_on_proposal_box"
        )
        axis.imshow(
            heatmap_to_array(node[heatmap_key]),
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        _draw_boxes(axis)
        zoom_axis_to_box(axis, union_box, tuple(image_tensor.shape[-2:]))
        evidence = node.get("positive_evidence_sum", 0.0)
        title_prefix = (
            "Exact attribution" if has_exact else "Pooled-grid evidence"
        )
        axis.set_title(
            f"{title_prefix} — Depth {node['depth'] + 1} | Node {node['node_index']}\n"
            f"Score: {node['score']:.2f} -> Went {node['decision'].upper()} "
            f"(evidence {evidence:.2f})"
        )
        axis.axis("off")

    if extra_panel_func is not None:
        ax_extra = plt.subplot(gs[2])
        extra_panel_func(ax_extra)

    plt.show()
    plt.close("all")
