from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .dataset import open_exported_symbolic_array_payload
from .config import SymbolicTrainConfig
from .evaluation import evaluate_symbolic_model as evaluate_symbolic_metrics
from .evaluation import evaluate_symbolic_spatial_metrics
from .sodt import SparseObliqueDecisionTreeClassifier
from .tao import fit_tree_with_tao, postprocess_tree
from util.io import ensure_dir, save_json


def _format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_progress_header(
    feature_count: int,
    feature_dim: int,
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
    iterations: int,
    total_node_fit_upper_bound: int,
) -> list[str]:
    return [
        (
            f"Loaded {feature_count} symbolic samples with {feature_dim} raw RoI-pooled feature dimensions."
        ),
        (
            "TAO consumes the full RoI Align feature vector. Symbolic sparsity comes only from "
            "L1 regularization and the tree structure."
        ),
        (
            f"Training one configured SODT on the configured trainval RoI set: "
            f"depth={tree_depth}, lambda={l1_lambda:g}, alpha={sparsity_alpha:g}."
        ),
        (
            f"Workload check: {iterations} TAO iterations and up to "
            f"{total_node_fit_upper_bound} LIBLINEAR node fits."
        ),
        (
            "Each TAO iteration gets its own node-fit progress bar and remains visible after completion."
        ),
    ]


def _materialize_sodt_config(search_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "tree_depth": int(search_config["tree_depth"]),
        "iterations": int(search_config["iterations"]),
        "l1_lambda": float(search_config["l1_lambda"]),
        "sparsity_alpha": float(search_config["sparsity_alpha"]),
        "logistic_max_iter": int(search_config["logistic_max_iter"]),
        "tolerance": float(search_config["tolerance"]),
        "zero_threshold": float(search_config["zero_threshold"]),
        "random_state": int(search_config["random_state"]),
    }


def _load_symbolic_training_data(
    export_path: str | Path,
    random_state: int,
    neg_ratio: float | None = None,
) -> tuple[Any, Any]:
    bundle = open_exported_symbolic_array_payload(
        torch.load(export_path, map_location="cpu", weights_only=True),
        random_state=random_state,
        feature_dtype="float32",
        neg_ratio=neg_ratio,
    )

    feature_matrix = bundle.feature_vectors
    labels = bundle.teacher_labels.detach().cpu().numpy()
    return bundle, (feature_matrix, labels)


def _load_symbolic_evaluation_data(
    export_path: str | Path,
    random_state: int = 42,
    neg_ratio: float | None = None,
) -> tuple[Any, Any, Any]:
    bundle = open_exported_symbolic_array_payload(
        torch.load(export_path, map_location="cpu", weights_only=True),
        random_state=random_state,
        feature_dtype="float32",
        neg_ratio=neg_ratio,
    )
    feature_matrix = bundle.feature_vectors
    labels = bundle.teacher_labels.detach().cpu().numpy()
    return bundle, feature_matrix, labels


def _build_symbolic_tree(
    bundle: Any,
    feature_matrix: Any,
    tree_depth: int,
) -> SparseObliqueDecisionTreeClassifier:
    return SparseObliqueDecisionTreeClassifier(
        max_depth=tree_depth,
        num_classes=len(bundle.class_names),
        input_dim=feature_matrix.shape[1],
        original_input_dim=feature_matrix.shape[1],
        feature_shape=bundle.feature_shape,
        class_names=bundle.class_names,
    )


def _augment_tree_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    augmented = dict(metrics)
    nonzero_counts = tree.nonzero_weight_counts()
    augmented.update(
        {
            "tree_depth": int(tree.max_depth),
            "num_leaves": int(tree.num_leaves),
            "num_internal_nodes": int(tree.num_internal_nodes),
            "nonzero_weights": sum(nonzero_counts),
            "mean_nonzero_per_node": float(np.mean(nonzero_counts))
            if nonzero_counts
            else 0.0,
            "active_internal_nodes": sum(count > 0 for count in nonzero_counts),
            "mean_nonzero_per_active_node": float(
                sum(nonzero_counts) / max(sum(count > 0 for count in nonzero_counts), 1)
            ),
        }
    )
    return augmented


def _evaluate_symbolic_model(
    tree: SparseObliqueDecisionTreeClassifier,
    bundle: Any,
    feature_matrix: Any,
    labels: Any,
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
    history: list[dict[str, Any]],
    random_state: int,
) -> dict[str, Any]:
    symbolic_metrics = evaluate_symbolic_metrics(
        tree,
        feature_matrix=feature_matrix,
        teacher_labels=labels,
        class_names=bundle.class_names,
    )
    spatial_metrics = evaluate_symbolic_spatial_metrics(
        tree,
        feature_grids=bundle.feature_grids,
        proposal_boxes=bundle.proposal_boxes,
        matched_gt_boxes=bundle.matched_gt_boxes,
        has_matched_gt=bundle.has_matched_gt,
        gt_iou=bundle.gt_iou,
        random_state=random_state,
    )

    metrics = _augment_tree_metrics(
        tree,
        {
            **symbolic_metrics,
            **spatial_metrics,
        },
    )
    return {
        "tree_depth": int(tree_depth),
        "l1_lambda": float(l1_lambda),
        "sparsity_alpha": float(sparsity_alpha),
        "metrics": metrics,
        "history": history,
        "tree_state": tree.to_state_dict(),
    }


def _evaluate_trained_model_on_bundle(
    trained_model: dict[str, Any],
    bundle: Any,
    feature_matrix: Any,
    labels: Any,
    random_state: int,
) -> dict[str, Any]:
    tree = SparseObliqueDecisionTreeClassifier.from_state_dict(
        trained_model["tree_state"]
    )
    symbolic_metrics = evaluate_symbolic_metrics(
        tree,
        feature_matrix=feature_matrix,
        teacher_labels=labels,
        class_names=bundle.class_names,
    )
    spatial_metrics = evaluate_symbolic_spatial_metrics(
        tree,
        feature_grids=bundle.feature_grids,
        proposal_boxes=bundle.proposal_boxes,
        matched_gt_boxes=bundle.matched_gt_boxes,
        has_matched_gt=bundle.has_matched_gt,
        gt_iou=bundle.gt_iou,
        random_state=random_state,
    )

    metrics = _augment_tree_metrics(
        tree,
        {
            **symbolic_metrics,
            **spatial_metrics,
        },
    )
    return {
        "tree_depth": trained_model["tree_depth"],
        "l1_lambda": trained_model["l1_lambda"],
        "sparsity_alpha": trained_model["sparsity_alpha"],
        "metrics": metrics,
    }


def evaluate_heldout(
    export_path: str | Path,
    checkpoint_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    print(f"Loading heldout evaluation data from {export_path}...", flush=True)
    t0 = perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    bundle, feature_matrix, labels = _load_symbolic_evaluation_data(
        export_path,
        random_state=random_state,
    )
    print(
        f"Loaded {feature_matrix.shape[0]:,} samples in {_format_duration(perf_counter() - t0)}",
        flush=True,
    )

    print("Evaluating on held-out set...", flush=True)
    heldout_model = _evaluate_trained_model_on_bundle(
        trained_model=checkpoint,
        bundle=bundle,
        feature_matrix=feature_matrix,
        labels=labels,
        random_state=random_state,
    )

    return {
        "export_path": str(export_path),
        "sample_count": feature_matrix.shape[0],
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "tree_depth": heldout_model["tree_depth"],
        "l1_lambda": heldout_model["l1_lambda"],
        "sparsity_alpha": heldout_model["sparsity_alpha"],
        "metrics": heldout_model["metrics"],
    }


def train_symbolic_tree(
    export_path: str | Path,
    output_path: str | Path,
    *,
    config: SymbolicTrainConfig,
    summary_path: str | Path | None = None,
    calibration_export_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_config = config["data"]
    sodt_config = _materialize_sodt_config(config["search"])

    load_start_time = perf_counter()
    print(f"Loading symbolic training export from {export_path}...", flush=True)
    bundle, (feature_matrix, labels) = _load_symbolic_training_data(
        export_path,
        random_state=sodt_config["random_state"],
        neg_ratio=data_config.get("neg_ratio"),
    )
    print(
        f"Symbolic training data ready in {_format_duration(perf_counter() - load_start_time)}.",
        flush=True,
    )

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n" + "=" * 45)
    print(f"{'Selected RoI Class Distribution':^45}")
    print("-" * 45)
    print(f"{'Class Name':<25} | {'Selected RoIs':>15}")
    print("-" * 45)
    for label, count in zip(unique_labels, counts):
        print(f"{bundle.class_names[label]:<25} | {count:>15,}")
    print("=" * 45 + "\n")

    for line in _format_progress_header(
        feature_count=feature_matrix.shape[0],
        feature_dim=feature_matrix.shape[1],
        tree_depth=sodt_config["tree_depth"],
        l1_lambda=sodt_config["l1_lambda"],
        sparsity_alpha=sodt_config["sparsity_alpha"],
        iterations=sodt_config["iterations"],
        total_node_fit_upper_bound=((2 ** sodt_config["tree_depth"]) - 1)
        * sodt_config["iterations"],
    ):
        print(line)

    latest_mimic_accuracy = 0.0
    latest_nonzero_weights = 0
    current_node_bar: tqdm | None = None
    current_iteration = 0

    def _set_node_bar_status(status: str) -> None:
        if current_node_bar is None:
            return
        current_node_bar.set_postfix_str(
            f"{status} | mimic={latest_mimic_accuracy:.4f} nz={latest_nonzero_weights}",
            refresh=True,
        )

    def _on_iteration(metrics: dict[str, Any]) -> None:
        nonlocal latest_mimic_accuracy, latest_nonzero_weights
        latest_mimic_accuracy = metrics["mimic_accuracy"]
        latest_nonzero_weights = metrics["nonzero_weights"]
        _set_node_bar_status("iteration_done")

    def _on_node(event: dict[str, Any]) -> None:
        nonlocal current_iteration, current_node_bar
        iteration = int(event["iteration"])
        if iteration != current_iteration:
            if current_node_bar is not None:
                current_node_bar.close()
            current_iteration = iteration
            current_node_bar = tqdm(
                total=int(event["num_internal_nodes"]),
                desc=f"TAO {iteration}/{int(event['iterations'])} nodes",
                leave=True,
            )

        status = str(event.get("status", "solving"))
        if event["phase"] == "start":
            _set_node_bar_status(status)
            return

        if current_node_bar is not None:
            current_node_bar.update(1)
        _set_node_bar_status(status)

    tree = _build_symbolic_tree(
        bundle,
        feature_matrix,
        tree_depth=sodt_config["tree_depth"],
    )

    # Extract teacher confidence for confidence-weighted TAO.
    # Downweights samples where the teacher's softmax was uncertain,
    # reducing the impact of noisy teacher labels on the tree.
    teacher_confidence_np: np.ndarray | None = None
    if bundle.teacher_confidence is not None:
        teacher_confidence_np = (
            bundle.teacher_confidence.detach().cpu().numpy().astype(np.float32)
        )
        print(
            f"Teacher confidence weighting enabled: "
            f"mean={teacher_confidence_np.mean():.4f}, "
            f"min={teacher_confidence_np.min():.4f}, "
            f"max={teacher_confidence_np.max():.4f}",
            flush=True,
        )

    history = fit_tree_with_tao(
        tree,
        feature_matrix,
        labels,
        iterations=sodt_config["iterations"],
        l1_lambda=sodt_config["l1_lambda"],
        sparsity_alpha=sodt_config["sparsity_alpha"],
        logistic_max_iter=sodt_config["logistic_max_iter"],
        tolerance=sodt_config["tolerance"],
        zero_threshold=sodt_config["zero_threshold"],
        random_state=sodt_config["random_state"],
        show_progress=False,
        progress_callback=_on_iteration,
        node_progress_callback=_on_node,
        teacher_confidence=teacher_confidence_np,
    )

    if current_node_bar is not None:
        current_node_bar.close()

    # Calibrate the SODT's leaf distributions via temperature scaling.
    # This rescales probabilities for detection postprocessing (NMS)
    # without changing any class decisions (argmax is T-invariant).
    # Guo et al. 2017: temperature MUST be calibrated on held-out data,
    # not training data, to avoid overfitting the calibration.

    metrics = _augment_tree_metrics(tree, {})

    artifact = {
        "tree_state": tree.to_state_dict(),
        "metrics": metrics,
        "history": history,
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "export_path": str(export_path),
        "tree_depth": int(sodt_config["tree_depth"]),
        "l1_lambda": float(sodt_config["l1_lambda"]),
        "sparsity_alpha": float(sodt_config["sparsity_alpha"]),
        "training_config": {
            "tree_depth": sodt_config["tree_depth"],
            "iterations": sodt_config["iterations"],
            "l1_lambda": sodt_config["l1_lambda"],
            "sparsity_alpha": sodt_config["sparsity_alpha"],
            "logistic_max_iter": sodt_config["logistic_max_iter"],
            "tolerance": sodt_config["tolerance"],
            "zero_threshold": sodt_config["zero_threshold"],
            "random_state": sodt_config["random_state"],
            "neg_ratio": data_config.get("neg_ratio"),
            "symbolic_input": "raw_roi_align_pooled_grid",
            "sparsity_source": "l1_regularization_and_tree_structure",
            "config_sections": dict(config),
        },
    }
    ensure_dir(Path(output_path).parent)
    torch.save(artifact, output_path)

    summary = {
        "export_path": str(export_path),
        "output_path": str(output_path),
        "metrics": metrics,
        "history": history,
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "tree_depth": int(sodt_config["tree_depth"]),
        "l1_lambda": float(sodt_config["l1_lambda"]),
        "sparsity_alpha": float(sodt_config["sparsity_alpha"]),
        "training_config": artifact["training_config"],
    }
    if summary_path is not None:
        ensure_dir(Path(summary_path).parent)
        save_json(summary, summary_path)

    return summary, {
        "tree": tree,
        "bundle": bundle,
        "feature_matrix": feature_matrix,
        "labels": labels,
    }


def load_symbolic_tree(
    checkpoint_path: str | Path,
) -> SparseObliqueDecisionTreeClassifier:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("tree_state")
    if state_dict is None:
        raise ValueError("No symbolic tree state found in checkpoint.")
    return SparseObliqueDecisionTreeClassifier.from_state_dict(state_dict)


def prune_symbolic_tree(
    *,
    tree: SparseObliqueDecisionTreeClassifier,
    bundle: Any,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    checkpoint_path: str | Path,
    export_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if export_path is None:
        export_path = str(checkpoint.get("export_path", ""))

    print("=== Pruning Symbolic Tree ===")
    print(
        f"Tree: {feature_matrix.shape[0]:,} RoIs, "
        f"{tree.num_internal_nodes} internal nodes"
    )

    t0 = perf_counter()
    pruned_count = postprocess_tree(
        tree,
        feature_matrix,
        labels,
        class_names=bundle.class_names,
        verbose=True,
    )

    active_nodes = sum(c > 0 for c in tree.nonzero_weight_counts())
    print(
        f"Pruning complete — {pruned_count} nodes removed, "
        f"{active_nodes} remain active ({_format_duration(perf_counter() - t0)})"
    )

    print()
    tree_metrics = _augment_tree_metrics(tree, {})
    history = list(checkpoint.get("history", []))
    pruning_entry = {
        "iteration": len(history) + 1,
        "nonzero_weights": tree_metrics["nonzero_weights"],
        "active_internal_nodes": tree_metrics["active_internal_nodes"],
        "postprocess_pruned_nodes": pruned_count,
    }
    history.append(pruning_entry)

    print(f"nonzero weights: {tree_metrics['nonzero_weights']}")
    print(f"active nodes: {tree_metrics['active_internal_nodes']}")

    training_config = dict(checkpoint.get("training_config", {}))
    training_config["postprocess_pruned_nodes"] = pruned_count

    tree_depth = int(checkpoint.get("tree_depth", tree.max_depth))
    l1_lambda = float(checkpoint.get("l1_lambda", 0.0))
    sparsity_alpha = float(checkpoint.get("sparsity_alpha", 0.0))

    artifact = {
        "tree_state": tree.to_state_dict(),
        "metrics": tree_metrics,
        "history": history,
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "export_path": str(export_path),
        "tree_depth": tree_depth,
        "l1_lambda": l1_lambda,
        "sparsity_alpha": sparsity_alpha,
        "training_config": training_config,
    }
    torch.save(artifact, checkpoint_path)
    print("Saving checkpoint... done")

    if summary_path is not None:
        ensure_dir(Path(summary_path).parent)
        save_json(
            {
                "export_path": str(export_path),
                "output_path": str(checkpoint_path),
                "metrics": tree_metrics,
                "history": history,
                "class_names": bundle.class_names,
                "feature_shape": bundle.feature_shape,
                "tree_depth": tree_depth,
                "l1_lambda": l1_lambda,
                "sparsity_alpha": sparsity_alpha,
                "training_config": training_config,
            },
            summary_path,
        )
        print("Saving metrics... done")

    return {
        "export_path": str(export_path),
        "output_path": str(checkpoint_path),
        "metrics": tree_metrics,
        "history": history,
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "tree_depth": tree_depth,
        "l1_lambda": l1_lambda,
        "sparsity_alpha": sparsity_alpha,
        "training_config": training_config,
    }
