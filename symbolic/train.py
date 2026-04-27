from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from tqdm import tqdm

from .dataset import flatten_exported_symbolic_payload
from .config import SymbolicTrainConfig
from .evaluation import evaluate_symbolic_model as evaluate_symbolic_metrics
from .evaluation import evaluate_symbolic_spatial_metrics
from .sodt import SparseObliqueDecisionTreeClassifier
from .tao import evaluate_tree, fit_tree_with_tao
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
            "Progress line shows the TAO step, mimic, sparsity, and ETA in one place. "
            "No side logs are emitted while the bar is running."
        ),
    ]


def _format_progress_postfix(
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
    iteration: int,
    total_iterations: int,
    mimic_accuracy: float,
    nonzero_weights: int,
    eta: float,
) -> str:
    return (
        f"d={tree_depth} lam={l1_lambda:g} a={sparsity_alpha:g} | "
        f"tao {iteration}/{total_iterations} | "
        f"mimic={mimic_accuracy:.4f} | "
        f"nz={int(nonzero_weights)} | "
        f"eta={_format_duration(eta)}"
    )


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
    include_background: bool,
    min_teacher_score: float | None,
    max_samples_total: int | None,
    random_state: int,
) -> tuple[Any, Any]:
    bundle = flatten_exported_symbolic_payload(
        torch.load(export_path, map_location="cpu", weights_only=True),
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=max_samples_total,
        random_state=random_state,
    )

    feature_matrix = bundle.feature_vectors.detach().cpu().numpy().astype("float32")
    labels = bundle.teacher_labels.detach().cpu().numpy().astype("int64")
    return bundle, (feature_matrix, labels)


def _load_symbolic_evaluation_data(
    export_path: str | Path,
    include_background: bool,
    min_teacher_score: float | None,
    random_state: int,
) -> tuple[Any, Any, Any]:
    bundle = flatten_exported_symbolic_payload(
        torch.load(export_path, map_location="cpu", weights_only=True),
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=None,
        random_state=random_state,
    )
    feature_matrix = bundle.feature_vectors.detach().cpu().numpy().astype("float32")
    labels = bundle.teacher_labels.detach().cpu().numpy().astype("int64")
    return bundle, feature_matrix, labels


def _build_symbolic_tree(
    bundle: Any,
    feature_matrix: Any,
    tree_depth: int,
) -> SparseObliqueDecisionTreeClassifier:
    return SparseObliqueDecisionTreeClassifier(
        max_depth=tree_depth,
        num_classes=len(bundle.class_names),
        input_dim=int(feature_matrix.shape[1]),
        original_input_dim=int(feature_matrix.shape[1]),
        feature_shape=bundle.feature_shape,
        class_names=bundle.class_names,
    )


def _augment_tree_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    augmented = dict(metrics)
    augmented.update(
        {
            "tree_depth": int(tree.max_depth),
            "num_leaves": int(tree.num_leaves),
            "num_internal_nodes": int(tree.num_internal_nodes),
            "mean_nonzero_per_active_node": float(
                augmented["nonzero_weights"] / max(augmented["active_internal_nodes"], 1)
            ),
        }
    )
    return augmented


def _model_identifier(
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
) -> str:
    return (
        f"depth{int(tree_depth)}"
        f"_lambda{float(l1_lambda):g}"
        f"_alpha{float(sparsity_alpha):g}"
    )


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
    base_metrics = evaluate_tree(tree, feature_matrix, labels)
    symbolic_metrics = evaluate_symbolic_metrics(
        tree,
        feature_matrix=feature_matrix,
        teacher_labels=labels,
        class_names=bundle.class_names,
        random_state=random_state,
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
            **base_metrics,
            **symbolic_metrics,
            **spatial_metrics,
        },
    )
    return {
        "model_id": _model_identifier(tree_depth, l1_lambda, sparsity_alpha),
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
    tree = SparseObliqueDecisionTreeClassifier.from_state_dict(trained_model["tree_state"])
    base_metrics = evaluate_tree(tree, feature_matrix, labels)
    symbolic_metrics = evaluate_symbolic_metrics(
        tree,
        feature_matrix=feature_matrix,
        teacher_labels=labels,
        class_names=bundle.class_names,
        random_state=random_state,
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
            **base_metrics,
            **symbolic_metrics,
            **spatial_metrics,
        },
    )
    evaluated_model = {
        "model_id": trained_model["model_id"],
        "tree_depth": trained_model["tree_depth"],
        "l1_lambda": trained_model["l1_lambda"],
        "sparsity_alpha": trained_model["sparsity_alpha"],
        "metrics": metrics,
    }
    return {
        "model_id": trained_model["model_id"],
        "tree_depth": trained_model["tree_depth"],
        "l1_lambda": trained_model["l1_lambda"],
        "sparsity_alpha": trained_model["sparsity_alpha"],
        "metrics": metrics,
        "report": _model_report(evaluated_model),
    }


def _build_heldout_review(
    heldout_export_path: str | Path,
    include_background: bool,
    min_teacher_score: float | None,
    random_state: int,
    trained_model: dict[str, Any],
) -> dict[str, Any]:
    bundle, feature_matrix, labels = _load_symbolic_evaluation_data(
        heldout_export_path,
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        random_state=random_state,
    )

    heldout_model = _evaluate_trained_model_on_bundle(
        trained_model=trained_model,
        bundle=bundle,
        feature_matrix=feature_matrix,
        labels=labels,
        random_state=random_state,
    )

    return {
        "export_path": str(heldout_export_path),
        "sample_count": int(feature_matrix.shape[0]),
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "model": {
            "model_id": heldout_model["model_id"],
            "tree_depth": heldout_model["tree_depth"],
            "l1_lambda": heldout_model["l1_lambda"],
            "sparsity_alpha": heldout_model["sparsity_alpha"],
            "metrics": heldout_model["metrics"],
        },
        "report": heldout_model["report"],
        "note": (
            "Held-out test evaluates the trained SODT once. It does not participate in training."
        ),
    }


def _model_report(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model["model_id"],
        "tree_depth": model["tree_depth"],
        "num_leaves": model["metrics"]["num_leaves"],
        "l1_lambda": model["l1_lambda"],
        "sparsity_alpha": model["sparsity_alpha"],
        "mimic_accuracy": model["metrics"]["mimic_accuracy"],
        "macro_f1_vs_teacher": model["metrics"]["macro_f1_vs_teacher"],
        "per_class_agreement_vs_teacher": model["metrics"]["per_class_agreement_vs_teacher"],
        "nonzero_weights": model["metrics"]["nonzero_weights"],
        "mean_nonzero_per_node": model["metrics"]["mean_nonzero_per_node"],
        "active_internal_nodes": model["metrics"]["active_internal_nodes"],
        "mean_path_feature_count": model["metrics"]["mean_path_feature_count"],
        "necessity_advantage_over_random": model["metrics"]["necessity_advantage_over_random"],
        "necessity_flip_advantage_over_random": model["metrics"]["necessity_flip_advantage_over_random"],
        "sufficiency_advantage_over_random": model["metrics"]["sufficiency_advantage_over_random"],
        "sufficiency_retention_advantage_over_random": (
            model["metrics"]["sufficiency_retention_advantage_over_random"]
        ),
        "box_grounded_roi_overlap": model["metrics"]["box_grounded_roi_overlap"],
        "pointing_score": model["metrics"]["pointing_score"],
        "energy_in_defect_ratio": model["metrics"]["energy_in_defect_ratio"],
        "heatmap_entropy": model["metrics"]["heatmap_entropy"],
        "stability_score": model["metrics"]["stability_score"],
    }


def train_symbolic_tree(
    export_path: str | Path,
    output_path: str | Path,
    *,
    config: SymbolicTrainConfig,
    summary_path: str | Path | None = None,
    heldout_export_path: str | Path | None = None,
) -> dict[str, Any]:
    data_config = config["data"]
    sodt_config = _materialize_sodt_config(config["search"])

    bundle, (feature_matrix, labels) = _load_symbolic_training_data(
        export_path,
        include_background=data_config["include_background"],
        min_teacher_score=data_config["min_teacher_score"],
        max_samples_total=data_config["max_samples_total"],
        random_state=sodt_config["random_state"],
    )

    total_node_fit_upper_bound = ((2 ** sodt_config["tree_depth"]) - 1) * sodt_config["iterations"]
    for line in _format_progress_header(
        feature_count=int(feature_matrix.shape[0]),
        feature_dim=int(feature_matrix.shape[1]),
        tree_depth=sodt_config["tree_depth"],
        l1_lambda=sodt_config["l1_lambda"],
        sparsity_alpha=sodt_config["sparsity_alpha"],
        iterations=sodt_config["iterations"],
        total_node_fit_upper_bound=total_node_fit_upper_bound,
    ):
        print(line)

    progress_bar = tqdm(
        total=sodt_config["iterations"],
        desc="Symbolic SODT training",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.1,
    )

    training_start_time = perf_counter()

    def _on_iteration(metrics: dict[str, Any]) -> None:
        progress_bar.update(1)
        elapsed = perf_counter() - training_start_time
        mean_seconds = elapsed / max(metrics["iteration"], 1)
        eta = mean_seconds * max(sodt_config["iterations"] - metrics["iteration"], 0)
        progress_bar.set_description_str("Symbolic SODT training")
        progress_bar.set_postfix_str(
            _format_progress_postfix(
                tree_depth=sodt_config["tree_depth"],
                l1_lambda=sodt_config["l1_lambda"],
                sparsity_alpha=sodt_config["sparsity_alpha"],
                iteration=int(metrics["iteration"]),
                total_iterations=sodt_config["iterations"],
                mimic_accuracy=float(metrics["mimic_accuracy"]),
                nonzero_weights=int(metrics["nonzero_weights"]),
                eta=eta,
            )
        )

    progress_bar.set_postfix_str(
        _format_progress_postfix(
            tree_depth=sodt_config["tree_depth"],
            l1_lambda=sodt_config["l1_lambda"],
            sparsity_alpha=sodt_config["sparsity_alpha"],
            iteration=0,
            total_iterations=sodt_config["iterations"],
            mimic_accuracy=0.0,
            nonzero_weights=0,
            eta=0.0,
        )
    )

    tree = _build_symbolic_tree(
        bundle,
        feature_matrix,
        tree_depth=sodt_config["tree_depth"],
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
    )

    progress_bar.close()

    trained_model = _evaluate_symbolic_model(
        tree,
        bundle=bundle,
        feature_matrix=feature_matrix,
        labels=labels,
        tree_depth=sodt_config["tree_depth"],
        l1_lambda=sodt_config["l1_lambda"],
        sparsity_alpha=sodt_config["sparsity_alpha"],
        history=history,
        random_state=sodt_config["random_state"],
    )
    heldout_review = None
    if heldout_export_path is not None:
        heldout_review = _build_heldout_review(
            heldout_export_path=heldout_export_path,
            include_background=data_config["include_background"],
            min_teacher_score=data_config["min_teacher_score"],
            random_state=sodt_config["random_state"],
            trained_model=trained_model,
        )

    artifact = {
        "tree_state": trained_model["tree_state"],
        "metrics": trained_model["metrics"],
        "history": trained_model["history"],
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "export_path": str(export_path),
        "model": {
            "model_id": trained_model["model_id"],
            "tree_depth": trained_model["tree_depth"],
            "l1_lambda": trained_model["l1_lambda"],
            "sparsity_alpha": trained_model["sparsity_alpha"],
            "metrics": trained_model["metrics"],
        },
        "train_report": _model_report(trained_model),
        "heldout_review": heldout_review,
        "training_config": {
            "tree_depth": sodt_config["tree_depth"],
            "iterations": sodt_config["iterations"],
            "l1_lambda": sodt_config["l1_lambda"],
            "sparsity_alpha": sodt_config["sparsity_alpha"],
            "logistic_max_iter": sodt_config["logistic_max_iter"],
            "tolerance": sodt_config["tolerance"],
            "zero_threshold": sodt_config["zero_threshold"],
            "random_state": sodt_config["random_state"],
            "include_background": data_config["include_background"],
            "min_teacher_score": data_config["min_teacher_score"],
            "max_samples_total": data_config["max_samples_total"],
            "symbolic_input": "raw_roi_align_pooled_grid",
            "sparsity_source": "l1_regularization_and_tree_structure",
            "heldout_export_path": str(heldout_export_path) if heldout_export_path is not None else None,
            "config_sections": dict(config),
        },
    }
    ensure_dir(Path(output_path).parent)
    torch.save(artifact, output_path)

    summary = {
        "export_path": str(export_path),
        "output_path": str(output_path),
        "metrics": trained_model["metrics"],
        "history": trained_model["history"],
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "model": artifact["model"],
        "train_report": artifact["train_report"],
        "heldout_review": heldout_review,
        "training_config": artifact["training_config"],
    }
    if summary_path is not None:
        ensure_dir(Path(summary_path).parent)
        save_json(summary, summary_path)

    return summary


def load_symbolic_tree(
    checkpoint_path: str | Path,
) -> SparseObliqueDecisionTreeClassifier:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("tree_state")
    if state_dict is None:
        raise ValueError("No symbolic tree state found in checkpoint.")
    return SparseObliqueDecisionTreeClassifier.from_state_dict(state_dict)
