from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from tqdm import tqdm

from .dataset import (
    SymbolicFeatureScreening,
    apply_symbolic_feature_screening,
    build_symbolic_feature_screening,
    flatten_exported_symbolic_payload,
)
from .evaluation import evaluate_symbolic_candidate, evaluate_symbolic_spatial_metrics
from .sodt import SparseObliqueDecisionTreeClassifier
from .tao import evaluate_tree, fit_tree_with_tao
from .utils import load_symbolic_payload, save_symbolic_payload, save_symbolic_summary


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
    screening: SymbolicFeatureScreening,
    total_candidates: int,
    iterations: int,
    total_outer_iterations: int,
    total_node_fit_upper_bound: int,
) -> list[str]:
    screening_summary = screening.to_summary()
    return [
        (
            f"Loaded {feature_count} symbolic samples with {screening.selected_feature_dim} retained features "
            f"from {screening.original_feature_dim} pooled-feature dimensions."
        ),
        (
            f"Feature screening: kept {screening.selected_feature_dim}/{screening.original_feature_dim} features "
            f"({screening_summary['kept_fraction']:.3f} kept)."
        ),
        (
            f"Workload check: {total_candidates} candidates x {iterations} TAO iterations = "
            f"{total_outer_iterations} outer iterations and up to {total_node_fit_upper_bound} LIBLINEAR node fits."
        ),
        (
            "Progress line shows the live candidate, TAO step, mimic, sparsity, and ETA in one place. "
            "No side logs are emitted while the bar is running."
        ),
    ]


def _format_progress_postfix(
    candidate_position: int,
    total_candidates: int,
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
    iteration: int,
    total_iterations: int,
    mimic_accuracy: float,
    nonzero_weights: int,
    candidate_eta: float,
    overall_eta: float,
) -> str:
    return (
        f"cand {candidate_position}/{total_candidates} | "
        f"d={tree_depth} lam={l1_lambda:g} a={sparsity_alpha:g} | "
        f"tao {iteration}/{total_iterations} | "
        f"mimic={mimic_accuracy:.4f} | "
        f"nz={int(nonzero_weights)} | "
        f"eta(cand)={_format_duration(candidate_eta)} | "
        f"eta(total)={_format_duration(overall_eta)}"
    )


def _resolve_feature_screening_config(
    feature_screening: dict[str, Any] | None,
) -> dict[str, Any]:
    config = {
        "activation_threshold": 1e-8,
        "min_feature_std": 1e-5,
        "min_feature_range": 1e-5,
        "min_activation_rate": 0.0,
    }
    if feature_screening is not None:
        config.update(feature_screening)
    return config


def _resolve_extended_thresholds(
    extended_thresholds: dict[str, Any] | None,
) -> dict[str, float | None]:
    config: dict[str, float | None] = {
        "min_mimic_accuracy": 0.95,
        "min_macro_f1_vs_teacher": 0.95,
        "min_necessity_advantage_over_random": 0.0,
        "min_necessity_flip_advantage_over_random": 0.0,
        "min_sufficiency_advantage_over_random": 0.0,
        "min_sufficiency_retention_advantage_over_random": 0.0,
        "min_box_grounded_roi_overlap": None,
        "min_pointing_score": None,
        "min_energy_in_defect_ratio": None,
        "max_heatmap_entropy": None,
    }
    if extended_thresholds is not None:
        config.update(extended_thresholds)
    return config


def _load_symbolic_training_data(
    export_path: str | Path,
    include_background: bool,
    min_teacher_score: float | None,
    max_samples_total: int | None,
    random_state: int,
    feature_screening: dict[str, Any] | None,
) -> tuple[Any, Any, Any, SymbolicFeatureScreening, dict[str, Any]]:
    payload = load_symbolic_payload(export_path)
    bundle = flatten_exported_symbolic_payload(
        payload,
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=max_samples_total,
        random_state=random_state,
    )

    screening_config = _resolve_feature_screening_config(feature_screening)
    screening = build_symbolic_feature_screening(
        bundle.feature_vectors,
        activation_threshold=float(screening_config["activation_threshold"]),
        min_feature_std=float(screening_config["min_feature_std"]),
        min_feature_range=float(screening_config["min_feature_range"]),
        min_activation_rate=float(screening_config["min_activation_rate"]),
    )

    screened_vectors = apply_symbolic_feature_screening(bundle.feature_vectors, screening)
    feature_matrix = screened_vectors.detach().cpu().numpy().astype("float32")
    labels = bundle.teacher_labels.detach().cpu().numpy().astype("int64")
    return payload, bundle, (feature_matrix, labels), screening, screening_config


def _build_symbolic_tree(
    bundle: Any,
    feature_matrix: Any,
    max_depth: int,
    screening: SymbolicFeatureScreening,
) -> SparseObliqueDecisionTreeClassifier:
    return SparseObliqueDecisionTreeClassifier(
        max_depth=max_depth,
        num_classes=len(bundle.class_names),
        input_dim=int(feature_matrix.shape[1]),
        original_input_dim=int(screening.original_feature_dim),
        selected_feature_indices=screening.selected_feature_indices.detach().cpu().numpy().astype("int64"),
        feature_shape=bundle.feature_shape,
        class_names=bundle.class_names,
    )


def _augment_tree_metrics(
    tree: SparseObliqueDecisionTreeClassifier,
    metrics: dict[str, Any],
    screening: SymbolicFeatureScreening,
) -> dict[str, Any]:
    augmented = dict(metrics)
    augmented.update(
        {
            "tree_depth": int(tree.max_depth),
            "num_leaves": int(tree.num_leaves),
            "num_internal_nodes": int(tree.num_internal_nodes),
            "selected_feature_dim": int(screening.selected_feature_dim),
            "original_feature_dim": int(screening.original_feature_dim),
            "feature_keep_ratio": float(screening.selected_feature_dim / max(screening.original_feature_dim, 1)),
            "mean_nonzero_per_active_node": float(
                augmented["nonzero_weights"] / max(augmented["active_internal_nodes"], 1)
            ),
        }
    )
    return augmented


def _candidate_identifier(
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
) -> str:
    return (
        f"depth{int(tree_depth)}"
        f"_lambda{float(l1_lambda):g}"
        f"_alpha{float(sparsity_alpha):g}"
    )


def _evaluate_candidate_variant(
    tree: SparseObliqueDecisionTreeClassifier,
    bundle: Any,
    feature_matrix: Any,
    labels: Any,
    screening: SymbolicFeatureScreening,
    tree_depth: int,
    l1_lambda: float,
    sparsity_alpha: float,
    history: list[dict[str, Any]],
    random_state: int,
) -> dict[str, Any]:
    base_metrics = evaluate_tree(tree, feature_matrix, labels)
    symbolic_metrics = evaluate_symbolic_candidate(
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
        screening=screening,
    )
    return {
        "candidate_id": _candidate_identifier(tree_depth, l1_lambda, sparsity_alpha),
        "tree_depth": int(tree_depth),
        "l1_lambda": float(l1_lambda),
        "sparsity_alpha": float(sparsity_alpha),
        "metrics": metrics,
        "history": history,
        "tree_state": tree.to_state_dict(),
    }


def _candidate_report(candidate: dict[str, Any]) -> dict[str, Any]:
    report = {
        "candidate_id": candidate["candidate_id"],
        "tree_depth": candidate["tree_depth"],
        "num_leaves": candidate["metrics"]["num_leaves"],
        "l1_lambda": candidate["l1_lambda"],
        "sparsity_alpha": candidate["sparsity_alpha"],
        "mimic_accuracy": candidate["metrics"]["mimic_accuracy"],
        "macro_f1_vs_teacher": candidate["metrics"]["macro_f1_vs_teacher"],
        "per_class_agreement_vs_teacher": candidate["metrics"]["per_class_agreement_vs_teacher"],
        "nonzero_weights": candidate["metrics"]["nonzero_weights"],
        "mean_nonzero_per_node": candidate["metrics"]["mean_nonzero_per_node"],
        "active_internal_nodes": candidate["metrics"]["active_internal_nodes"],
        "feature_keep_ratio": candidate["metrics"]["feature_keep_ratio"],
        "mean_path_feature_count": candidate["metrics"]["mean_path_feature_count"],
        "necessity_advantage_over_random": candidate["metrics"]["necessity_advantage_over_random"],
        "necessity_flip_advantage_over_random": candidate["metrics"]["necessity_flip_advantage_over_random"],
        "sufficiency_advantage_over_random": candidate["metrics"]["sufficiency_advantage_over_random"],
        "sufficiency_retention_advantage_over_random": (
            candidate["metrics"]["sufficiency_retention_advantage_over_random"]
        ),
        "box_grounded_roi_overlap": candidate["metrics"]["box_grounded_roi_overlap"],
        "pointing_score": candidate["metrics"]["pointing_score"],
        "energy_in_defect_ratio": candidate["metrics"]["energy_in_defect_ratio"],
        "heatmap_entropy": candidate["metrics"]["heatmap_entropy"],
        "stability_score": candidate["metrics"]["stability_score"],
    }
    return report


def _candidate_min_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["metrics"]["nonzero_weights"],
        candidate["metrics"]["mean_nonzero_per_node"],
        candidate["metrics"]["active_internal_nodes"],
        candidate["tree_depth"],
        -candidate["metrics"]["mimic_accuracy"],
        -candidate["metrics"]["macro_f1_vs_teacher"],
    )


def _select_paper_faithful_candidate(
    candidates: list[dict[str, Any]],
    mimic_tolerance: float,
    macro_f1_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        raise ValueError("No symbolic training candidates were produced.")

    best_mimic = max(candidate["metrics"]["mimic_accuracy"] for candidate in candidates)
    best_macro_f1 = max(candidate["metrics"]["macro_f1_vs_teacher"] for candidate in candidates)
    survivors = [
        candidate
        for candidate in candidates
        if (
            best_mimic - candidate["metrics"]["mimic_accuracy"] <= float(mimic_tolerance)
            and best_macro_f1 - candidate["metrics"]["macro_f1_vs_teacher"] <= float(macro_f1_tolerance)
        )
    ]
    fallback_used = False
    if not survivors:
        survivors = [
            candidate
            for candidate in candidates
            if best_mimic - candidate["metrics"]["mimic_accuracy"] <= float(mimic_tolerance)
        ]
        fallback_used = True
    if not survivors:
        survivors = list(candidates)
        fallback_used = True

    selected_candidate = min(survivors, key=_candidate_min_key)
    ranking = [
        {
            "candidate_id": candidate["candidate_id"],
            "mimic_gap": float(best_mimic - candidate["metrics"]["mimic_accuracy"]),
            "macro_f1_gap": float(best_macro_f1 - candidate["metrics"]["macro_f1_vs_teacher"]),
            "survived": bool(candidate in survivors),
        }
        for candidate in candidates
    ]
    return selected_candidate, {
        "policy": "paper_faithful_selector",
        "best_mimic_accuracy": float(best_mimic),
        "best_macro_f1_vs_teacher": float(best_macro_f1),
        "mimic_tolerance": float(mimic_tolerance),
        "macro_f1_tolerance": float(macro_f1_tolerance),
        "fallback_used": fallback_used,
        "num_survivors": int(len(survivors)),
        "selected_reason": (
            "Near-best teacher-faithful candidate, then sparsest acceptable tree."
        ),
        "ranking": ranking,
    }


def _metric_value(candidate: dict[str, Any], metric_name: str) -> float:
    value = candidate["metrics"].get(metric_name, float("nan"))
    return float(value) if value is not None else float("nan")


def _passes_extended_thresholds(
    candidate: dict[str, Any],
    thresholds: dict[str, float | None],
) -> tuple[bool, list[str]]:
    checks = [
        ("mimic_accuracy", thresholds["min_mimic_accuracy"], ">="),
        ("macro_f1_vs_teacher", thresholds["min_macro_f1_vs_teacher"], ">="),
        ("necessity_advantage_over_random", thresholds["min_necessity_advantage_over_random"], ">="),
        ("necessity_flip_advantage_over_random", thresholds["min_necessity_flip_advantage_over_random"], ">="),
        ("sufficiency_advantage_over_random", thresholds["min_sufficiency_advantage_over_random"], ">="),
        (
            "sufficiency_retention_advantage_over_random",
            thresholds["min_sufficiency_retention_advantage_over_random"],
            ">=",
        ),
        ("box_grounded_roi_overlap", thresholds["min_box_grounded_roi_overlap"], ">="),
        ("pointing_score", thresholds["min_pointing_score"], ">="),
        ("energy_in_defect_ratio", thresholds["min_energy_in_defect_ratio"], ">="),
        ("heatmap_entropy", thresholds["max_heatmap_entropy"], "<="),
    ]

    failure_reasons: list[str] = []
    for metric_name, threshold, direction in checks:
        if threshold is None:
            continue
        metric_value = _metric_value(candidate, metric_name)
        if direction == ">=":
            if np.isnan(metric_value) or metric_value < float(threshold):
                failure_reasons.append(f"{metric_name}<{float(threshold):.4f}")
        else:
            if np.isnan(metric_value) or metric_value > float(threshold):
                failure_reasons.append(f"{metric_name}>{float(threshold):.4f}")
    return len(failure_reasons) == 0, failure_reasons


def _rank_candidates(
    candidates: list[dict[str, Any]],
    metric_name: str,
    higher_is_better: bool,
) -> dict[str, int]:
    values = []
    for candidate in candidates:
        value = _metric_value(candidate, metric_name)
        if np.isnan(value):
            value = float("-inf") if higher_is_better else float("inf")
        values.append((candidate["candidate_id"], value))

    ordered = sorted(
        values,
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    return {
        candidate_id: rank_index + 1
        for rank_index, (candidate_id, _) in enumerate(ordered)
    }


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    objectives: list[tuple[str, bool]],
) -> bool:
    left_better_or_equal = True
    left_strictly_better = False
    for metric_name, higher_is_better in objectives:
        left_value = _metric_value(left, metric_name)
        right_value = _metric_value(right, metric_name)
        if np.isnan(left_value):
            left_value = float("-inf") if higher_is_better else float("inf")
        if np.isnan(right_value):
            right_value = float("-inf") if higher_is_better else float("inf")

        if higher_is_better:
            if left_value < right_value:
                left_better_or_equal = False
                break
            if left_value > right_value:
                left_strictly_better = True
        else:
            if left_value > right_value:
                left_better_or_equal = False
                break
            if left_value < right_value:
                left_strictly_better = True
    return left_better_or_equal and left_strictly_better


def _pareto_front(
    candidates: list[dict[str, Any]],
    objectives: list[tuple[str, bool]],
) -> list[str]:
    front: list[str] = []
    for candidate in candidates:
        dominated = any(
            other["candidate_id"] != candidate["candidate_id"] and _dominates(other, candidate, objectives)
            for other in candidates
        )
        if not dominated:
            front.append(candidate["candidate_id"])
    return front


def _select_thesis_extended_candidate(
    candidates: list[dict[str, Any]],
    thresholds: dict[str, float | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        raise ValueError("No symbolic training candidates were produced.")

    threshold_results = []
    survivors: list[dict[str, Any]] = []
    for candidate in candidates:
        passed, failure_reasons = _passes_extended_thresholds(candidate, thresholds)
        threshold_results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "passed_thresholds": passed,
                "failure_reasons": failure_reasons,
            }
        )
        if passed:
            survivors.append(candidate)

    fallback_used = False
    ranking_pool = survivors
    if not ranking_pool:
        ranking_pool = list(candidates)
        fallback_used = True

    ranking_objectives = [
        ("necessity_advantage_over_random", True),
        ("necessity_flip_advantage_over_random", True),
        ("sufficiency_advantage_over_random", True),
        ("sufficiency_retention_advantage_over_random", True),
        ("box_grounded_roi_overlap", True),
        ("pointing_score", True),
        ("energy_in_defect_ratio", True),
        ("stability_score", True),
        ("heatmap_entropy", False),
        ("active_internal_nodes", False),
        ("mean_nonzero_per_node", False),
        ("nonzero_weights", False),
        ("tree_depth", False),
    ]
    rank_maps = {
        metric_name: _rank_candidates(ranking_pool, metric_name, higher_is_better)
        for metric_name, higher_is_better in ranking_objectives
    }
    pareto_front = _pareto_front(ranking_pool, ranking_objectives)
    ranking_summary = []
    for candidate in ranking_pool:
        rank_total = int(
            sum(rank_maps[metric_name][candidate["candidate_id"]] for metric_name, _ in ranking_objectives)
        )
        ranking_summary.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rank_total": rank_total,
                "on_pareto_front": candidate["candidate_id"] in pareto_front,
                **{
                    f"{metric_name}_rank": rank_maps[metric_name][candidate["candidate_id"]]
                    for metric_name, _ in ranking_objectives
                },
            }
        )

    ranking_lookup = {entry["candidate_id"]: entry for entry in ranking_summary}
    selected_candidate = min(
        ranking_pool,
        key=lambda candidate: (
            candidate["candidate_id"] not in pareto_front,
            ranking_lookup[candidate["candidate_id"]]["rank_total"],
            candidate["metrics"]["active_internal_nodes"],
            candidate["metrics"]["mean_nonzero_per_node"],
            candidate["metrics"]["nonzero_weights"],
            candidate["tree_depth"],
            -candidate["metrics"]["mimic_accuracy"],
        ),
    )
    return selected_candidate, {
        "policy": "thesis_extended_selector",
        "thresholds": thresholds,
        "fallback_used": fallback_used,
        "num_survivors": int(len(survivors)),
        "selected_reason": (
            "Candidates were filtered by minimum faithfulness thresholds, then ranked transparently by "
            "faithfulness-vs-random, box-grounded spatial concentration, and compactness."
        ),
        "threshold_results": threshold_results,
        "pareto_front": pareto_front,
        "ranking_summary": ranking_summary,
        "evaluation_extension": {
            "note": (
                "The thesis-extended selector uses necessity/sufficiency/random-control and spatial metrics as "
                "evaluation extensions. It does not replace the separate paper-faithful selector."
            )
        },
    }


def _selection_snapshot(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "tree_depth": candidate["tree_depth"],
        "l1_lambda": candidate["l1_lambda"],
        "sparsity_alpha": candidate["sparsity_alpha"],
        "metrics": candidate["metrics"],
    }


def _selection_tree_state(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return candidate["tree_state"]


def _relative_threshold(reference_value: float, allowed_drop: float) -> float | None:
    if np.isnan(reference_value):
        return None
    return float(reference_value - allowed_drop)


def _preserves_strong_intrinsic_faithfulness(
    candidate: dict[str, Any],
    default_candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    threshold_specs = [
        ("mimic_accuracy", _relative_threshold(_metric_value(default_candidate, "mimic_accuracy"), 0.005), ">="),
        (
            "macro_f1_vs_teacher",
            _relative_threshold(_metric_value(default_candidate, "macro_f1_vs_teacher"), 0.005),
            ">=",
        ),
        (
            "necessity_advantage_over_random",
            _relative_threshold(_metric_value(default_candidate, "necessity_advantage_over_random"), 0.02),
            ">=",
        ),
        (
            "sufficiency_advantage_over_random",
            _relative_threshold(_metric_value(default_candidate, "sufficiency_advantage_over_random"), 0.02),
            ">=",
        ),
        (
            "box_grounded_roi_overlap",
            _relative_threshold(_metric_value(default_candidate, "box_grounded_roi_overlap"), 0.03),
            ">=",
        ),
        (
            "pointing_score",
            _relative_threshold(_metric_value(default_candidate, "pointing_score"), 0.03),
            ">=",
        ),
    ]

    failure_reasons: list[str] = []
    for metric_name, threshold, direction in threshold_specs:
        if threshold is None:
            continue
        candidate_value = _metric_value(candidate, metric_name)
        if np.isnan(candidate_value):
            failure_reasons.append(f"{metric_name}=nan")
            continue
        if direction == ">=" and candidate_value < threshold:
            failure_reasons.append(f"{metric_name}<{threshold:.4f}")
    return len(failure_reasons) == 0, failure_reasons


def _clearer_intrinsic_gains(
    candidate: dict[str, Any],
    default_candidate: dict[str, Any],
) -> list[str]:
    gains: list[str] = []
    if candidate["tree_depth"] < default_candidate["tree_depth"]:
        gains.append("shallower tree")
    if candidate["metrics"]["active_internal_nodes"] < default_candidate["metrics"]["active_internal_nodes"]:
        gains.append("fewer active internal nodes")
    if candidate["metrics"]["mean_nonzero_per_node"] <= (0.9 * default_candidate["metrics"]["mean_nonzero_per_node"]):
        gains.append("lower mean nonzero per node")
    if candidate["metrics"]["mean_path_feature_count"] <= (
        0.9 * default_candidate["metrics"]["mean_path_feature_count"]
    ):
        gains.append("lower mean path feature count")
    return gains


def _build_intrinsic_interpretability_review(
    candidates: list[dict[str, Any]],
    default_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if default_candidate is None or not candidates:
        return None

    comparison_table: list[dict[str, Any]] = []
    viable_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        preserves_faithfulness, failure_reasons = _preserves_strong_intrinsic_faithfulness(
            candidate,
            default_candidate,
        )
        interpretability_gains = _clearer_intrinsic_gains(candidate, default_candidate)
        row = {
            "candidate_id": candidate["candidate_id"],
            "tree_depth": candidate["tree_depth"],
            "mimic_accuracy": candidate["metrics"]["mimic_accuracy"],
            "macro_f1_vs_teacher": candidate["metrics"]["macro_f1_vs_teacher"],
            "box_grounded_roi_overlap": candidate["metrics"]["box_grounded_roi_overlap"],
            "pointing_score": candidate["metrics"]["pointing_score"],
            "active_internal_nodes": candidate["metrics"]["active_internal_nodes"],
            "mean_nonzero_per_node": candidate["metrics"]["mean_nonzero_per_node"],
            "mean_path_feature_count": candidate["metrics"]["mean_path_feature_count"],
            "feature_keep_ratio": candidate["metrics"]["feature_keep_ratio"],
            "preserves_strong_faithfulness": preserves_faithfulness,
            "faithfulness_failure_reasons": failure_reasons,
            "interpretability_gains": interpretability_gains,
            "mimic_gap_vs_default": float(
                candidate["metrics"]["mimic_accuracy"] - default_candidate["metrics"]["mimic_accuracy"]
            ),
            "macro_f1_gap_vs_default": float(
                candidate["metrics"]["macro_f1_vs_teacher"] - default_candidate["metrics"]["macro_f1_vs_teacher"]
            ),
            "active_internal_nodes_delta_vs_default": int(
                candidate["metrics"]["active_internal_nodes"] - default_candidate["metrics"]["active_internal_nodes"]
            ),
            "mean_nonzero_per_node_delta_vs_default": float(
                candidate["metrics"]["mean_nonzero_per_node"]
                - default_candidate["metrics"]["mean_nonzero_per_node"]
            ),
            "mean_path_feature_count_delta_vs_default": float(
                candidate["metrics"]["mean_path_feature_count"]
                - default_candidate["metrics"]["mean_path_feature_count"]
            ),
            "recommended_intrinsic_order": None,
        }
        comparison_table.append(row)
        if candidate["candidate_id"] != default_candidate["candidate_id"] and preserves_faithfulness and interpretability_gains:
            viable_candidates.append(candidate)

    if viable_candidates:
        recommended_candidate = min(
            viable_candidates,
            key=lambda candidate: (
                candidate["metrics"]["active_internal_nodes"],
                candidate["metrics"]["mean_nonzero_per_node"],
                candidate["metrics"]["mean_path_feature_count"],
                candidate["tree_depth"],
                candidate["metrics"]["nonzero_weights"],
                -candidate["metrics"]["mimic_accuracy"],
                -candidate["metrics"]["macro_f1_vs_teacher"],
            ),
        )
        recommendation_changed = True
        recommendation_reason = (
            "A different intrinsic candidate is recommended because it keeps very strong faithfulness while "
            "making the symbolic reasoning easier to inspect through a smaller active path/tree footprint."
        )
    else:
        recommended_candidate = default_candidate
        recommendation_changed = False
        recommendation_reason = (
            "No other intrinsic candidate was clearly easier to inspect without materially weakening the current "
            "strong faithfulness. The current default strong candidate remains recommended."
        )

    ranked_candidates = sorted(
        comparison_table,
        key=lambda row: (
            row["candidate_id"] != recommended_candidate["candidate_id"],
            not row["preserves_strong_faithfulness"],
            -len(row["interpretability_gains"]),
            row["active_internal_nodes"],
            row["mean_nonzero_per_node"],
            row["mean_path_feature_count"],
            row["tree_depth"],
        ),
    )
    for rank_index, row in enumerate(ranked_candidates, start=1):
        row["recommended_intrinsic_order"] = rank_index

    return {
        "default_candidate": _selection_snapshot(default_candidate),
        "recommended_candidate": _selection_snapshot(recommended_candidate),
        "recommendation_changed_from_default": recommendation_changed,
        "recommendation_reason": recommendation_reason,
        "comparison_table": ranked_candidates,
        "note": (
            "This interpretability review improves readability and candidate comparison only. "
            "It does not replace the existing paper-faithful or thesis-extended selectors."
        ),
    }


def train_symbolic_tree_regularization_path(
    export_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    max_depth: int = 4,
    depth_values: list[int] | None = None,
    iterations: int = 40,
    lambda_values: list[float] | None = None,
    alpha_values: list[float] | None = None,
    logistic_max_iter: int = 200,
    tolerance: float = 1e-4,
    zero_threshold: float = 1e-5,
    random_state: int = 42,
    include_background: bool = True,
    min_teacher_score: float | None = None,
    max_samples_total: int | None = 12000,
    mimic_tolerance: float = 1e-3,
    macro_f1_tolerance: float = 1e-3,
    feature_screening: dict[str, Any] | None = None,
    extended_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, bundle, (feature_matrix, labels), screening, screening_config = _load_symbolic_training_data(
        export_path,
        include_background=include_background,
        min_teacher_score=min_teacher_score,
        max_samples_total=max_samples_total,
        random_state=random_state,
        feature_screening=feature_screening,
    )

    depth_values = sorted({int(value) for value in (depth_values or [max_depth])})
    lambda_values = [float(value) for value in (lambda_values or [1.0, 0.1, 0.01, 0.001])]
    alpha_values = sorted(float(value) for value in (alpha_values or [-1.0, -0.5, 0.0, 0.5, 1.0]))
    extended_thresholds_config = _resolve_extended_thresholds(extended_thresholds)

    candidates: list[dict[str, Any]] = []
    candidate_grid = [
        (int(depth), float(l1_lambda), float(sparsity_alpha))
        for depth in depth_values
        for l1_lambda in lambda_values
        for sparsity_alpha in alpha_values
    ]
    total_candidates = len(candidate_grid)
    total_outer_iterations = total_candidates * iterations
    total_node_fit_upper_bound = sum(((2**depth) - 1) * iterations for depth, _, _ in candidate_grid)
    for line in _format_progress_header(
        feature_count=int(feature_matrix.shape[0]),
        screening=screening,
        total_candidates=total_candidates,
        iterations=iterations,
        total_outer_iterations=total_outer_iterations,
        total_node_fit_upper_bound=total_node_fit_upper_bound,
    ):
        print(line)

    progress_bar = tqdm(
        total=total_outer_iterations,
        desc="Symbolic regularization path",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.1,
    )
    candidate_order = {
        (int(depth), float(l1_lambda), float(sparsity_alpha)): index + 1
        for index, (depth, l1_lambda, sparsity_alpha) in enumerate(candidate_grid)
    }

    path_start_time = perf_counter()

    def _build_progress_callback(
        tree_depth: int,
        l1_lambda: float,
        sparsity_alpha: float,
    ) -> Any:
        candidate_position = candidate_order[(int(tree_depth), float(l1_lambda), float(sparsity_alpha))]
        candidate_start_time = perf_counter()
        completed_before_candidate = (candidate_position - 1) * iterations

        def _on_iteration(metrics: dict[str, Any]) -> None:
            progress_bar.update(1)
            completed_overall = completed_before_candidate + metrics["iteration"]
            elapsed_overall = perf_counter() - path_start_time
            elapsed_candidate = perf_counter() - candidate_start_time
            mean_overall_seconds = elapsed_overall / max(completed_overall, 1)
            mean_candidate_seconds = elapsed_candidate / max(metrics["iteration"], 1)
            overall_eta = mean_overall_seconds * max(total_outer_iterations - completed_overall, 0)
            candidate_eta = mean_candidate_seconds * max(iterations - metrics["iteration"], 0)
            progress_bar.set_description_str("Symbolic regularization path")
            progress_bar.set_postfix_str(
                _format_progress_postfix(
                    candidate_position=candidate_position,
                    total_candidates=total_candidates,
                    tree_depth=tree_depth,
                    l1_lambda=l1_lambda,
                    sparsity_alpha=sparsity_alpha,
                    iteration=int(metrics["iteration"]),
                    total_iterations=iterations,
                    mimic_accuracy=float(metrics["mimic_accuracy"]),
                    nonzero_weights=int(metrics["nonzero_weights"]),
                    candidate_eta=candidate_eta,
                    overall_eta=overall_eta,
                )
            )

        return _on_iteration

    for tree_depth in depth_values:
        for l1_lambda in lambda_values:
            tree = _build_symbolic_tree(
                bundle,
                feature_matrix,
                max_depth=tree_depth,
                screening=screening,
            )
            for sparsity_alpha in alpha_values:
                candidate_position = candidate_order[(int(tree_depth), float(l1_lambda), float(sparsity_alpha))]
                progress_bar.set_postfix_str(
                    _format_progress_postfix(
                        candidate_position=candidate_position,
                        total_candidates=total_candidates,
                        tree_depth=tree_depth,
                        l1_lambda=l1_lambda,
                        sparsity_alpha=sparsity_alpha,
                        iteration=0,
                        total_iterations=iterations,
                        mimic_accuracy=0.0,
                        nonzero_weights=0,
                        candidate_eta=0.0,
                        overall_eta=max(total_outer_iterations - progress_bar.n, 0),
                    )
                )
                history = fit_tree_with_tao(
                    tree,
                    feature_matrix,
                    labels,
                    iterations=iterations,
                    l1_lambda=l1_lambda,
                    sparsity_alpha=sparsity_alpha,
                    logistic_max_iter=logistic_max_iter,
                    tolerance=tolerance,
                    zero_threshold=zero_threshold,
                    random_state=random_state,
                    show_progress=False,
                    progress_callback=_build_progress_callback(tree_depth, l1_lambda, sparsity_alpha),
                )

                pre_candidate = _evaluate_candidate_variant(
                    tree,
                    bundle=bundle,
                    feature_matrix=feature_matrix,
                    labels=labels,
                    screening=screening,
                    tree_depth=tree_depth,
                    l1_lambda=l1_lambda,
                    sparsity_alpha=sparsity_alpha,
                    history=history,
                    random_state=random_state,
                )
                candidates.append(pre_candidate)

    progress_bar.close()

    paper_candidate, paper_summary = _select_paper_faithful_candidate(
        candidates,
        mimic_tolerance=mimic_tolerance,
        macro_f1_tolerance=macro_f1_tolerance,
    )
    extended_candidate, extended_summary = _select_thesis_extended_candidate(
        candidates,
        thresholds=extended_thresholds_config,
    )
    interpretability_review = _build_intrinsic_interpretability_review(
        candidates,
        default_candidate=paper_candidate,
    )

    artifact = {
        "tree_state": _selection_tree_state(paper_candidate),
        "metrics": paper_candidate["metrics"],
        "history": paper_candidate["history"],
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "export_path": str(export_path),
        "feature_screening": screening.to_summary(),
        "candidate_tables": [_candidate_report(candidate) for candidate in candidates],
        "selected_models": {
            "paper_faithful": _selection_snapshot(paper_candidate),
            "thesis_extended": _selection_snapshot(extended_candidate),
        },
        "selected_tree_states": {
            "paper_faithful": _selection_tree_state(paper_candidate),
            "thesis_extended": _selection_tree_state(extended_candidate),
        },
        "selection": {
            "paper_faithful": paper_summary,
            "thesis_extended": extended_summary,
        },
        "interpretability_review": interpretability_review,
        "training_config": {
            "max_depth": max_depth,
            "depth_values": [int(value) for value in depth_values],
            "iterations": iterations,
            "lambda_values": [float(value) for value in lambda_values],
            "alpha_values": [float(value) for value in alpha_values],
            "logistic_max_iter": logistic_max_iter,
            "tolerance": tolerance,
            "zero_threshold": zero_threshold,
            "random_state": random_state,
            "include_background": include_background,
            "min_teacher_score": min_teacher_score,
            "max_samples_total": max_samples_total,
            "mimic_tolerance": float(mimic_tolerance),
            "macro_f1_tolerance": float(macro_f1_tolerance),
            "feature_screening": screening_config,
            "extended_thresholds": extended_thresholds_config,
        },
        "default_selection_lane": "paper_faithful",
    }
    save_symbolic_payload(artifact, output_path)

    summary = {
        "export_path": str(export_path),
        "output_path": str(output_path),
        "metrics": paper_candidate["metrics"],
        "history": paper_candidate["history"],
        "class_names": bundle.class_names,
        "feature_shape": bundle.feature_shape,
        "feature_screening": screening.to_summary(),
        "candidate_tables": artifact["candidate_tables"],
        "selected_models": artifact["selected_models"],
        "selection": artifact["selection"],
        "interpretability_review": interpretability_review,
    }
    if summary_path is not None:
        save_symbolic_summary(summary, summary_path)

    return summary


def load_symbolic_tree(
    checkpoint_path: str | Path,
    selection_lane: str = "paper_faithful",
) -> SparseObliqueDecisionTreeClassifier:
    checkpoint = load_symbolic_payload(checkpoint_path)
    if "selected_tree_states" in checkpoint:
        lane_group = checkpoint["selected_tree_states"].get(selection_lane)
        state_dict = lane_group
        if isinstance(lane_group, dict):
            state_dict = lane_group.get("pre_prune")
            if state_dict is None:
                state_dict = next((value for value in lane_group.values() if value is not None), None)
        if state_dict is None:
            fallback_lane = checkpoint.get("default_selection_lane", "paper_faithful")
            lane_group = checkpoint["selected_tree_states"].get(fallback_lane)
            state_dict = lane_group
            if isinstance(lane_group, dict):
                state_dict = lane_group.get("pre_prune")
                if state_dict is None:
                    state_dict = next((value for value in lane_group.values() if value is not None), None)
        if state_dict is None:
            raise ValueError(
                f"No symbolic tree state found for selection_lane={selection_lane!r}."
            )
        return SparseObliqueDecisionTreeClassifier.from_state_dict(state_dict)

    return SparseObliqueDecisionTreeClassifier.from_state_dict(checkpoint["tree_state"])
