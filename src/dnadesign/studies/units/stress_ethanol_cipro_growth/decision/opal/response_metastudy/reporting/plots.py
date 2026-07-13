"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plots.py

Plot orchestration for the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from ..core.contracts import RecommendationThresholds, SfxiEvidenceFrame
from ..core.policies import SCORE_SURFACE_POLICY_ID
from ..core.response_contracts import ResponseMetricScreen
from .diagnostic_plots import (
    write_logic_effect_scatter,
    write_logic_gate_feasibility,
    write_score_correlation_matrix,
    write_selected_setpoint_residuals,
    write_selected_vec8_profiles,
)
from .metric_behavior_plots import (
    write_denominator_sensitivity,
    write_policy_comparison_panel_roles,
    write_sfxi_score_contours,
    write_target_view_pareto_fronts,
)
from .metric_comparison_plots import write_measured_response_examples, write_metric_compensation_comparison
from .model_validation_plot import write_model_validation
from .plot_catalog import PLOT_TIER_DIRS, build_plot_manifest, specs_by_id
from .primary_plots import (
    write_policy_decision_frontier,
    write_policy_guardrail_matrix,
    write_score_component_dominance,
)
from .response_assay_plots import (
    write_reader_event_intervals,
    write_repeated_design_agreement,
    write_response_constraint_coverage,
    write_response_separation_stability,
    write_response_uncertainty_sources,
)
from .response_model_plots import (
    write_greedy_support_evidence,
    write_label_model_screen,
    write_retrospective_enrichment,
)
from .rmf_contract_plot import write_rmf_cardinality_pressure
from .screen_plots import (
    write_logic_effect_tradeoff_fidelity,
    write_logic_effect_tradeoff_overlap,
    write_policy_overlap_summary,
    write_topk_overlap_curve,
)
from .sfxi_comparison_plots import write_sfxi_comparison_stability, write_sfxi_comparison_target_coverage
from .support_plot import write_candidate_logic_support


def write_visuals(
    out_dir: Path,
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    candidates: pd.DataFrame,
    overlap_by_k: pd.DataFrame,
    denominator_sensitivity: pd.DataFrame,
    comparison_panel: pd.DataFrame,
    model_validation: pd.DataFrame,
    setpoint_support: pd.DataFrame,
    sfxi_comparison: pd.DataFrame,
    response_screen: ResponseMetricScreen,
    metric_comparison: pd.DataFrame,
    rmf_cardinality_pressure: pd.DataFrame,
    scored: dict[str, dict[str, pd.DataFrame]],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    thresholds: RecommendationThresholds,
    comparison_policy_id: str,
    model_support_passed: bool,
    primary_reduction_id: str,
) -> tuple[dict[str, Path], pd.DataFrame]:
    specs = specs_by_id()
    paths = {plot_id: out_dir / "plots" / PLOT_TIER_DIRS[spec.tier] / spec.filename for plot_id, spec in specs.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_metric_compensation_comparison(metric_comparison, paths["metric_compensation_comparison"])
    write_measured_response_examples(metric_comparison, paths["measured_response_examples"])
    write_rmf_cardinality_pressure(
        rmf_cardinality_pressure,
        paths["rmf_cardinality_pressure"],
    )

    write_policy_guardrail_matrix(
        summary,
        paths["policy_guardrail_matrix"],
        thresholds=thresholds,
        comparison_policy_id=comparison_policy_id,
        model_support_passed=model_support_passed,
    )
    write_policy_decision_frontier(
        summary,
        paths["policy_decision_frontier"],
        thresholds=thresholds,
        comparison_policy_id=comparison_policy_id,
    )
    write_score_component_dominance(
        summary,
        pairwise,
        paths["score_component_dominance"],
        comparison_policy_id=comparison_policy_id,
    )

    write_selected_setpoint_residuals(
        summary,
        candidates,
        paths["selected_setpoint_residuals"],
        comparison_policy_id=comparison_policy_id,
        target_views=tuple(frame.target_view for frame in sfxi_evidence),
    )
    write_logic_gate_feasibility(summary, paths["logic_gate_feasibility"], thresholds=thresholds)
    write_logic_effect_scatter(
        summary,
        candidates,
        paths["logic_effect_topk_scatter"],
        comparison_policy_id=comparison_policy_id,
    )
    write_score_correlation_matrix(
        summary,
        pairwise,
        paths["score_correlation_matrix"],
        comparison_policy_id=comparison_policy_id,
    )
    write_selected_vec8_profiles(
        summary,
        candidates,
        paths["selected_vec8_profiles"],
        comparison_policy_id=comparison_policy_id,
    )
    write_sfxi_score_contours(
        summary,
        paths["sfxi_score_contours"],
        score_surface_policy_id=SCORE_SURFACE_POLICY_ID,
    )
    write_target_view_pareto_fronts(
        summary,
        scored,
        sfxi_evidence,
        paths["target_view_pareto_fronts"],
        comparison_policy_id=comparison_policy_id,
    )
    write_denominator_sensitivity(denominator_sensitivity, paths["denominator_sensitivity"])
    write_policy_comparison_panel_roles(
        comparison_panel,
        paths["policy_comparison_panel_roles"],
    )
    write_model_validation(model_validation, paths["model_validation"])
    write_candidate_logic_support(
        setpoint_support,
        paths["candidate_logic_support"],
        thresholds=thresholds,
    )
    write_sfxi_comparison_stability(sfxi_comparison, paths["sfxi_comparison_stability"])
    write_sfxi_comparison_target_coverage(
        sfxi_comparison,
        paths["sfxi_comparison_target_coverage"],
        logic_threshold=thresholds.min_target_view_median_logic,
    )
    write_reader_event_intervals(
        response_screen.event_intervals,
        paths["reader_event_intervals"],
    )
    write_response_separation_stability(
        response_screen.stability,
        paths["response_separation_stability"],
    )
    write_response_constraint_coverage(
        response_screen.stability,
        paths["response_constraint_coverage"],
        primary_reduction_id=primary_reduction_id,
    )
    write_response_uncertainty_sources(
        response_screen.uncertainty,
        paths["response_uncertainty_sources"],
    )
    write_label_model_screen(
        response_screen.model_screen,
        paths["label_model_screen"],
    )
    write_retrospective_enrichment(
        response_screen.enrichment_summary,
        response_screen.model_screen,
        paths["retrospective_enrichment"],
    )
    write_greedy_support_evidence(
        response_screen.greedy_support,
        paths["greedy_support_evidence"],
    )
    write_repeated_design_agreement(
        response_screen.repeated_agreement,
        paths["repeated_design_agreement"],
    )

    write_logic_effect_tradeoff_overlap(summary, paths["logic_effect_tradeoff_overlap"])
    write_logic_effect_tradeoff_fidelity(summary, paths["logic_effect_tradeoff_fidelity"])
    write_policy_overlap_summary(
        summary,
        paths["policy_overlap_summary"],
        comparison_policy_id=comparison_policy_id,
    )
    write_topk_overlap_curve(
        summary,
        overlap_by_k,
        paths["topk_overlap_curve"],
        comparison_policy_id=comparison_policy_id,
    )

    manifest = build_plot_manifest(paths, root=out_dir)
    return paths, manifest
