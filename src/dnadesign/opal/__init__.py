"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/__init__.py

Public OPAL package API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_EXPORTS = {
    "CampaignAnalysis": ".src.analysis.campaign",
    "assess_records_contract": ".src.analysis.campaign_progress",
    "assess_records_contract_for_schema": ".src.analysis.campaign_progress",
    "assess_records_contract_for_values": ".src.analysis.campaign_progress",
    "apply_candidate_eligibility": ".src.eligibility",
    "available_rounds": ".src.analysis.ledger",
    "build_ledger_status_table": ".src.analysis.campaign_progress",
    "CandidateEligibilityBlock": ".src.config.types",
    "CandidateEligibilityResult": ".src.eligibility",
    "CandidateEligibilityRuleResult": ".src.eligibility",
    "build_artifact_garden_audit": ".src.reporting.artifact_garden",
    "build_campaign_progress": ".src.reporting.progress",
    "build_campaign_review": ".src.reporting.review",
    "build_notebook_view_model": ".src.reporting.notebook",
    "build_campaign_set_notebook_view_model": ".src.reporting.notebook_set",
    "build_records_preview": ".src.analysis.campaign_progress",
    "cli_handoff_lines": ".src.analysis.campaign_progress",
    "describe_plot_kind": ".src.registries.plots",
    "enforce_x_matrix_memory_budget": ".src.runtime.memory_guard",
    "estimate_x_matrix_memory": ".src.runtime.memory_guard",
    "frozen_round0_scores": ".src.analysis.learning_loop_baselines",
    "latest_round": ".src.analysis.ledger",
    "latest_run_id": ".src.analysis.ledger",
    "list_configured_plot_specs": ".src.plots.config",
    "list_plot_kinds": ".src.registries.plots",
    "load_config": ".src.config.loader",
    "load_plot_artifact_manifest": ".src.plots.manifests",
    "load_plot_config": ".src.plots.config",
    "load_plot_manifest_index": ".src.plots.manifests",
    "load_review_manifest": ".src.reporting.review",
    "load_selection_set": ".src.reporting.selection_set",
    "load_selection_batch": ".src.reporting.selection_set",
    "materialize_campaign_set_collection_visuals": ".src.reporting.campaign_set_artifacts",
    "parse_enabled": ".src.plots.config",
    "parse_tags": ".src.plots.config",
    "PluginRef": ".src.config.types",
    "prune_stale_artifacts": ".src.reporting.artifact_garden",
    "OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION": ".api.observed_labels",
    "OBSERVED_LABELS_API_VERSION": ".api.observed_labels",
    "ObservedLabelPromotionBinding": ".api.observed_labels",
    "ObservedLabelVerificationError": ".api.observed_labels",
    "RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION": ".api.response_magnitude_feasibility",
    "read_optional_table": ".src.analysis.campaign_progress",
    "read_selection_artifact": ".src.reporting.verify_outputs",
    "read_campaign_predictions": ".src.reporting.predictions",
    "read_campaign_selection_view_predictions": ".src.reporting.predictions",
    "records_status_lines": ".src.analysis.campaign_progress",
    "render_campaign_notebook": ".src.analysis.notebook_template",
    "render_campaign_set_notebook": ".src.analysis.notebook_set_template",
    "render_campaign_progress_text": ".src.reporting.progress",
    "require_columns": ".src.analysis.ledger",
    "RestrictionSiteHit": ".src.eligibility",
    "RestrictionSiteScanReport": ".src.eligibility",
    "RestrictionSiteSpec": ".src.eligibility",
    "ResponseMagnitudeFeasibilityComponents": ".api.response_magnitude_feasibility",
    "ResponseMagnitudeFeasibilityScore": ".api.response_magnitude_feasibility",
    "run_campaign_plots": ".src.plots.api",
    "scan_restriction_sites": ".src.eligibility",
    "SFXIScoringConfig": ".api.sfxi",
    "SFXIScoringResult": ".api.sfxi",
    "SFXI_STATE_ORDER": ".api.sfxi",
    "score_vec8": ".api.sfxi",
    "score_vec8_with_denom": ".api.sfxi",
    "binary_target_mask": ".api.response_magnitude_feasibility",
    "calibrate_response_magnitude_feasibility": ".api.response_magnitude_feasibility",
    "response_magnitude_feasibility_components": ".api.response_magnitude_feasibility",
    "score_response_magnitude_feasibility": ".api.response_magnitude_feasibility",
    "smoke_check_notebook": ".src.reporting.notebook",
    "table_status_lines": ".src.analysis.campaign_progress",
    "unavailable_table": ".src.analysis.campaign_progress",
    "validate_x_parquet_column": ".src.storage.x_contracts",
    "validated_response_magnitude": ".api.response_magnitude_feasibility",
    "VerifiedObservedLabelPromotion": ".api.observed_labels",
    "VerifiedObservedLabelSnapshot": ".api.observed_labels",
    "candidate_snapshot_record": ".api.observed_labels",
    "verify_observed_label_snapshot": ".api.observed_labels",
    "x_provenance_status_lines": ".src.analysis.campaign_progress",
}


def main() -> None:
    """Run the OPAL CLI entrypoint."""

    from .src.cli import main as cli_main

    cli_main()


__all__ = [
    *_PUBLIC_EXPORTS,
    "main",
]


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_PUBLIC_EXPORTS])
