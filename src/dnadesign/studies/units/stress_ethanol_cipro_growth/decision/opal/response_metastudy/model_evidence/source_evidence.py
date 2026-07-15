"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/source_evidence.py

Source, corpus, and selection-support projections for model evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import ModelEvidenceError
from .fields import nonnegative_integer, required_mapping, required_string, sha256_digest


def target_views(source: dict[str, object]) -> dict[str, list[int]]:
    rows = source.get("target_views")
    if not isinstance(rows, list) or not rows:
        raise ModelEvidenceError("source.target_views must be a non-empty list.")
    result: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ModelEvidenceError(f"source.target_views[{index}] must be a mapping.")
        view_id = required_string(row, "selection_view_id")
        mask = row.get("target_mask")
        if not isinstance(mask, list) or len(mask) != 4 or set(mask) - {0, 1, 0.0, 1.0}:
            raise ModelEvidenceError(f"source.target_views[{index}].target_mask must contain four binary values.")
        if view_id in result:
            raise ModelEvidenceError(f"source.target_views contains duplicate view {view_id!r}.")
        result[view_id] = [int(value) for value in mask]
    return dict(sorted(result.items()))


def support_by_view(
    screen: dict[str, object],
    *,
    support_field: str,
    expected_views: tuple[str, ...],
    expected_model_id: str,
    expected_model_role: str,
    expected_evidence_basis: str,
    expected_representation_id: str,
) -> dict[str, dict[str, object]]:
    rows = screen.get(support_field)
    if not isinstance(rows, list):
        raise ModelEvidenceError(f"response_metric_screen.{support_field} must be a list.")
    result: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ModelEvidenceError(f"response_metric_screen.{support_field}[{index}] must be a mapping.")
        view_id = required_string(row, "selection_view_id")
        if row.get("model_id") != expected_model_id or row.get("representation_id") != expected_representation_id:
            raise ModelEvidenceError(f"{support_field} must identify the expected model and representation.")
        if row.get("model_role") != expected_model_role or row.get("evidence_basis") != expected_evidence_basis:
            raise ModelEvidenceError(f"{support_field} has incorrect model-role or evidence-basis semantics.")
        if view_id in result:
            raise ModelEvidenceError(f"{support_field} contains duplicate view {view_id!r}.")
        result[view_id] = dict(row)
    if tuple(sorted(result)) != expected_views:
        raise ModelEvidenceError(f"{support_field} has views {tuple(sorted(result))}; expected {expected_views}.")
    return result


def per_view_evidence(
    view_ids: tuple[str, ...],
    *,
    views: dict[str, list[int]],
    campaign_model: dict[str, object],
    challenger: dict[str, object],
    baseline: dict[str, object],
    campaign_support: dict[str, dict[str, object]],
    challenger_support: dict[str, dict[str, object]],
) -> dict[str, object]:
    campaign_ordering = required_mapping(campaign_model, "target_view_ordering")
    challenger_ordering = required_mapping(challenger, "target_view_ordering")
    baseline_ordering = required_mapping(baseline, "target_view_ordering")
    return {
        view_id: {
            "target_mask": views[view_id],
            "campaign_model_ordering": campaign_ordering[view_id],
            "best_fixed_challenger_ordering": challenger_ordering[view_id],
            "baseline_ordering": baseline_ordering[view_id],
            "retrospective_campaign_model_greedy_support": campaign_support[view_id],
            "retrospective_best_fixed_challenger_greedy_support": challenger_support[view_id],
        }
        for view_id in view_ids
    }


def upstream_artifacts(source: dict[str, object]) -> dict[str, object]:
    campaign_config = required_mapping(required_mapping(source, "stress_campaign"), "config")
    selection_config = required_mapping(required_mapping(source, "response_measurement_selection"), "config")
    return {
        "campaign_config": {
            "path": required_string(campaign_config, "path"),
            "sha256": sha256_digest(campaign_config.get("sha256"), "campaign config"),
        },
        "response_measurement_selection": {
            "path": required_string(selection_config, "path"),
            "sha256": sha256_digest(selection_config.get("sha256"), "response measurement selection"),
        },
        "response_x_matrix_sha256": sha256_digest(source.get("response_x_matrix_sha256"), "response X matrix"),
    }


def corpus_snapshot(source: dict[str, object], *, screen: dict[str, object]) -> dict[str, int]:
    reader_counts = required_mapping(required_mapping(source, "reader_bundle"), "counts")
    binding = required_mapping(source, "candidate_identity_binding")
    selection = required_mapping(source, "response_measurement_selection")
    return {
        "model_screen_candidate_count": nonnegative_integer(screen, "model_screen_candidate_count"),
        "reader_experiment_count": nonnegative_integer(reader_counts, "experiments"),
        "unique_reader_design_count": nonnegative_integer(reader_counts, "unique_design_ids"),
        "candidate_universe_count": nonnegative_integer(binding, "candidate_count"),
        "candidate_binding_count": nonnegative_integer(binding, "binding_count"),
        "screen_measurement_row_count": nonnegative_integer(selection, "row_count"),
    }


def upstream_manifests(
    source: dict[str, object],
    *,
    label_truth: dict[str, object],
    metastudy_manifest_sha256: str,
) -> dict[str, object]:
    reader_manifest = required_mapping(required_mapping(source, "reader_bundle"), "manifest")
    binding = required_mapping(source, "candidate_identity_binding")
    binding_files = binding.get("files")
    if not isinstance(binding_files, list):
        raise ModelEvidenceError("source.candidate_identity_binding.files must be a list.")
    manifests = [row for row in binding_files if isinstance(row, dict) and row.get("path") == "manifest.json"]
    if len(manifests) != 1:
        raise ModelEvidenceError("source.candidate_identity_binding.files must contain exactly one manifest.json.")
    records: dict[str, object] = {
        "metastudy": {"sha256": sha256_digest(metastudy_manifest_sha256, "metastudy manifest")},
        "reader_response_window_bundle": {
            "path": required_string(reader_manifest, "path"),
            "sha256": sha256_digest(reader_manifest.get("sha256"), "Reader bundle manifest"),
        },
        "promoter_candidate_bindings": {
            "path": required_string(manifests[0], "path"),
            "sha256": sha256_digest(manifests[0].get("sha256"), "candidate-binding manifest"),
        },
    }
    promotion = label_truth.get("observed_label_promotion_manifest")
    if promotion is not None:
        if not isinstance(promotion, dict):
            raise ModelEvidenceError("label_truth.observed_label_promotion_manifest must be null or a mapping.")
        records["observed_label_promotion"] = {
            "path": required_string(promotion, "path"),
            "sha256": sha256_digest(promotion.get("sha256"), "observed-label promotion manifest"),
        }
    return records


__all__ = [
    "corpus_snapshot",
    "per_view_evidence",
    "support_by_view",
    "target_views",
    "upstream_artifacts",
    "upstream_manifests",
]
