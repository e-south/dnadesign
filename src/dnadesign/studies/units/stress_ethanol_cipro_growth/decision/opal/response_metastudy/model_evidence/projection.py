"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/projection.py

Project a verified metastudy manifest into protocol and snapshot records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from ..runtime.publication import METASTUDY_SCHEMA_VERSION
from .contracts import ModelEvidenceError, content_digest
from .fields import (
    enum_string,
    model_record,
    model_screen_records,
    nonnegative_integer,
    required_bool,
    required_mapping,
    required_number,
    required_string,
    sha256_digest,
)
from .protocol_projection import build_protocol
from .source_evidence import (
    corpus_snapshot,
    per_view_evidence,
    support_by_view,
    target_views,
    upstream_artifacts,
    upstream_manifests,
)

DECISION_GATE_KEYS = (
    "label_truth_ready",
    "model_support_ready",
    "selection_policy_promoted",
    "synthesis_authorized",
)
LABEL_TRUTH_KEYS = {
    "state",
    "source",
    "screen_source_scope",
    "screen_source_label_truth_role",
    "label_source_state",
    "observed_label_promotion_manifest",
}


@dataclass(frozen=True)
class ModelEvidenceProjection:
    protocol: dict[str, object]
    protocol_digest: str
    snapshot: dict[str, object]


def project_verified_manifest(
    manifest: dict[str, object],
    *,
    metastudy_manifest_sha256: str,
) -> ModelEvidenceProjection:
    """Build comparable scientific evidence without OPAL operational state."""

    if manifest.get("schema_version") != METASTUDY_SCHEMA_VERSION:
        raise ModelEvidenceError(f"unsupported metastudy manifest schema: {manifest.get('schema_version')!r}.")
    source = required_mapping(manifest, "source")
    screen = required_mapping(manifest, "response_metric_screen")
    label_truth = required_mapping(manifest, "label_truth")
    gates = required_mapping(manifest, "decision_gates")
    _validate_decision_gates(gates, screen=screen, label_truth=label_truth)

    campaign_model = model_record(screen, "campaign_model_screen", expected_role="campaign_model")
    challenger = model_record(screen, "best_fixed_model_screen", expected_role="fixed_challenger")
    baseline = model_record(screen, "baseline_model_screen", expected_role="baseline")
    views = target_views(source)
    view_ids = tuple(sorted(views))
    _validate_model_views(
        view_ids,
        campaign_model=campaign_model,
        challenger=challenger,
        baseline=baseline,
    )
    campaign_support = support_by_view(
        screen,
        support_field="campaign_greedy_support",
        expected_views=view_ids,
        expected_model_id=str(campaign_model["model_id"]),
        expected_model_role="campaign_model",
        expected_evidence_basis="configured_campaign_model",
        expected_representation_id=str(campaign_model["representation_id"]),
    )
    challenger_support = support_by_view(
        screen,
        support_field="best_fixed_challenger_greedy_support",
        expected_views=view_ids,
        expected_model_id=str(challenger["model_id"]),
        expected_model_role="fixed_challenger",
        expected_evidence_basis="best_fixed_challenger",
        expected_representation_id=str(challenger["representation_id"]),
    )
    protocol = build_protocol(
        source=source,
        screen=screen,
        label_truth=label_truth,
        campaign_model=campaign_model,
        views=views,
    )
    snapshot = _snapshot(
        source=source,
        screen=screen,
        label_truth=label_truth,
        gates=gates,
        campaign_model=campaign_model,
        challenger=challenger,
        baseline=baseline,
        views=views,
        campaign_support=campaign_support,
        challenger_support=challenger_support,
        metastudy_manifest_sha256=metastudy_manifest_sha256,
    )
    return ModelEvidenceProjection(protocol=protocol, protocol_digest=content_digest(protocol), snapshot=snapshot)


def _snapshot(
    *,
    source: dict[str, object],
    screen: dict[str, object],
    label_truth: dict[str, object],
    gates: dict[str, object],
    campaign_model: dict[str, object],
    challenger: dict[str, object],
    baseline: dict[str, object],
    views: dict[str, list[int]],
    campaign_support: dict[str, dict[str, object]],
    challenger_support: dict[str, dict[str, object]],
    metastudy_manifest_sha256: str,
) -> dict[str, object]:
    view_ids = tuple(sorted(views))
    return {
        "study_id": "stress_ethanol_cipro_growth",
        "evidence_timing": enum_string(screen, "evidence_timing", {"retrospective", "prospective"}),
        "source_metastudy": {
            "schema_version": METASTUDY_SCHEMA_VERSION,
            "manifest_sha256": sha256_digest(metastudy_manifest_sha256, "metastudy manifest"),
            "status": required_string(screen, "status"),
        },
        "label_truth": dict(label_truth),
        "decision_gates": {key: required_bool(gates, key) for key in DECISION_GATE_KEYS},
        "campaign_model": campaign_model,
        "best_fixed_challenger": challenger,
        "baseline": baseline,
        "prespecified_model_screens": model_screen_records(screen),
        "review_calibration_by_selection_view": dict(required_mapping(screen, "review_calibration_by_selection_view")),
        "per_view_evidence": per_view_evidence(
            view_ids,
            views=views,
            campaign_model=campaign_model,
            challenger=challenger,
            baseline=baseline,
            campaign_support=campaign_support,
            challenger_support=challenger_support,
        ),
        "corpus": corpus_snapshot(source, screen=screen),
        "repeats": {
            "repeated_design_count": nonnegative_integer(screen, "repeated_design_count"),
            "maximum_screen_source_to_cross_experiment_median_abs_difference": required_number(
                screen,
                "maximum_screen_source_to_cross_experiment_median_abs_difference",
            ),
            "label_source_state": required_string(label_truth, "label_source_state"),
        },
        "upstream_manifests": upstream_manifests(
            source,
            label_truth=label_truth,
            metastudy_manifest_sha256=metastudy_manifest_sha256,
        ),
        "upstream_artifacts": upstream_artifacts(source),
        "operational_state_boundary": (
            "OPAL campaign initialization, rounds, predictions, selections, and ledgers are operational state "
            "and are not part of this scientific model-evidence checkpoint."
        ),
    }


def _validate_decision_gates(
    gates: dict[str, object],
    *,
    screen: dict[str, object],
    label_truth: dict[str, object],
) -> None:
    label_state = _validate_label_truth(label_truth)
    missing = [key for key in (*DECISION_GATE_KEYS, "opal_operational_state_included") if key not in gates]
    if missing:
        raise ModelEvidenceError(f"decision_gates.{missing[0]} is required.")
    values = {key: required_bool(gates, key) for key in DECISION_GATE_KEYS}
    if required_bool(gates, "opal_operational_state_included"):
        raise ModelEvidenceError("decision_gates.opal_operational_state_included must be false.")
    if values["model_support_ready"] != required_bool(screen, "model_support_ready"):
        raise ModelEvidenceError("decision_gates.model_support_ready disagrees with the campaign model screen.")
    if values["label_truth_ready"] != (label_state == "promoted"):
        raise ModelEvidenceError("decision_gates.label_truth_ready disagrees with label_truth.state.")
    if values["synthesis_authorized"] and not all(values[key] for key in DECISION_GATE_KEYS[:-1]):
        raise ModelEvidenceError("synthesis authorization requires all preceding decision gates.")


def _validate_label_truth(label_truth: dict[str, object]) -> str:
    if set(label_truth) != LABEL_TRUTH_KEYS:
        raise ModelEvidenceError("label_truth fields disagree with the current manifest contract.")
    state = enum_string(label_truth, "state", {"not_ready", "promoted"})
    source_state = enum_string(label_truth, "label_source_state", {"not_verified", "verified"})
    promotion = label_truth["observed_label_promotion_manifest"]
    if state == "promoted":
        if source_state != "verified":
            raise ModelEvidenceError("promoted label truth requires a verified label source.")
        if not isinstance(promotion, dict):
            raise ModelEvidenceError("promoted label truth requires a promotion manifest.")
    else:
        if source_state != "not_verified":
            raise ModelEvidenceError("not-ready label truth cannot claim a verified label source.")
        if promotion is not None:
            raise ModelEvidenceError("not-ready label truth cannot include a promotion manifest.")
    return state


def _validate_model_views(view_ids: tuple[str, ...], **records: dict[str, object]) -> None:
    for label, record in records.items():
        observed = tuple(sorted(required_mapping(record, "target_view_ordering")))
        if observed != view_ids:
            raise ModelEvidenceError(f"{label}.target_view_ordering has views {observed}; expected {view_ids}.")


__all__ = ["ModelEvidenceProjection", "project_verified_manifest"]
