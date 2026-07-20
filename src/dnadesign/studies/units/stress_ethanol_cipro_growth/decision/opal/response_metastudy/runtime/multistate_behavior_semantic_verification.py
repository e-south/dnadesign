"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_semantic_verification.py

Semantic binding checks for persisted multistate behavior shadow records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_bundle_contract import SCHEMA_ID
from .multistate_behavior_normalization_verification import verify_behavior_normalization_record
from .multistate_behavior_record_fields import (
    ACTIVATION,
    SOURCE_DIGEST_FIELDS,
    mapping,
    nonnegative_int,
    positive_float,
    positive_int,
    require_fields,
    require_literals,
    unprefixed_digest,
)


@dataclass(frozen=True)
class BehaviorBundleSemantics:
    """Verified dynamic values used to check every persisted table."""

    state_ids: tuple[str, ...]
    view_masks: dict[str, tuple[int, ...]]
    softmin_scale: float
    bootstrap_samples: int
    unit_count: int
    prediction_count: int
    excluded_nonexact_unit_count: int
    protocol_id: str
    protocol_sha256: str
    source_rows_sha256: str
    prediction_run_id: str
    prediction_source_sha256: str
    prediction_raw_top_k: int

    @property
    def view_ids(self) -> tuple[str, ...]:
        return tuple(self.view_masks)

    @property
    def response_pairs(self) -> tuple[tuple[str, str], ...]:
        index = {state_id: position for position, state_id in enumerate(self.state_ids)}
        pairs: set[tuple[str, str]] = set()
        for mask in self.view_masks.values():
            on = [self.state_ids[position] for position, value in enumerate(mask) if value == 1]
            off = [self.state_ids[position] for position, value in enumerate(mask) if value == 0]
            for left in on:
                for right in off:
                    pairs.add(tuple(sorted((left, right), key=index.__getitem__)))
        return tuple(sorted(pairs, key=lambda pair: (index[pair[0]], index[pair[1]])))


def verify_behavior_record_semantics(
    manifest: dict[str, object],
    normalization: dict[str, object],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> BehaviorBundleSemantics:
    """Bind manifest and normalization records to the persisted study protocol."""

    require_fields(
        manifest,
        {
            "schema_id",
            "schema_version",
            "study_id",
            "protocol_id",
            "status",
            "activation",
            "objective_name",
            "comparator",
            "source",
            "normalization_source",
            "cohort",
            "excluded_nonexact_unit_count",
            "decision",
            "tables",
            "artifacts",
            "claim_boundary",
        },
        context="manifest",
    )
    require_literals(
        manifest,
        {
            "schema_id": SCHEMA_ID,
            "schema_version": "1",
            "study_id": protocol.study_id,
            "protocol_id": protocol.protocol_id,
            "status": "shadow_only",
            "activation": ACTIVATION,
            "objective_name": protocol.objective_name,
            "claim_boundary": "shadow_evidence_only_no_campaign_activation_or_synthesis_authorization",
        },
        context="manifest",
    )
    comparator = mapping(manifest["comparator"], context="manifest.comparator")
    require_literals(
        comparator,
        {
            "objective_name": protocol.comparator_objective_name,
            "score_channel": protocol.comparator_score_channel,
            "direction": protocol.comparator_direction,
            "comparison_role": protocol.comparison_role,
        },
        context="manifest.comparator",
        exact=True,
    )
    require_literals(
        mapping(manifest["decision"], context="manifest.decision"),
        {
            "promotion_decision": "no_go",
            "campaign_activation": "prohibited",
            "synthesis": "prohibited",
        },
        context="manifest.decision",
        exact=True,
    )

    verify_behavior_normalization_record(normalization, protocol=protocol)
    source = mapping(manifest["source"], context="manifest.source")
    require_fields(source, {"prediction", *SOURCE_DIGEST_FIELDS}, context="manifest.source")
    normalization_source = mapping(normalization["source"], context="normalization.source")
    if manifest["normalization_source"] != normalization_source:
        raise ValueError("manifest normalization source disagrees with normalization.json.")
    for field in SOURCE_DIGEST_FIELDS:
        if source[field] != normalization_source[field]:
            raise ValueError(f"manifest source {field!r} disagrees with normalization.json.")

    normalization_values = mapping(normalization["normalization"], context="normalization.normalization")
    cohort = mapping(manifest["cohort"], context="manifest.cohort")
    require_fields(
        cohort,
        {
            "cohort_id",
            "unit_count",
            "candidate_count",
            "reader_experiment_count",
            "unit_ids_sha256",
            "source_rows_sha256",
        },
        context="manifest.cohort",
    )
    for field in ("cohort_id", "unit_count", "candidate_count", "reader_experiment_count"):
        if cohort[field] != normalization_values[field]:
            raise ValueError(f"manifest cohort {field!r} disagrees with normalization.json.")
    source_rows_sha256 = unprefixed_digest(
        cohort["source_rows_sha256"],
        field="cohort.source_rows_sha256",
    )
    if normalization_source["source_rows_sha256"] != f"sha256:{source_rows_sha256}":
        raise ValueError("manifest cohort source-row digest disagrees with normalization.json.")
    unprefixed_digest(cohort["unit_ids_sha256"], field="cohort.unit_ids_sha256")
    excluded = nonnegative_int(
        manifest["excluded_nonexact_unit_count"],
        field="excluded_nonexact_unit_count",
    )
    if excluded != normalization_values["excluded_nonexact_unit_count"]:
        raise ValueError("excluded nonexact unit count disagrees with normalization.json.")

    prediction = mapping(source["prediction"], context="manifest.source.prediction")
    return BehaviorBundleSemantics(
        state_ids=tuple(str(value) for value in normalization["assay"]["state_ids"]),
        view_masks={
            str(row["id"]): tuple(int(value) for value in row["target_mask"]) for row in normalization["target_views"]
        },
        softmin_scale=positive_float(normalization_values["softmin_scale"], field="softmin_scale"),
        bootstrap_samples=positive_int(normalization_values["bootstrap_samples"], field="bootstrap_samples"),
        unit_count=positive_int(normalization_values["unit_count"], field="unit_count"),
        prediction_count=positive_int(prediction.get("candidate_count"), field="prediction.candidate_count"),
        excluded_nonexact_unit_count=excluded,
        protocol_id=protocol.protocol_id,
        protocol_sha256=f"sha256:{protocol.source_sha256}",
        source_rows_sha256=f"sha256:{source_rows_sha256}",
        prediction_run_id=str(prediction.get("run_id")),
        prediction_source_sha256=str(prediction.get("ledger_sha256")),
        prediction_raw_top_k=protocol.prediction_raw_top_k,
    )


__all__ = ["BehaviorBundleSemantics", "verify_behavior_record_semantics"]
