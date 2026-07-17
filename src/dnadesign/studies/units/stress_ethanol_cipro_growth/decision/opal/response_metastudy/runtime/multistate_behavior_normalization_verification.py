"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_normalization_verification.py

Schema and protocol checks for behavior normalization records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..evaluation.multistate_behavior_normalization import NORMALIZATION_SCHEMA_ID
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_record_fields import (
    ACTIVATION,
    SOURCE_DIGEST_FIELDS,
    mapping,
    nonnegative_int,
    positive_float,
    positive_int,
    prefixed_digest,
    require_fields,
    require_literals,
)


def verify_behavior_normalization_record(
    record: dict[str, object],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    """Require normalization.json to instantiate the current study protocol."""

    require_fields(
        record,
        {
            "schema_id",
            "schema_version",
            "study_id",
            "protocol_id",
            "status",
            "activation",
            "objective",
            "assay",
            "target_views",
            "normalization",
            "evidence_roles",
            "source",
        },
        context="normalization",
    )
    require_literals(
        record,
        {
            "schema_id": NORMALIZATION_SCHEMA_ID,
            "schema_version": "1",
            "study_id": protocol.study_id,
            "protocol_id": protocol.protocol_id,
            "status": "shadow_only",
            "activation": ACTIVATION,
        },
        context="normalization",
    )
    require_literals(
        mapping(record["objective"], context="normalization.objective"),
        {
            "name": protocol.objective_name,
            "family_weighting": protocol.family_weighting,
            "normalized_temperature": protocol.normalization.normalized_temperature,
        },
        context="normalization.objective",
        exact=True,
    )
    require_literals(
        mapping(record["assay"], context="normalization.assay"),
        {
            "state_ids": list(protocol.state_ids),
            "primary_reduction_id": protocol.primary_reduction_id,
            "fluorescence_reference": protocol.fluorescence_reference,
        },
        context="normalization.assay",
        exact=True,
    )
    expected_views = [
        {"id": view.id, "target_mask": [int(value) for value in view.target_mask]} for view in protocol.target_views
    ]
    if record["target_views"] != expected_views:
        raise ValueError("normalization target views disagree with the study protocol.")
    _verify_normalization_values(
        mapping(record["normalization"], context="normalization.normalization"),
        protocol=protocol,
    )
    expected_roles = {
        "bootstrap": protocol.normalization.bootstrap_role,
        "event_time": protocol.normalization.event_time_role,
        "repeat": protocol.normalization.repeat_role,
        "censor": protocol.normalization.censor_role,
    }
    require_literals(
        mapping(record["evidence_roles"], context="normalization.evidence_roles"),
        expected_roles,
        context="normalization.evidence_roles",
        exact=True,
    )
    source = mapping(record["source"], context="normalization.source")
    require_fields(
        source,
        {"protocol_sha256", "source_rows_sha256", *SOURCE_DIGEST_FIELDS},
        context="normalization.source",
    )
    if source["protocol_sha256"] != f"sha256:{protocol.source_sha256}":
        raise ValueError("normalization protocol digest disagrees with the persisted study protocol.")
    for field, value in source.items():
        prefixed_digest(value, field=f"normalization.source.{field}")


def _verify_normalization_values(
    values: dict[str, object],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    require_fields(
        values,
        {
            "response_scale",
            "fluorescence_scale",
            "scale_quantile",
            "quantile_method",
            "response_scale_basis",
            "fluorescence_scale_basis",
            "pair_deduplication",
            "cohort_id",
            "unit",
            "unit_count",
            "candidate_count",
            "reader_experiment_count",
            "excluded_nonexact_unit_count",
            "response_pair_count",
            "bootstrap_samples",
        },
        context="normalization.normalization",
    )
    require_literals(
        values,
        {
            "scale_quantile": protocol.normalization.scale_quantile,
            "quantile_method": protocol.normalization.quantile_method,
            "response_scale_basis": protocol.normalization.response_scale_basis,
            "fluorescence_scale_basis": protocol.normalization.fluorescence_scale_basis,
            "pair_deduplication": protocol.normalization.pair_deduplication,
            "cohort_id": protocol.normalization.cohort_id,
            "unit": protocol.normalization.unit,
        },
        context="normalization.normalization",
    )
    for field in ("response_scale", "fluorescence_scale"):
        positive_float(values[field], field=field)
    for field in (
        "unit_count",
        "candidate_count",
        "reader_experiment_count",
        "response_pair_count",
        "bootstrap_samples",
    ):
        positive_int(values[field], field=field)
    nonnegative_int(values["excluded_nonexact_unit_count"], field="excluded_nonexact_unit_count")


__all__ = ["verify_behavior_normalization_record"]
