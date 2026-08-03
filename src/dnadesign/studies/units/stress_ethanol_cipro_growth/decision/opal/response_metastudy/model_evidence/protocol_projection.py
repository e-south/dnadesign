"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/protocol_projection.py

Frozen scientific protocol projection for model-evidence trajectories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import PROTOCOL_ID, PROTOCOL_SCHEMA_VERSION
from .evaluator_protocol import evaluator_sources
from .fields import (
    enum_string,
    fixed_model_definitions,
    nonnegative_integer,
    required_mapping,
    required_number,
    required_string,
)


def build_protocol(
    *,
    source: dict[str, object],
    screen: dict[str, object],
    label_truth: dict[str, object],
    campaign_model: dict[str, object],
    views: dict[str, list[int]],
    source_manifest_schema: str,
) -> dict[str, object]:
    screen_protocol = required_mapping(screen, "response_screen_protocol")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_id": "stress_ethanol_cipro_growth",
        "evidence_name": "model-evidence trajectory",
        "status": "frozen",
        "source_manifest_schema": source_manifest_schema,
        "target_views": views,
        "label_truth_contract": {
            "state": enum_string(label_truth, "state", {"not_ready", "promoted"}),
            "source": required_string(label_truth, "source"),
            "screen_source_scope": required_string(label_truth, "screen_source_scope"),
            "screen_source_label_truth_role": required_string(label_truth, "screen_source_label_truth_role"),
            "label_source_state": required_string(label_truth, "label_source_state"),
        },
        "response_screen_protocol": dict(screen_protocol),
        "fixed_model_definitions": fixed_model_definitions(screen),
        "evaluator_sources": evaluator_sources(source),
        "campaign_model": {
            key: campaign_model[key]
            for key in (
                "model_id",
                "representation_id",
                "target_transform",
                "validation",
                "metric_scope",
                "configured_model_params",
            )
        },
        "model_support_gate": {
            "minimum_ordering_spearman": required_number(screen_protocol, "model_min_within_group_spearman"),
            "minimum_defined_group_count": nonnegative_integer(screen_protocol, "model_min_defined_group_count"),
        },
        "role_boundary": {
            "campaign_model": "configured model and primary response-window target",
            "fixed_challenger": "descriptive comparison without campaign-model promotion",
            "baseline": "reference floor for model evidence",
        },
        "operational_state": "excluded; OPAL campaign progress is tracked by OPAL, not this scientific protocol",
    }


__all__ = ["build_protocol"]
