"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/evaluation_baseline_parser.py

Schema parsing for the round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import STUDY_ALIAS_NAMESPACE
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings.study_alias_registry import (
    REGISTRY_SCHEMA_ID,
)

from .evaluation_baseline_artifacts import load_frozen_artifact, load_frozen_file
from .evaluation_baseline_contracts import (
    CAMPAIGN_CONFIG_PATH,
    CAMPAIGN_SLUG,
    CLAIM_LIMIT_STATEMENT,
    COMPARISON_METHOD,
    COMPARISON_ROLE,
    COMPARISON_STATEMENT,
    COMPARISON_SUBSET_COUNT,
    COMPARISON_SUBSET_SIZE,
    EXPECTED_CANDIDATE_OUTPUTS,
    EXPECTED_ENDPOINTS,
    EXPECTED_EVALUATION_CONVENTIONS,
    EXPECTED_QUOTAS,
    OBJECTIVE_ID,
    PROTOCOL_ID,
    ROOT_FIELDS,
    RUN_ID,
    SCHEMA_ID,
    SCHEMA_VERSION,
    VIEW_PRIORITY,
    FrozenAllocation,
    FrozenFile,
    MsrbEvaluationBaselineError,
    ParsedBaseline,
)


def parse_baseline(payload: object, *, root: Path) -> ParsedBaseline:
    """Parse the exact receipt schema and return a typed verification plan."""

    raw = _mapping(payload, context="baseline")
    _exact_fields(raw, ROOT_FIELDS, context="baseline")
    _literal(raw, "schema_id", SCHEMA_ID, context="baseline")
    _literal(raw, "schema_version", SCHEMA_VERSION, context="baseline")
    _literal(raw, "study_id", "stress_ethanol_cipro_growth", context="baseline")
    _literal(raw, "baseline_id", "secg_msrb_round0_evaluation_v1", context="baseline")
    campaign_config, selection_allocation_api_version = _parse_campaign(raw["campaign"], root=root)

    artifacts = _mapping(raw["artifacts"], context="artifacts")
    _exact_fields(artifacts, {"prediction_ledger", "selection_batch", "labels_used"}, context="artifacts")
    prediction_ledger = load_frozen_artifact(
        artifacts["prediction_ledger"],
        root=root,
        artifact_id="prediction_ledger",
        expected_count=154_785,
    )
    selection_batch = load_frozen_artifact(
        artifacts["selection_batch"],
        root=root,
        artifact_id="selection_batch",
        expected_count=18,
    )
    labels_used = load_frozen_artifact(
        artifacts["labels_used"],
        root=root,
        artifact_id="labels_used",
        expected_count=27,
    )

    allocations = _allocations(raw["allocations"])
    alias_registry_path = _alias_registry_path(raw["alias_registry"])
    comparison_ids, comparison_method, comparison_subset_size, comparison_subset_count = _comparison_set(
        raw["comparison_set"]
    )
    endpoint_ids = _evaluation(raw["evaluation"])
    claims = _claims(raw["claim_limits"])
    return ParsedBaseline(
        campaign_config=campaign_config,
        selection_allocation_api_version=selection_allocation_api_version,
        prediction_ledger=prediction_ledger,
        selection_batch=selection_batch,
        labels_used=labels_used,
        allocations=allocations,
        alias_registry_path=alias_registry_path,
        comparison_candidate_ids=comparison_ids,
        comparison_method=comparison_method,
        comparison_subset_size=comparison_subset_size,
        comparison_subset_count=comparison_subset_count,
        endpoint_ids=endpoint_ids,
        acquisition_efficacy_claim=claims[0],
        hill_climb_claim=claims[1],
        synthesis_authorization=claims[2],
        claim_limit_statement=claims[3],
    )


def _parse_campaign(value: object, *, root: Path) -> tuple[FrozenFile, str]:
    campaign = _mapping(value, context="campaign")
    _exact_fields(
        campaign,
        {
            "slug",
            "protocol_id",
            "run_id",
            "round_index",
            "objective_id",
            "selection_policy",
            "config",
            "selection_allocation_api_version",
        },
        context="campaign",
    )
    _literal(campaign, "slug", CAMPAIGN_SLUG, context="campaign", label="campaign slug")
    _literal(campaign, "protocol_id", PROTOCOL_ID, context="campaign")
    _literal(campaign, "run_id", RUN_ID, context="campaign", label="campaign run ID")
    _literal(campaign, "round_index", 0, context="campaign", label="campaign round index")
    _literal(campaign, "objective_id", OBJECTIVE_ID, context="campaign")
    _literal(
        campaign,
        "selection_policy",
        "greedy_top6_per_view_round_robin_sequence_unique",
        context="campaign",
    )
    config = load_frozen_file(campaign["config"], root=root, source_id="campaign.config")
    if config.path != CAMPAIGN_CONFIG_PATH:
        raise MsrbEvaluationBaselineError(
            "campaign.config.path mismatch: "
            f"expected {CAMPAIGN_CONFIG_PATH.as_posix()!r}, observed {config.path.as_posix()!r}."
        )
    allocation_api_version = _text(
        campaign["selection_allocation_api_version"],
        context="campaign.selection_allocation_api_version",
    )
    if allocation_api_version != "1":
        raise MsrbEvaluationBaselineError(
            "campaign.selection_allocation_api_version must equal the frozen public API version '1'."
        )
    return config, allocation_api_version


def _alias_registry_path(value: object) -> str:
    raw = _mapping(value, context="alias_registry")
    _exact_fields(raw, {"schema_id", "namespace", "path"}, context="alias_registry")
    _literal(raw, "schema_id", REGISTRY_SCHEMA_ID, context="alias_registry")
    _literal(raw, "namespace", STUDY_ALIAS_NAMESPACE, context="alias_registry")
    path = Path(_text(raw["path"], context="alias_registry.path"))
    if path.is_absolute() or ".." in path.parts:
        raise MsrbEvaluationBaselineError("alias_registry.path must be repository-relative without '..'.")
    return path.as_posix()


def _allocations(value: object) -> tuple[FrozenAllocation, ...]:
    if not isinstance(value, list) or len(value) != 18:
        raise MsrbEvaluationBaselineError("allocations must contain exactly 18 rows.")
    rows: list[FrozenAllocation] = []
    for index, item in enumerate(value):
        raw = _mapping(item, context=f"allocations[{index}]")
        _exact_fields(
            raw,
            {"study_alias", "candidate_id", "sequence_sha256", "selection_view", "allocation_slot"},
            context=f"allocations[{index}]",
        )
        rows.append(
            FrozenAllocation(
                study_alias=_text(raw["study_alias"], context=f"allocations[{index}].study_alias"),
                candidate_id=_text(raw["candidate_id"], context=f"allocations[{index}].candidate_id"),
                sequence_sha256=_sha256_text(
                    raw["sequence_sha256"],
                    context=f"allocations[{index}].sequence_sha256",
                ),
                selection_view=_text(raw["selection_view"], context=f"allocations[{index}].selection_view"),
                allocation_slot=_positive_integer(
                    raw["allocation_slot"],
                    context=f"allocations[{index}].allocation_slot",
                ),
            )
        )
    _unique([row.candidate_id for row in rows], label="allocation candidate IDs")
    _unique([row.sequence_sha256 for row in rows], label="allocation sequence digests")
    _unique([row.study_alias for row in rows], label="allocation study aliases")
    quotas = Counter(row.selection_view for row in rows)
    if quotas != EXPECTED_QUOTAS:
        raise MsrbEvaluationBaselineError(
            f"allocation quotas must be exactly {EXPECTED_QUOTAS}; observed {dict(quotas)}."
        )
    for view_id in VIEW_PRIORITY:
        slots = sorted(row.allocation_slot for row in rows if row.selection_view == view_id)
        if slots != list(range(1, 7)):
            raise MsrbEvaluationBaselineError(
                f"allocation slots for {view_id!r} must be exactly 1..6; observed {slots}."
            )
    return tuple(rows)


def _comparison_set(value: object) -> tuple[tuple[str, ...], str, int, int]:
    raw = _mapping(value, context="comparison_set")
    _exact_fields(
        raw,
        {"role", "source_artifact", "generator", "candidate_ids", "physical_random_control", "statement"},
        context="comparison_set",
    )
    _literal(raw, "role", COMPARISON_ROLE, context="comparison_set")
    _literal(raw, "source_artifact", "labels_used", context="comparison_set")
    _literal(raw, "physical_random_control", False, context="comparison_set")
    _literal(raw, "statement", COMPARISON_STATEMENT, context="comparison_set")
    candidate_ids = _identifier_list(raw["candidate_ids"], context="comparison_set.candidate_ids")
    if len(candidate_ids) != 27:
        raise MsrbEvaluationBaselineError("comparison_set must contain exactly 27 candidate IDs.")
    generator = _mapping(raw["generator"], context="comparison_set.generator")
    _exact_fields(
        generator,
        {"method", "subset_size", "subset_count", "random_seed"},
        context="comparison_set.generator",
    )
    _literal(generator, "method", COMPARISON_METHOD, context="comparison_set.generator")
    _literal(generator, "subset_size", COMPARISON_SUBSET_SIZE, context="comparison_set.generator")
    _literal(generator, "subset_count", COMPARISON_SUBSET_COUNT, context="comparison_set.generator")
    _literal(generator, "random_seed", None, context="comparison_set.generator")
    if comb(len(candidate_ids), COMPARISON_SUBSET_SIZE) != COMPARISON_SUBSET_COUNT:
        raise MsrbEvaluationBaselineError("comparison_set.generator subset_count is not exhaustive for its inputs.")
    return candidate_ids, COMPARISON_METHOD, COMPARISON_SUBSET_SIZE, COMPARISON_SUBSET_COUNT


def _evaluation(value: object) -> tuple[str, ...]:
    raw = _mapping(value, context="evaluation")
    _exact_fields(raw, {"conventions", "endpoints", "required_candidate_outputs"}, context="evaluation")
    conventions = _mapping(raw["conventions"], context="evaluation.conventions")
    if conventions != EXPECTED_EVALUATION_CONVENTIONS:
        raise MsrbEvaluationBaselineError("evaluation conventions do not match the frozen round-0 definitions.")
    endpoints = raw["endpoints"]
    if not isinstance(endpoints, list):
        raise MsrbEvaluationBaselineError("evaluation.endpoints must be a list.")
    observed: list[tuple[str, str, str, str]] = []
    for index, item in enumerate(endpoints):
        endpoint = _mapping(item, context=f"evaluation.endpoints[{index}]")
        _exact_fields(endpoint, {"id", "method", "unit", "scope"}, context=f"evaluation.endpoints[{index}]")
        observed.append(
            tuple(
                _text(endpoint[field], context=f"evaluation.endpoints[{index}].{field}")
                for field in ("id", "method", "unit", "scope")
            )
        )
    if tuple(observed) != EXPECTED_ENDPOINTS:
        raise MsrbEvaluationBaselineError("evaluation endpoints do not match the frozen round-0 definitions.")
    outputs = _identifier_list(raw["required_candidate_outputs"], context="evaluation.required_candidate_outputs")
    if outputs != EXPECTED_CANDIDATE_OUTPUTS:
        raise MsrbEvaluationBaselineError("evaluation required candidate outputs do not match the frozen definition.")
    return tuple(endpoint[0] for endpoint in observed)


def _claims(value: object) -> tuple[str, str, str, str]:
    raw = _mapping(value, context="claim_limits")
    _exact_fields(
        raw,
        {"allowed_interpretation", "acquisition_efficacy", "hill_climb", "synthesis_authorization", "statement"},
        context="claim_limits",
    )
    _literal(
        raw,
        "allowed_interpretation",
        "prospectively_frozen_greedy_learning_probe_evaluation",
        context="claim_limits",
    )
    _literal(raw, "acquisition_efficacy", "not_supported", context="claim_limits")
    _literal(raw, "hill_climb", "not_supported", context="claim_limits")
    _literal(raw, "synthesis_authorization", "prohibited", context="claim_limits")
    _literal(raw, "statement", CLAIM_LIMIT_STATEMENT, context="claim_limits")
    return "not_supported", "not_supported", "prohibited", CLAIM_LIMIT_STATEMENT


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MsrbEvaluationBaselineError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _exact_fields(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise MsrbEvaluationBaselineError(
            f"{context} fields do not match v1: expected {sorted(expected)}, observed {sorted(value)}."
        )


def _literal(
    value: dict[str, Any],
    field: str,
    expected: object,
    *,
    context: str,
    label: str | None = None,
) -> None:
    if value[field] != expected:
        raise MsrbEvaluationBaselineError(
            f"{label or f'{context}.{field}'} mismatch: expected {expected!r}, observed {value[field]!r}."
        )


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MsrbEvaluationBaselineError(f"{context} must be a non-empty string.")
    return value.strip()


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MsrbEvaluationBaselineError(f"{context} must be a positive integer.")
    return value


def _sha256_text(value: object, *, context: str) -> str:
    text = _text(value, context=context).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise MsrbEvaluationBaselineError(f"{context} must be a lowercase SHA-256 digest.")
    return text


def _identifier_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MsrbEvaluationBaselineError(f"{context} must be a list.")
    items = tuple(_text(item, context=context) for item in value)
    _unique(list(items), label=context)
    return items


def _unique(values: list[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise MsrbEvaluationBaselineError(f"{label} must be unique.")
