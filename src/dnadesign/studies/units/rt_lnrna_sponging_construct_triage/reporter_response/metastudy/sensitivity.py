"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity.py

Typed non-selectable reporter-response sensitivity evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable

from .. import profile_to_dict
from ..profile import EndpointReduction, TimeWindowReduction
from .audits import profile_audit_payload
from .contracts import (
    DEFAULT_PROTOCOL,
    MaterializationAttemptReceipt,
    MetastudyContractError,
    ProfileEvidence,
    SensitivityEvaluation,
    canonical_digest,
)
from .evidence_projection import ProfileEvidenceProjection, parse_profile_evidence_projection
from .sensitivity_coverage import (
    SensitivityCoverageLedger,
    parse_sensitivity_coverage,
    sensitivity_coverage_payload,
    sensitivity_profile_coordinate_key,
    validate_sensitivity_coverage_set,
)

SENSITIVITY_EVIDENCE_CONTRACT_ID = "rt_lnrna_reporter_response_sensitivity_evidence.v3"


def evaluate_sensitivity(
    evidence: Iterable[ProfileEvidence | ProfileEvidenceProjection],
) -> tuple[SensitivityEvaluation, ...]:
    """Summarize admissible non-selectable dose, endpoint, and alternate-window evidence."""

    grouped: dict[tuple[str, float], list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in evidence:
        if not isinstance(row, (ProfileEvidence, ProfileEvidenceProjection)):
            raise MetastudyContractError("sensitivity evidence rows must be ProfileEvidence")
        reduction = row.profile.reduction
        if isinstance(reduction, EndpointReduction):
            if reduction.recorded_time_h not in DEFAULT_PROTOCOL.endpoint_sensitivity_h:
                raise MetastudyContractError("endpoint sensitivity time is undeclared")
            grouped[("endpoint", reduction.recorded_time_h)].append(row)
        elif isinstance(reduction, TimeWindowReduction):
            width = reduction.recorded_end_time_h - reduction.recorded_start_time_h
            if width not in DEFAULT_PROTOCOL.centered_window_sensitivity_widths_h:
                raise MetastudyContractError("centered-window sensitivity width is undeclared")
            grouped[("centered_window", width)].append(row)
        else:
            raise MetastudyContractError("sensitivity reduction is undeclared")
        for dose in row.profile.dose_grid_uM:
            if dose in DEFAULT_PROTOCOL.sensitivity_doses_uM:
                grouped[("dose", dose)].append(row)
    return tuple(
        SensitivityEvaluation(
            kind=kind,
            value=value,
            profile_count=len(group_rows),
            evidence_digest=canonical_digest(
                [_profile_evidence_payload(row) for row in sorted(group_rows, key=lambda item: item.profile.profile_id)]
            ),
        )
        for (kind, value), group_rows in sorted(grouped.items())
    )


def _profile_evidence_payload(row: ProfileEvidence | ProfileEvidenceProjection) -> dict[str, object]:
    profile = profile_to_dict(row.profile) if isinstance(row, ProfileEvidence) else dict(row.profile.serialized_payload)
    return {
        "profile": profile,
        "audit": profile_audit_payload(row.audit),
    }


def sensitivity_evaluations_to_payload(
    evaluations: Iterable[SensitivityEvaluation],
) -> list[dict[str, object]]:
    """Serialize strict, canonically ordered non-selectable summaries."""

    rows = tuple(evaluations)
    payload = [
        {
            "kind": row.kind,
            "value": row.value,
            "profile_count": row.profile_count,
            "evidence_digest": row.evidence_digest,
            "selectable": row.selectable,
        }
        for row in rows
    ]
    if tuple(_sensitivity_evaluation_from_payload(row, index=index) for index, row in enumerate(payload)) != rows:
        raise MetastudyContractError("sensitivity evaluations do not match the exact contract")
    _require_canonical_order(rows)
    return payload


def parse_sensitivity_evaluations(payload: object) -> tuple[SensitivityEvaluation, ...]:
    """Parse strict summaries without accepting unknown fields or alternate order."""

    if not isinstance(payload, list):
        raise MetastudyContractError("sensitivity evaluations must be an array")
    rows = tuple(_sensitivity_evaluation_from_payload(row, index=index) for index, row in enumerate(payload))
    _require_canonical_order(rows)
    return rows


def sensitivity_evidence_payload(
    evidence: Iterable[ProfileEvidence],
    *,
    evaluations: Iterable[SensitivityEvaluation],
    coverages: Iterable[SensitivityCoverageLedger],
    attempts: Iterable[MaterializationAttemptReceipt],
) -> dict[str, object]:
    """Build the complete offline-verifiable sibling sensitivity projection."""

    coverage_rows = tuple(sorted(coverages, key=lambda row: row.experiment_id))
    rows = tuple(sorted(evidence, key=sensitivity_profile_coordinate_key))
    profile_ids = tuple(row.profile.profile_id for row in rows)
    if len(profile_ids) != len(set(profile_ids)):
        raise MetastudyContractError("sensitivity evidence profile_id values must be unique")
    summaries = tuple(evaluations)
    expected = evaluate_sensitivity(rows)
    if summaries != expected:
        raise MetastudyContractError("sensitivity summaries differ from canonical evidence evaluation")
    validate_sensitivity_coverage_set(coverage_rows, evidence=rows, attempts=tuple(attempts))
    return {
        "contract_id": SENSITIVITY_EVIDENCE_CONTRACT_ID,
        "evaluations": sensitivity_evaluations_to_payload(summaries),
        "profiles": [_profile_evidence_payload(row) for row in rows],
        "coverages": [sensitivity_coverage_payload(row) for row in coverage_rows],
    }


def verify_sensitivity_evidence_payload(
    payload: object,
    *,
    attempts: Iterable[MaterializationAttemptReceipt],
) -> tuple[SensitivityEvaluation, ...]:
    """Recompute every summary from bundled profiles and reject projection tampering."""

    if not isinstance(payload, dict) or set(payload) != {"contract_id", "evaluations", "profiles", "coverages"}:
        raise MetastudyContractError("sensitivity evidence fields do not match the exact contract")
    if payload["contract_id"] != SENSITIVITY_EVIDENCE_CONTRACT_ID:
        raise MetastudyContractError("sensitivity evidence contract_id changed")
    evaluations = parse_sensitivity_evaluations(payload["evaluations"])
    profiles = payload["profiles"]
    if not isinstance(profiles, list):
        raise MetastudyContractError("sensitivity evidence profiles must be an array")
    try:
        evidence = tuple(parse_profile_evidence_projection(row, index=index) for index, row in enumerate(profiles))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MetastudyContractError):
            raise
        raise MetastudyContractError(f"sensitivity evidence projection is invalid: {exc}") from exc
    if evaluate_sensitivity(evidence) != evaluations:
        raise MetastudyContractError("sensitivity summaries differ from bundled evidence")
    coverages_payload = payload["coverages"]
    if not isinstance(coverages_payload, list):
        raise MetastudyContractError("sensitivity coverages must be an array")
    coverages = tuple(parse_sensitivity_coverage(row, index=index) for index, row in enumerate(coverages_payload))
    attempt_rows = tuple(attempts)
    validate_sensitivity_coverage_set(coverages, evidence=evidence, attempts=attempt_rows)
    canonical = sensitivity_evidence_payload(
        evidence,
        evaluations=evaluations,
        coverages=coverages,
        attempts=attempt_rows,
    )
    if json.loads(json.dumps(canonical, allow_nan=False)) != payload:
        raise MetastudyContractError("sensitivity evidence projection is not canonical")
    return evaluations


def _sensitivity_evaluation_from_payload(payload: object, *, index: int) -> SensitivityEvaluation:
    expected = {"kind", "value", "profile_count", "evidence_digest", "selectable"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise MetastudyContractError(f"sensitivity evaluation {index} fields do not match the exact contract")
    try:
        return SensitivityEvaluation(
            kind=payload["kind"],
            value=payload["value"],
            profile_count=payload["profile_count"],
            evidence_digest=payload["evidence_digest"],
            selectable=payload["selectable"],
        )
    except TypeError as exc:
        raise MetastudyContractError(f"sensitivity evaluation {index} is invalid: {exc}") from exc


def _require_canonical_order(rows: tuple[SensitivityEvaluation, ...]) -> None:
    keys = tuple((row.kind, row.value) for row in rows)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise MetastudyContractError("sensitivity evaluations must be unique and canonically ordered")


__all__ = [
    "SENSITIVITY_EVIDENCE_CONTRACT_ID",
    "evaluate_sensitivity",
    "parse_sensitivity_evaluations",
    "sensitivity_evaluations_to_payload",
    "sensitivity_evidence_payload",
    "verify_sensitivity_evidence_payload",
]
