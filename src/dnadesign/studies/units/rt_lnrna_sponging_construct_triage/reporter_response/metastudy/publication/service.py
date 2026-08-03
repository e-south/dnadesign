"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/publication/service.py

Create reporter-response publications through shared artifact mechanics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

from ..acquisition_projection import acquisition_projection_payload, build_acquisition_projection
from ..contracts._values import MetastudyContractError
from ..contracts.decision import MetastudyDecision, decision_is_evidence_bearing
from ..contracts.decision_codec import decision_to_dict, validate_decision_payload
from ..contracts.objective import DEFAULT_OBJECTIVE_READINESS, ObjectiveReadiness
from ..contracts.profile import ProfileEvidence
from ..contracts.sensitivity import SensitivityEvaluation
from ..evaluation.evidence import decision_evidence_payload
from ..sensitivity import sensitivity_evidence_payload
from ..sensitivity_coverage import SensitivityCoverageLedger
from .report import _render_report
from .verification import _PUBLICATION_SCHEMA_ID, _digest, _verify_publication_bundle, verify_publication

_PRIVATE_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


def publish_metastudy(
    decision: MetastudyDecision,
    destination: Path,
    *,
    primary_evidence: Iterable[ProfileEvidence] = (),
    sensitivity_evidence: Iterable[ProfileEvidence] = (),
    sensitivity_evaluations: Iterable[SensitivityEvaluation] = (),
    sensitivity_coverages: Iterable[SensitivityCoverageLedger] = (),
    objective_readiness: ObjectiveReadiness = DEFAULT_OBJECTIVE_READINESS,
) -> Path:
    """Validate all study semantics before publishing one immutable bundle."""

    bundle = _build_bundle(
        decision,
        primary_evidence=primary_evidence,
        sensitivity_evidence=sensitivity_evidence,
        sensitivity_evaluations=sensitivity_evaluations,
        sensitivity_coverages=sensitivity_coverages,
        objective_readiness=objective_readiness,
    )
    _verify_publication_bundle(bundle)
    publication = CreateOnlyDirectoryPublication.prepare(
        destination,
        published_root_mode=_PRIVATE_MODE,
    )
    with publication:
        for name, payload in sorted(bundle.items()):
            member = publication.stage / name
            member.write_bytes(payload)
            member.chmod(_PRIVATE_FILE_MODE)
        publication.publish(required_manifest="manifest.json")
        try:
            verify_publication(publication.final)
        except BaseException as verification_error:
            try:
                removed = publication.rollback()
            except BaseException as rollback_error:
                raise PublicationError(
                    "Invalid reporter-response publication could not be rolled back safely"
                ) from rollback_error
            if not removed:
                raise PublicationError(
                    "Invalid reporter-response publication remains because rollback ownership could not be proved"
                ) from verification_error
            raise
        return publication.final


def _build_bundle(
    decision: MetastudyDecision,
    *,
    primary_evidence: Iterable[ProfileEvidence],
    sensitivity_evidence: Iterable[ProfileEvidence],
    sensitivity_evaluations: Iterable[SensitivityEvaluation],
    sensitivity_coverages: Iterable[SensitivityCoverageLedger],
    objective_readiness: ObjectiveReadiness,
) -> dict[str, bytes]:
    payload = decision_to_dict(decision)
    validate_decision_payload(payload)
    if objective_readiness != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("publication objective readiness differs from the study-owned gate")
    evidence_rows = tuple(primary_evidence)
    evidence_payload = None
    acquisition_payload = None
    if decision_is_evidence_bearing(payload):
        if not evidence_rows:
            raise MetastudyContractError("evidence-bearing publication requires canonical profile evidence")
        evidence_payload = decision_evidence_payload(evidence_rows, decision=decision)
        if decision.selected_reduction is not None:
            acquisition_payload = acquisition_projection_payload(
                build_acquisition_projection(evidence_rows, selected_reduction=decision.selected_reduction)
            )
    elif evidence_rows:
        raise MetastudyContractError("readiness-only publication does not accept profile evidence")
    sensitivity_payload = sensitivity_evidence_payload(
        tuple(sensitivity_evidence),
        evaluations=tuple(sensitivity_evaluations),
        coverages=tuple(sensitivity_coverages),
        attempts=decision.materialization_attempts,
    )
    report_bytes = _render_report(payload).encode("utf-8")
    sensitivity_bytes = _json_bytes(sensitivity_payload)
    bundle = {
        "report.md": report_bytes,
        "sensitivity.json": sensitivity_bytes,
    }
    manifest = {
        "schema_id": _PUBLICATION_SCHEMA_ID,
        "decision": payload,
        "objective_readiness": asdict(objective_readiness),
        "report_digest": _digest(report_bytes),
        "sensitivity_file_digest": _digest(sensitivity_bytes),
    }
    if evidence_payload is not None:
        evidence_bytes = _json_bytes(evidence_payload)
        bundle["evidence.json"] = evidence_bytes
        manifest["evidence_file_digest"] = _digest(evidence_bytes)
    if acquisition_payload is not None:
        acquisition_bytes = _json_bytes(acquisition_payload)
        bundle["acquisition.json"] = acquisition_bytes
        manifest["acquisition_file_digest"] = _digest(acquisition_bytes)
    bundle["manifest.json"] = _json_bytes(manifest)
    return bundle


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


__all__ = ["publish_metastudy"]
