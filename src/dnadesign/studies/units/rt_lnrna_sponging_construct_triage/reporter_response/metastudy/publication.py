"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/publication.py

Deterministic create-only meta-study publication and verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .acquisition_projection import (
    acquisition_projection_payload,
    build_acquisition_projection,
    validate_acquisition_projection_payload,
)
from .contracts._values import MetastudyContractError, canonical_digest
from .contracts.decision import (
    DEFAULT_OBJECTIVE_READINESS,
    MetastudyDecision,
    ObjectiveReadiness,
    SensitivityEvaluation,
    decision_is_evidence_bearing,
    decision_to_dict,
    objective_readiness_from_payload,
    validate_decision_payload,
)
from .contracts.materialization import EvidenceReadiness, materialization_attempt_from_payload
from .contracts.profile import ProfileEvidence
from .evaluation.evidence import decision_evidence_payload
from .evaluation.selection import reevaluate_evidence_projection
from .evidence_projection import parse_profile_evidence_projection
from .sensitivity import sensitivity_evidence_payload, verify_sensitivity_evidence_payload
from .sensitivity_coverage import SensitivityCoverageLedger

_EVIDENCE_FREE_FILES = {"manifest.json", "report.md", "sensitivity.json"}
_EVIDENCE_BEARING_FILES = {
    "manifest.json",
    "report.md",
    "sensitivity.json",
    "evidence.json",
}
_SELECTED_FILES = {
    *_EVIDENCE_BEARING_FILES,
    "acquisition.json",
}
_PUBLICATION_SCHEMA_ID = "rt_lnrna_reporter_response_metastudy_publication.v6"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4


def _rename_directory_create_only(stage: Path, target: Path) -> None:
    """Atomically rename ``stage`` to absent ``target`` without replacement."""

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(os.fsencode(stage), os.fsencode(target), _RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = getattr(libc, "renameat2", None)
        if rename_exclusive is None:
            raise OSError(errno.ENOTSUP, "atomic create-only directory rename is unavailable")
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            _AT_FDCWD,
            os.fsencode(stage),
            _AT_FDCWD,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
    elif os.name == "nt":
        try:
            os.rename(stage, target)
        except FileExistsError as exc:
            raise FileExistsError(f"meta-study publication is create-only: {target}") from exc
        return
    else:
        raise OSError(errno.ENOTSUP, "atomic create-only directory rename is unavailable")

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"meta-study publication is create-only: {target}")
    raise OSError(error_number, os.strerror(error_number), target)


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
    """Validate, stage, verify, then atomically install one complete meta-study."""

    requested_target = Path(destination).expanduser()
    target = requested_target.parent.resolve() / requested_target.name
    if os.path.lexists(target):
        raise FileExistsError(f"meta-study publication is create-only: {target}")
    payload = decision_to_dict(decision)
    validate_decision_payload(payload)
    if objective_readiness != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("publication objective readiness differs from the study-owned gate")
    evidence_rows = tuple(primary_evidence)
    sensitivity_rows = tuple(sensitivity_evidence)
    sensitivity_summaries = tuple(sensitivity_evaluations)
    coverage_rows = tuple(sensitivity_coverages)
    evidence_payload = None
    acquisition_payload = None
    if decision_is_evidence_bearing(payload):
        if not evidence_rows:
            raise MetastudyContractError("evidence-bearing publication requires canonical profile evidence")
        evidence_payload = decision_evidence_payload(evidence_rows, decision=decision)
        if decision.selected_reduction is not None:
            acquisition_payload = acquisition_projection_payload(
                build_acquisition_projection(
                    evidence_rows,
                    selected_reduction=decision.selected_reduction,
                )
            )
    elif evidence_rows:
        raise MetastudyContractError("readiness-only publication does not accept profile evidence")
    sensitivity_payload = sensitivity_evidence_payload(
        sensitivity_rows,
        evaluations=sensitivity_summaries,
        coverages=coverage_rows,
        attempts=decision.materialization_attempts,
    )
    report = _render_report(payload)
    sensitivity_bytes = (json.dumps(sensitivity_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    manifest = {
        "schema_id": _PUBLICATION_SCHEMA_ID,
        "decision": payload,
        "objective_readiness": asdict(objective_readiness),
        "report_digest": "sha256:" + hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "sensitivity_file_digest": "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest(),
    }
    evidence_bytes = None
    acquisition_bytes = None
    if evidence_payload is not None:
        evidence_bytes = (json.dumps(evidence_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        manifest["evidence_file_digest"] = "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
    if acquisition_payload is not None:
        acquisition_bytes = (json.dumps(acquisition_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        manifest["acquisition_file_digest"] = "sha256:" + hashlib.sha256(acquisition_bytes).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "report.md").write_text(report, encoding="utf-8")
        (stage / "sensitivity.json").write_bytes(sensitivity_bytes)
        if evidence_bytes is not None:
            (stage / "evidence.json").write_bytes(evidence_bytes)
        if acquisition_bytes is not None:
            (stage / "acquisition.json").write_bytes(acquisition_bytes)
        verify_publication(stage)
        _rename_directory_create_only(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def verify_publication(path: Path) -> MetastudyDecision | None:
    """Verify exact files, decision contract, and report digest without mutation."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise MetastudyContractError("publication must be a directory")
    observed_files = {entry.name for entry in root.iterdir()}
    if observed_files not in (_EVIDENCE_FREE_FILES, _EVIDENCE_BEARING_FILES, _SELECTED_FILES):
        raise MetastudyContractError("publication files do not match a readiness, evaluated, or selected contract")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetastudyContractError("publication manifest is unreadable") from exc
    expected_manifest_fields = {
        "schema_id",
        "decision",
        "objective_readiness",
        "report_digest",
        "sensitivity_file_digest",
    }
    if observed_files in (_EVIDENCE_BEARING_FILES, _SELECTED_FILES):
        expected_manifest_fields.add("evidence_file_digest")
    if observed_files == _SELECTED_FILES:
        expected_manifest_fields.add("acquisition_file_digest")
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise MetastudyContractError("publication manifest fields do not match the exact contract")
    if manifest["schema_id"] != _PUBLICATION_SCHEMA_ID:
        raise MetastudyContractError("publication schema_id changed")
    decision = manifest["decision"]
    if not isinstance(decision, dict):
        raise MetastudyContractError("publication decision must be an object")
    validate_decision_payload(decision)
    objective_readiness = manifest["objective_readiness"]
    if objective_readiness_from_payload(objective_readiness) != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("publication objective readiness changed")
    has_evidence = observed_files in (_EVIDENCE_BEARING_FILES, _SELECTED_FILES)
    if decision_is_evidence_bearing(decision) != has_evidence:
        raise MetastudyContractError("decision evidence shape and publication bundle kind differ")
    if (decision["selected_reduction"] is not None) != (observed_files == _SELECTED_FILES):
        raise MetastudyContractError("selected decision and acquisition publication shape differ")
    report = (root / "report.md").read_bytes()
    observed = "sha256:" + hashlib.sha256(report).hexdigest()
    if manifest["report_digest"] != observed:
        raise MetastudyContractError("publication report digest mismatch")
    expected_report = _render_report(decision).encode("utf-8")
    if report != expected_report:
        raise MetastudyContractError("publication report bytes do not equal the canonical rendered decision")
    sensitivity_bytes = (root / "sensitivity.json").read_bytes()
    observed_sensitivity_digest = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    if manifest["sensitivity_file_digest"] != observed_sensitivity_digest:
        raise MetastudyContractError("publication sensitivity file digest mismatch")
    try:
        sensitivity_payload = json.loads(sensitivity_bytes)
    except json.JSONDecodeError as exc:
        raise MetastudyContractError("publication sensitivity evidence is unreadable") from exc
    attempts_payload = decision["materialization_attempts"]
    if not isinstance(attempts_payload, list):
        raise MetastudyContractError("publication decision materialization_attempts must be an array")
    parsed_attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts_payload)
    )
    verify_sensitivity_evidence_payload(sensitivity_payload, attempts=parsed_attempts)
    evidence_payload: object | None = None
    if decision_is_evidence_bearing(decision):
        evidence_bytes = (root / "evidence.json").read_bytes()
        observed_evidence_file_digest = "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
        if manifest["evidence_file_digest"] != observed_evidence_file_digest:
            raise MetastudyContractError("publication evidence file digest mismatch")
        try:
            evidence_payload = json.loads(evidence_bytes)
        except json.JSONDecodeError as exc:
            raise MetastudyContractError("publication evidence is unreadable") from exc
        verify_decision_evidence_payload(evidence_payload, decision)
    if decision["selected_reduction"] is not None:
        acquisition_bytes = (root / "acquisition.json").read_bytes()
        if manifest["acquisition_file_digest"] != "sha256:" + hashlib.sha256(acquisition_bytes).hexdigest():
            raise MetastudyContractError("publication acquisition file digest mismatch")
        try:
            acquisition_payload = json.loads(acquisition_bytes)
        except json.JSONDecodeError as exc:
            raise MetastudyContractError("publication acquisition projection is unreadable") from exc
        declared_projection = validate_acquisition_projection_payload(acquisition_payload)
        assert isinstance(evidence_payload, dict)
        profiles = evidence_payload["profiles"]
        assert isinstance(profiles, list)
        parsed_profiles = tuple(
            parse_profile_evidence_projection(row, index=index) for index, row in enumerate(profiles)
        )
        reduction = decision["selected_reduction"]
        assert isinstance(reduction, list)
        expected_projection = build_acquisition_projection(
            parsed_profiles,
            selected_reduction=tuple(reduction),
        )
        if acquisition_projection_payload(declared_projection) != acquisition_projection_payload(expected_projection):
            raise MetastudyContractError("publication acquisition projection differs from bundled profiles")
    return None


def verify_decision_evidence_payload(evidence: object, decision: dict[str, object]) -> None:
    """Prove bundled evidence-to-decision consistency without claiming live-source authenticity."""

    if not isinstance(evidence, dict) or set(evidence) != {
        "readiness_receipt_digest",
        "materialization_attempts",
        "profiles",
    }:
        raise MetastudyContractError("publication evidence fields do not match the exact contract")
    if evidence["readiness_receipt_digest"] != decision["readiness"]["receipt_digest"]:
        raise MetastudyContractError("publication evidence readiness receipt mismatch")
    evidence_attempts = json.loads(json.dumps(evidence["materialization_attempts"], allow_nan=False))
    decision_attempts = json.loads(json.dumps(decision["materialization_attempts"], allow_nan=False))
    if evidence_attempts != decision_attempts:
        raise MetastudyContractError("publication evidence attempt ledger mismatch")
    profiles = evidence["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise MetastudyContractError("evidence-bearing publication requires profiles")
    if canonical_digest(evidence) != decision["evidence_digest"]:
        raise MetastudyContractError("publication evidence does not match decision evidence_digest")
    try:
        parsed_profiles = tuple(
            parse_profile_evidence_projection(row, index=index) for index, row in enumerate(profiles)
        )
        attempts_payload = evidence["materialization_attempts"]
        if not isinstance(attempts_payload, list):
            raise MetastudyContractError("publication evidence materialization_attempts must be an array")
        parsed_attempts = tuple(
            materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts_payload)
        )
        readiness_payload = decision["readiness"]
        if not isinstance(readiness_payload, dict):
            raise MetastudyContractError("publication decision readiness must be an object")
        readiness = EvidenceReadiness(
            selected_experiment_count=readiness_payload["selected_experiment_count"],
            ready_experiment_count=readiness_payload["ready_experiment_count"],
            ready_experiment_ids=tuple(readiness_payload["ready_experiment_ids"]),
            blocked_experiment_ids=tuple(readiness_payload["blocked_experiment_ids"]),
            receipt_digest=readiness_payload["receipt_digest"],
        )
        recomputed = reevaluate_evidence_projection(
            parsed_profiles,
            readiness=readiness,
            attempts=parsed_attempts,
            evidence_digest=canonical_digest(evidence),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MetastudyContractError):
            raise
        raise MetastudyContractError(f"publication evidence projection is invalid: {exc}") from exc
    canonical_decision = json.loads(json.dumps(decision_to_dict(recomputed), allow_nan=False))
    if canonical_decision != decision:
        raise MetastudyContractError("publication decision differs from canonical evidence evaluation")


def _render_report(payload: dict[str, object]) -> str:
    reduction = payload["selected_reduction"]
    reduction_text = "none" if reduction is None else f"{reduction[0]:g}-{reduction[1]:g} h"
    blockers = payload["blockers"]
    blocker_lines = "\n".join(f"- {value}" for value in blockers) if blockers else "- none"
    limitations = payload["limitations"]
    limitation_lines = "\n".join(f"- {value}" for value in limitations) if limitations else "- none"
    return (
        "# RT-lnRNA reporter-response reduction recommendation\n\n"
        f"- Protocol: `{payload['protocol_id']}`\n"
        f"- Status: `{payload['status']}`\n"
        f"- Evidence grade: `{payload['evidence_grade']}`\n"
        f"- Selected reduction: `{reduction_text}`\n"
        f"- Policy digest: `{payload['policy_digest']}`\n"
        f"- Evidence digest: `{payload['evidence_digest']}`\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n\n"
        "## Limitations\n\n"
        f"{limitation_lines}\n"
    )


__all__ = ["publish_metastudy", "verify_decision_evidence_payload", "verify_publication"]
