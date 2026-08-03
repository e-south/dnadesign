"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/publication/verification.py

Offline verification for reporter-response meta-study publications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from ..acquisition_projection import (
    acquisition_projection_payload,
    build_acquisition_projection,
    validate_acquisition_projection_payload,
)
from ..contracts._values import MetastudyContractError, canonical_digest
from ..contracts.decision import decision_is_evidence_bearing
from ..contracts.decision_codec import decision_to_dict, validate_decision_payload
from ..contracts.materialization import EvidenceReadiness, materialization_attempt_from_payload
from ..contracts.objective import DEFAULT_OBJECTIVE_READINESS, objective_readiness_from_payload
from ..evaluation.selection import reevaluate_evidence_projection
from ..evidence_projection.parsing import parse_profile_evidence_projection
from ..sensitivity import verify_sensitivity_evidence_payload
from .report import _render_report

_EVIDENCE_FREE_FILES = frozenset({"manifest.json", "report.md", "sensitivity.json"})
_EVIDENCE_BEARING_FILES = frozenset({*_EVIDENCE_FREE_FILES, "evidence.json"})
_SELECTED_FILES = frozenset({*_EVIDENCE_BEARING_FILES, "acquisition.json"})
_PUBLICATION_SCHEMA_ID = "rt_lnrna_reporter_response_metastudy_publication.v7"


def _read_publication_bundle(root: Path) -> dict[str, bytes]:
    """Snapshot an exact bundle without following member symlinks."""

    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise MetastudyContractError("publication directory is unreadable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise MetastudyContractError("publication must be a directory")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise MetastudyContractError("no-follow publication verification is unavailable")
    directory_flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise MetastudyContractError("publication directory is unreadable") from exc
    try:
        try:
            observed_files = set(os.listdir(root_fd))
        except OSError as exc:
            raise MetastudyContractError("publication directory is unreadable") from exc
        if observed_files not in (_EVIDENCE_FREE_FILES, _EVIDENCE_BEARING_FILES, _SELECTED_FILES):
            raise MetastudyContractError("publication files do not match a readiness, evaluated, or selected contract")
        payloads: dict[str, bytes] = {}
        file_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        for name in sorted(observed_files):
            try:
                metadata = os.lstat(name, dir_fd=root_fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise MetastudyContractError("publication bundle members must be regular files")
                member_fd = os.open(name, file_flags, dir_fd=root_fd)
            except MetastudyContractError:
                raise
            except OSError as exc:
                raise MetastudyContractError("publication bundle members must be regular files") from exc
            try:
                if not stat.S_ISREG(os.fstat(member_fd).st_mode):
                    raise MetastudyContractError("publication bundle members must be regular files")
                with os.fdopen(member_fd, "rb", closefd=False) as stream:
                    payloads[name] = stream.read()
            except OSError as exc:
                raise MetastudyContractError(f"publication bundle member is unreadable: {name}") from exc
            finally:
                os.close(member_fd)
        return payloads
    finally:
        os.close(root_fd)


def _verify_publication_bundle(bundle: dict[str, bytes]) -> None:
    """Verify a complete immutable bundle snapshot without filesystem access."""

    observed_files = set(bundle)
    if observed_files not in (_EVIDENCE_FREE_FILES, _EVIDENCE_BEARING_FILES, _SELECTED_FILES):
        raise MetastudyContractError("publication files do not match a readiness, evaluated, or selected contract")
    try:
        manifest = json.loads(bundle["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    if objective_readiness_from_payload(manifest["objective_readiness"]) != DEFAULT_OBJECTIVE_READINESS:
        raise MetastudyContractError("publication objective readiness changed")
    has_evidence = observed_files in (_EVIDENCE_BEARING_FILES, _SELECTED_FILES)
    if decision_is_evidence_bearing(decision) != has_evidence:
        raise MetastudyContractError("decision evidence shape and publication bundle kind differ")
    if (decision["selected_reduction"] is not None) != (observed_files == _SELECTED_FILES):
        raise MetastudyContractError("selected decision and acquisition publication shape differ")
    report = bundle["report.md"]
    if manifest["report_digest"] != _digest(report):
        raise MetastudyContractError("publication report digest mismatch")
    if report != _render_report(decision).encode("utf-8"):
        raise MetastudyContractError("publication report bytes do not equal the canonical rendered decision")
    sensitivity_bytes = bundle["sensitivity.json"]
    if manifest["sensitivity_file_digest"] != _digest(sensitivity_bytes):
        raise MetastudyContractError("publication sensitivity file digest mismatch")
    try:
        sensitivity_payload = json.loads(sensitivity_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetastudyContractError("publication sensitivity evidence is unreadable") from exc
    attempts_payload = decision["materialization_attempts"]
    if not isinstance(attempts_payload, list):
        raise MetastudyContractError("publication decision materialization_attempts must be an array")
    parsed_attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts_payload)
    )
    verify_sensitivity_evidence_payload(sensitivity_payload, attempts=parsed_attempts)
    evidence_payload: object | None = None
    if has_evidence:
        evidence_bytes = bundle["evidence.json"]
        if manifest["evidence_file_digest"] != _digest(evidence_bytes):
            raise MetastudyContractError("publication evidence file digest mismatch")
        try:
            evidence_payload = json.loads(evidence_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetastudyContractError("publication evidence is unreadable") from exc
        _verify_decision_evidence_payload(evidence_payload, decision)
    if observed_files == _SELECTED_FILES:
        _verify_acquisition_projection(bundle["acquisition.json"], manifest, decision, evidence_payload)


def verify_publication(path: Path) -> None:
    """Verify exact files and canonical scientific projections without mutation."""

    root = Path(path).expanduser().absolute()
    _verify_publication_bundle(_read_publication_bundle(root))


def _verify_acquisition_projection(
    acquisition_bytes: bytes,
    manifest: dict[str, object],
    decision: dict[str, object],
    evidence_payload: object | None,
) -> None:
    if manifest["acquisition_file_digest"] != _digest(acquisition_bytes):
        raise MetastudyContractError("publication acquisition file digest mismatch")
    try:
        acquisition_payload = json.loads(acquisition_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetastudyContractError("publication acquisition projection is unreadable") from exc
    declared_projection = validate_acquisition_projection_payload(acquisition_payload)
    assert isinstance(evidence_payload, dict)
    profiles = evidence_payload["profiles"]
    assert isinstance(profiles, list)
    parsed_profiles = tuple(parse_profile_evidence_projection(row, index=index) for index, row in enumerate(profiles))
    reduction = decision["selected_reduction"]
    assert isinstance(reduction, list)
    expected_projection = build_acquisition_projection(parsed_profiles, selected_reduction=tuple(reduction))
    if acquisition_projection_payload(declared_projection) != acquisition_projection_payload(expected_projection):
        raise MetastudyContractError("publication acquisition projection differs from bundled profiles")


def _verify_decision_evidence_payload(evidence: object, decision: dict[str, object]) -> None:
    if not isinstance(evidence, dict) or set(evidence) != {
        "readiness_receipt_digest",
        "materialization_attempts",
        "profiles",
    }:
        raise MetastudyContractError("publication evidence fields do not match the exact contract")
    readiness = decision["readiness"]
    if not isinstance(readiness, dict):
        raise MetastudyContractError("publication decision readiness must be an object")
    if evidence["readiness_receipt_digest"] != readiness["receipt_digest"]:
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
        readiness_contract = EvidenceReadiness(
            selected_experiment_count=readiness["selected_experiment_count"],
            ready_experiment_count=readiness["ready_experiment_count"],
            ready_experiment_ids=tuple(readiness["ready_experiment_ids"]),
            blocked_experiment_ids=tuple(readiness["blocked_experiment_ids"]),
            receipt_digest=readiness["receipt_digest"],
        )
        recomputed = reevaluate_evidence_projection(
            parsed_profiles,
            readiness=readiness_contract,
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


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = ["verify_publication"]
