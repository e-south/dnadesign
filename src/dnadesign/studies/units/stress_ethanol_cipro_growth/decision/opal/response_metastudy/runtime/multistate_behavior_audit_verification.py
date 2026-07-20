"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_audit_verification.py

Fail-closed provenance checks for the behavior shadow's independent audit.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime

from .multistate_behavior_record_fields import prefixed_digest, require_fields

_AUDIT_FIELDS = {
    "auditor_id",
    "completed_at",
    "schema_id",
    "schema_version",
    "status",
    "method",
    "scope",
    "findings",
    "blockers",
    "reviewed_source_commit",
    "reviewed_preliminary_manifest_sha256",
}

# This receipt identifies the current pre-release objective snapshot. Exact
# source bytes are bound separately by the activation receipt.
_AUDIT_PROVENANCE = {
    "auditor_id": "codex_subagent.metric_adversarial_audit.v1",
    "completed_at": "2026-07-19T22:32:54Z",
    "reviewed_source_commit": "5fde329e11fe551cab493b47b99ab54f7ccc2825",  # pragma: allowlist secret
    "reviewed_preliminary_manifest_sha256": ("sha256:987a3e8e155af7fb72e31af077784fcca4d4acbfa1f291fdf8ed66cf049a8293"),
}


def verify_behavior_adversarial_audit_record(audit: dict[str, object]) -> None:
    """Verify the exact independent reviewer and source snapshot for audit v1."""

    require_fields(audit, _AUDIT_FIELDS, context="independent audit")
    if audit["schema_id"] != "stress_ethanol_cipro_growth.multistate_response_behavior_adversarial_audit.v1":
        raise ValueError("independent audit schema identity drifted.")
    if audit["schema_version"] != "1":
        raise ValueError("independent audit schema version drifted.")
    for field, expected in _AUDIT_PROVENANCE.items():
        if audit[field] != expected:
            raise ValueError(f"independent audit {field} disagrees with the reviewed snapshot.")
    auditor_id = audit["auditor_id"]
    if not isinstance(auditor_id, str) or not auditor_id.startswith("codex_subagent."):
        raise ValueError("independent audit auditor_id must identify the stable Codex subagent role.")
    completed_at = audit["completed_at"]
    if not isinstance(completed_at, str):
        raise ValueError("independent audit completed_at must be a UTC timestamp.")
    try:
        datetime.strptime(completed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("independent audit completed_at must be an exact UTC timestamp.") from exc
    commit = audit["reviewed_source_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("independent audit reviewed_source_commit must be one full lowercase Git SHA.")
    prefixed_digest(
        audit["reviewed_preliminary_manifest_sha256"],
        field="independent audit reviewed_preliminary_manifest_sha256",
    )
    if audit["status"] not in {"pass", "fail"}:
        raise ValueError("independent audit status must be pass or fail.")
    for field in ("scope", "findings", "blockers"):
        if not isinstance(audit[field], list) or any(
            not isinstance(value, str) or not value.strip() for value in audit[field]
        ):
            raise ValueError(f"independent audit {field} must be a list of nonempty findings.")


__all__ = ["verify_behavior_adversarial_audit_record"]
