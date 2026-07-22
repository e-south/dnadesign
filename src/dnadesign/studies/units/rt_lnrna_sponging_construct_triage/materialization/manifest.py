"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/materialization/manifest.py

Projection-manifest loading and source authority checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..construct_projection import validate_projection_manifest_payload
from ..genbank_authority import GenBankAuthorityAudit, run_default_authority_audit
from ..source_promotions import ConstructWindowPolicy
from .common import _list, _mapping, _span_0
from .contracts import (
    _PROJECTION_MANIFEST_PATH,
    _REQUIRED_SLOT_IDS,
    MaterializationContractError,
)


def _load_projection_manifest(repo_root: Path) -> dict[str, object]:
    payload = yaml.safe_load((repo_root / _PROJECTION_MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MaterializationContractError("Construct projection manifest must be a mapping.")
    return payload


def _require_valid_projection_manifest(manifest: dict[str, object]) -> None:
    audit = validate_projection_manifest_payload(manifest)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"Construct projection manifest is invalid: {joined}")


def _require_genbank_authority(repo_root: Path) -> GenBankAuthorityAudit:
    audit = run_default_authority_audit(repo_root=repo_root)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"GenBank source authority is invalid: {joined}")
    return audit


def _template_sequence(*, manifest: dict[str, object], authority: GenBankAuthorityAudit) -> str:
    template = _mapping(manifest["construct_template"], label="construct_template")
    source_id = str(template["source_authority_id"])
    return authority.source(source_id).sequence


def _target_context_bounds(manifest: dict[str, object]) -> tuple[int, int]:
    template = _mapping(manifest["construct_template"], label="construct_template")
    target = _mapping(template["target_context"], label="construct_template.target_context")
    start = int(target["window_start_0"])
    end = int(target["window_end_0"])
    if end <= start:
        raise MaterializationContractError("target_context.window_end_0 must be greater than window_start_0.")
    expected_length = int(target.get("length_nt", end - start))
    if end - start != expected_length:
        raise MaterializationContractError("target_context window span must equal target_context.length_nt.")
    return start, end


def _source_promotion_window_policy(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    target_start: int,
    target_end: int,
) -> ConstructWindowPolicy:
    template = _mapping(manifest["construct_template"], label="construct_template")
    target = _mapping(template["target_context"], label="construct_template.target_context")
    slots = {str(slot["slot_id"]): slot for slot in _list(manifest["slots"], label="slots")}
    missing = sorted(set(_REQUIRED_SLOT_IDS) - set(slots))
    if missing:
        joined = ", ".join(missing)
        raise MaterializationContractError(f"Source promotion window policy missing required slot(s): {joined}")
    return ConstructWindowPolicy(
        context_id=str(target["context_id"]),
        target_start_0=target_start,
        target_length_nt=target_end - target_start,
        template_length_nt=len(template_sequence),
        lnrna_template_span_0=_span_0(slots["lnrna"]["template_span_0"], label="lnrna.template_span_0"),
        rt_cds_template_span_0=_span_0(slots["rt_cds"]["template_span_0"], label="rt_cds.template_span_0"),
    )
