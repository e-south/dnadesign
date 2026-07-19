"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/opal_run_receipt.py

Study-owned verification of the declared OPAL round-0 run receipt.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from dnadesign.ops.status import resolve_repo_relative_path

_STATUS_KIND = "stress-ethanol-cipro-growth-status"
_ARTIFACT_SPECS = (
    {
        "label": "round_context",
        "contract_id": "opal_round0_run_context",
        "role": "opal_round_context",
        "ref_key": "round_context",
        "sha_key": "round_context_sha256",
    },
    {
        "label": "selection_batch",
        "contract_id": "opal_round0_selection_batch",
        "role": "opal_selection_batch",
        "ref_key": "selection_batch",
        "sha_key": "selection_batch_sha256",
    },
)


def inspect_opal_round0_run_receipt(
    *,
    study_context: object,
    opal_config: Mapping[str, object],
) -> dict[str, object]:
    """Verify that the two study records and materialized round agree exactly."""
    pipeline = _mapping(opal_config.get("round0"))
    contracts = _mapping(getattr(getattr(study_context, "ops_contract", None), "artifacts", {}))
    configured = bool(pipeline or any(_mapping(contracts.get(spec["contract_id"])) for spec in _ARTIFACT_SPECS))
    if not configured:
        return {
            "configured": False,
            "state": "not_configured",
            "drives_top_level_attention": False,
            "summary": "OPAL round-0 run receipt is not configured",
            "mismatches": [],
        }

    mismatches: list[dict[str, object]] = []
    expected = {
        "run_id": _required_text(pipeline.get("run_id"), "pipeline.round0.run_id", mismatches),
        "campaign_slug": _required_text(pipeline.get("campaign_slug"), "pipeline.round0.campaign_slug", mismatches),
        "round_index": _required_int(pipeline.get("round_index"), "pipeline.round0.round_index", mismatches),
    }
    repo_root = getattr(study_context, "study_repo_root", None)
    if not isinstance(repo_root, Path):
        _mismatch(mismatches, "study_context.study_repo_root", "resolved repository path", repo_root)
        repo_root = None

    artifact_payloads = {
        spec["label"]: _inspect_artifact(
            spec=spec,
            pipeline=pipeline,
            contract=_mapping(contracts.get(spec["contract_id"])),
            expected=expected,
            repo_root=repo_root,
            mismatches=mismatches,
        )
        for spec in _ARTIFACT_SPECS
    }
    _verify_round_context(
        path=_path_or_none(artifact_payloads["round_context"]["resolved_ref"]),
        expected=expected,
        mismatches=mismatches,
    )

    state = "attention" if mismatches else "ok"
    summary = (
        f"OPAL round-0 run receipt has {len(mismatches)} integrity mismatch(es)"
        if mismatches
        else f"OPAL round-0 run receipt verified {expected['run_id']}"
    )
    return {
        "configured": True,
        "state": state,
        "drives_top_level_attention": bool(mismatches),
        "summary": summary,
        **expected,
        "artifacts": artifact_payloads,
        "mismatches": mismatches,
    }


def _inspect_artifact(
    *,
    spec: Mapping[str, str],
    pipeline: Mapping[str, object],
    contract: Mapping[str, object],
    expected: Mapping[str, object],
    repo_root: Path | None,
    mismatches: list[dict[str, object]],
) -> dict[str, object]:
    label = spec["label"]
    if not contract:
        _mismatch(mismatches, f"contract.{label}", "declared artifact", None)
    for field in ("run_id", "campaign_slug", "round_index"):
        value = _int_or_none(contract.get(field)) if field == "round_index" else _text_or_none(contract.get(field))
        _compare(mismatches, f"contract.{label}.{field}", expected[field], value)
    _compare(mismatches, f"contract.{label}.role", spec["role"], _text_or_none(contract.get("role")))

    pipeline_ref = _required_text(pipeline.get(spec["ref_key"]), f"pipeline.round0.{spec['ref_key']}", mismatches)
    contract_ref = _required_text(contract.get("ref"), f"contract.{label}.ref", mismatches)
    pipeline_path = _resolve_ref(repo_root, pipeline_ref, f"pipeline.round0.{spec['ref_key']}", mismatches)
    contract_path = _resolve_ref(repo_root, contract_ref, f"contract.{label}.ref", mismatches)
    _compare(
        mismatches,
        f"contract.{label}.resolved_ref",
        str(pipeline_path) if pipeline_path is not None else None,
        str(contract_path) if contract_path is not None else None,
    )

    pipeline_sha = _required_text(pipeline.get(spec["sha_key"]), f"pipeline.round0.{spec['sha_key']}", mismatches)
    contract_sha = _required_text(contract.get("sha256"), f"contract.{label}.sha256", mismatches)
    _compare(mismatches, f"contract.{label}.sha256", pipeline_sha, contract_sha)
    path = contract_path or pipeline_path
    declared_sha = contract_sha or pipeline_sha
    actual_sha = _sha256(path, f"artifacts.{label}", mismatches)
    _compare(mismatches, f"artifacts.{label}.actual_sha256", declared_sha, actual_sha)
    return {
        "ref": contract_ref or pipeline_ref,
        "resolved_ref": str(path) if path is not None else None,
        "declared_sha256": declared_sha,
        "actual_sha256": actual_sha,
        "verified": declared_sha is not None and actual_sha == declared_sha,
    }


def _verify_round_context(
    *,
    path: Path | None,
    expected: Mapping[str, object],
    mismatches: list[dict[str, object]],
) -> None:
    if path is None or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _mismatch(mismatches, "round_context.json", "valid JSON object", f"{type(exc).__name__}: {exc}")
        return
    if not isinstance(payload, Mapping):
        _mismatch(mismatches, "round_context.json", "JSON object", type(payload).__name__)
        return
    _compare(
        mismatches,
        "round_context.core/run_id",
        expected["run_id"],
        _text_or_none(payload.get("core/run_id")),
    )
    _compare(
        mismatches,
        "round_context.core/campaign_slug",
        expected["campaign_slug"],
        _text_or_none(payload.get("core/campaign_slug")),
    )
    _compare(
        mismatches,
        "round_context.core/round_index",
        expected["round_index"],
        _int_or_none(payload.get("core/round_index")),
    )


def _sha256(path: Path | None, field: str, mismatches: list[dict[str, object]]) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        _mismatch(mismatches, field, "materialized file", str(path))
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _mismatch(mismatches, field, "readable file", f"{type(exc).__name__}: {exc}")
        return None
    return digest.hexdigest()


def _resolve_ref(
    repo_root: Path | None,
    raw_ref: str | None,
    field: str,
    mismatches: list[dict[str, object]],
) -> Path | None:
    if repo_root is None or raw_ref is None:
        return None
    try:
        return resolve_repo_relative_path(repo_root=repo_root, raw_path=raw_ref, status_kind=_STATUS_KIND)
    except (OSError, ValueError) as exc:
        _mismatch(mismatches, field, "valid repository path reference", f"{type(exc).__name__}: {exc}")
        return None


def _required_text(value: object, field: str, mismatches: list[dict[str, object]]) -> str | None:
    resolved = _text_or_none(value)
    if resolved is None:
        _mismatch(mismatches, field, "non-empty string", value)
    return resolved


def _required_int(value: object, field: str, mismatches: list[dict[str, object]]) -> int | None:
    resolved = _int_or_none(value)
    if resolved is None:
        _mismatch(mismatches, field, "integer", value)
    return resolved


def _compare(mismatches: list[dict[str, object]], field: str, expected: object, actual: object) -> None:
    if expected != actual:
        _mismatch(mismatches, field, expected, actual)


def _mismatch(mismatches: list[dict[str, object]], field: str, expected: object, actual: object) -> None:
    mismatches.append({"field": field, "expected": expected, "actual": actual})


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _path_or_none(value: object) -> Path | None:
    text = _text_or_none(value)
    return Path(text) if text is not None else None


__all__ = ["inspect_opal_round0_run_receipt"]
