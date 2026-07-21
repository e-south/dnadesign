"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/synthesis_handoff_surface.py

Study-owned status inspection for the active OPAL synthesis handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dnadesign.ops.status import resolve_path_ref
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    get_synthesis_handoff_record,
    handoff_record_payload,
)


def inspect_synthesis_handoff_surface(
    *,
    study_context: object,
    opal_config: Mapping[str, object],
) -> dict[str, object]:
    """Cross-check the active OPAL handoff assertion against study authority."""

    raw_assertion = opal_config.get("synthesis_handoff")
    if raw_assertion is None:
        return {
            "configured": False,
            "state": "not_configured",
            "drives_top_level_attention": False,
            "summary": "OPAL synthesis handoff is not configured",
            "mismatches": [],
        }
    if not isinstance(raw_assertion, Mapping):
        return _attention(
            mismatches=[
                {
                    "field": "pipeline.opal.synthesis_handoff",
                    "expected": "mapping",
                    "actual": type(raw_assertion).__name__,
                }
            ]
        )

    mismatches: list[dict[str, object]] = []
    handoff_id = _required_text(
        raw_assertion.get("handoff_id"),
        "pipeline.opal.synthesis_handoff.handoff_id",
        mismatches,
    )
    required_lifecycle_status = _required_text(
        raw_assertion.get("required_lifecycle_status"),
        "pipeline.opal.synthesis_handoff.required_lifecycle_status",
        mismatches,
    )
    round0 = _mapping(opal_config.get("round0"))
    expected_campaign_slug = _required_text(
        round0.get("campaign_slug"),
        "pipeline.opal.round0.campaign_slug",
        mismatches,
    )
    expected_run_id = _required_text(
        round0.get("run_id"),
        "pipeline.opal.round0.run_id",
        mismatches,
    )
    expected_model_round = _required_int(
        round0.get("round_index"),
        "pipeline.opal.round0.round_index",
        mismatches,
    )

    repo_root = getattr(study_context, "study_repo_root", None)
    study_root = getattr(study_context, "resolved_study_dir", None)
    if not isinstance(repo_root, Path):
        _mismatch(mismatches, "study_context.study_repo_root", "resolved repository path", repo_root)
    if not isinstance(study_root, Path):
        _mismatch(mismatches, "study_context.resolved_study_dir", "resolved study path", study_root)

    record_ref = _record_ref(study_context=study_context, mismatches=mismatches)
    record_path = _resolve_record_path(
        record_ref=record_ref,
        repo_root=repo_root if isinstance(repo_root, Path) else None,
        study_root=study_root if isinstance(study_root, Path) else None,
        mismatches=mismatches,
    )
    record_payload: dict[str, object] | None = None
    if record_path is not None and handoff_id is not None:
        try:
            record = get_synthesis_handoff_record(record_path, handoff_id)
            record_payload = handoff_record_payload(record, repo_root=repo_root)
        except (OSError, UnicodeError, ValueError) as exc:
            _mismatch(
                mismatches,
                "synthesis_handoff.record",
                f"valid lifecycle record for {handoff_id}",
                f"{type(exc).__name__}: {exc}",
            )
        else:
            _compare(
                mismatches,
                "synthesis_handoff.lifecycle_status",
                required_lifecycle_status,
                record.lifecycle_status,
            )
            _compare(
                mismatches,
                "synthesis_handoff.source_authority",
                "opal_selection_batch",
                record.source_authority,
            )
            _compare(
                mismatches,
                "synthesis_handoff.selection_epoch",
                "opal_model_round",
                record.selection_epoch,
            )
            _compare(
                mismatches,
                "synthesis_handoff.campaign_slug",
                expected_campaign_slug,
                record.campaign_slug,
            )
            _compare(mismatches, "synthesis_handoff.run_id", expected_run_id, record.run_id)
            _compare(
                mismatches,
                "synthesis_handoff.model_as_of_round",
                expected_model_round,
                record.model_as_of_round,
            )
            if record.lifecycle_status == "accepted_for_order":
                artifact_status = _mapping(record_payload.get("artifact_status"))
                artifact_summary = _mapping(artifact_status.get("summary"))
                _compare(
                    mismatches,
                    "synthesis_handoff.artifact_status.current_contract_ready",
                    True,
                    artifact_summary.get("current_contract_ready"),
                )

    if mismatches:
        return _attention(
            mismatches=mismatches,
            handoff_id=handoff_id,
            required_lifecycle_status=required_lifecycle_status,
            record_ref=record_ref,
            record_path=record_path,
            record_payload=record_payload,
        )

    assert handoff_id is not None
    assert required_lifecycle_status is not None
    assert record_payload is not None
    vendor_submission = "not_performed" if required_lifecycle_status == "accepted_for_order" else "not_declared"
    return {
        "configured": True,
        "state": "ok",
        "drives_top_level_attention": False,
        "summary": f"OPAL synthesis handoff verified {handoff_id} ({required_lifecycle_status})",
        "handoff_id": handoff_id,
        "required_lifecycle_status": required_lifecycle_status,
        "record_ref": record_ref,
        "resolved_record_ref": str(record_path),
        "vendor_submission": vendor_submission,
        "record": record_payload,
        "mismatches": [],
    }


def _record_ref(*, study_context: object, mismatches: list[dict[str, object]]) -> str | None:
    ops_contract = getattr(study_context, "ops_contract", None)
    record_sources = _mapping(getattr(ops_contract, "record_sources", None))
    return _required_text(
        record_sources.get("synthesis_handoffs_ref"),
        "ops.study.yaml.record_sources.synthesis_handoffs_ref",
        mismatches,
    )


def _resolve_record_path(
    *,
    record_ref: str | None,
    repo_root: Path | None,
    study_root: Path | None,
    mismatches: list[dict[str, object]],
) -> Path | None:
    if record_ref is None or repo_root is None or study_root is None:
        return None
    try:
        return resolve_path_ref(
            record_ref,
            repo_root=repo_root,
            manifest_dir=study_root,
            default_base="manifest",
            label="ops.study.yaml record_sources.synthesis_handoffs_ref",
        )
    except (OSError, ValueError) as exc:
        _mismatch(
            mismatches,
            "ops.study.yaml.record_sources.synthesis_handoffs_ref",
            "valid study record path",
            f"{type(exc).__name__}: {exc}",
        )
        return None


def _attention(
    *,
    mismatches: list[dict[str, object]],
    handoff_id: str | None = None,
    required_lifecycle_status: str | None = None,
    record_ref: str | None = None,
    record_path: Path | None = None,
    record_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "configured": True,
        "state": "attention",
        "drives_top_level_attention": True,
        "summary": f"OPAL synthesis handoff has {len(mismatches)} integrity mismatch(es)",
        "handoff_id": handoff_id,
        "required_lifecycle_status": required_lifecycle_status,
        "record_ref": record_ref,
        "resolved_record_ref": str(record_path) if record_path is not None else None,
        "vendor_submission": "unknown",
        "mismatches": mismatches,
    }
    if record_payload is not None:
        payload["record"] = record_payload
    return payload


def _required_text(value: object, field: str, mismatches: list[dict[str, object]]) -> str | None:
    resolved = _text_or_none(value)
    if resolved is None:
        _mismatch(mismatches, field, "non-empty string", value)
    return resolved


def _required_int(value: object, field: str, mismatches: list[dict[str, object]]) -> int | None:
    resolved = value if isinstance(value, int) and not isinstance(value, bool) else None
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


__all__ = ["inspect_synthesis_handoff_surface"]
