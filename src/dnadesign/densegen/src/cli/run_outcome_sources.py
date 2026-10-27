"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/run_outcome_sources.py

Resolve DenseGen run outcome/pressure summaries from finalized manifests or the.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..core.pipeline.attempts import _load_attempts_snapshot
from ..core.run_paths import run_manifest_path, run_state_path
from ..core.run_state import load_run_state
from .run_intro import RunContractSummary, RunOutcomeSummary
from .run_outcome_extract import extract_outcome


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _quota_by_plan(contract: RunContractSummary) -> dict[str, int]:
    return {str(plan.name): int(plan.quota) for plan in contract.plans}


def _ordered_plan_names(
    plan_order: Sequence[str],
    *,
    generated_by_plan: Mapping[str, int],
    pressure_by_plan: Mapping[str, Mapping[str, int]],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in plan_order:
        token = str(name).strip()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    for mapping in (generated_by_plan, pressure_by_plan):
        for name in mapping:
            token = str(name).strip()
            if not token or token in seen:
                continue
            ordered.append(token)
            seen.add(token)
    return ordered


def _load_manifest_payload(path: Path) -> tuple[Mapping[str, object] | None, str | None]:
    if not path.exists():
        return (
            None,
            "finalized run manifest is not available yet; run `dense run` to materialize "
            "`outputs/meta/run_manifest.json`.",
        )
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None, "run outcomes are unavailable because `run_manifest.json` could not be parsed."
    if not isinstance(payload, Mapping):
        return None, "run outcomes are unavailable because `run_manifest.json` is not a JSON object."
    return payload, None


def _load_pressure_by_plan(run_root: Path) -> tuple[dict[str, dict[str, int]], str | None]:
    pressure_by_plan: dict[str, dict[str, int]] = {}
    source_tokens: list[str] = []

    tables_root = run_root / "outputs" / "tables"
    attempts_df = _load_attempts_snapshot(
        tables_root,
        columns=["plan_name", "status"],
    )
    if not attempts_df.empty:
        source_tokens.append("attempts")
        statuses = attempts_df["status"].astype(str).str.strip().str.lower()
        failures = attempts_df[statuses.isin({"rejected", "failed"})].copy()
        if not failures.empty:
            grouped = failures.groupby("plan_name", dropna=False).size()
            for plan_name, count in grouped.items():
                token = str(plan_name or "").strip()
                if not token:
                    continue
                state = pressure_by_plan.setdefault(
                    token,
                    {"stall_events": 0, "total_resamples": 0, "failed_solutions": 0},
                )
                state["failed_solutions"] += int(count)

    events_path = run_root / "outputs" / "meta" / "events.jsonl"
    if events_path.exists():
        source_tokens.append("events")
        for raw_line in events_path.read_text().splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except Exception:
                continue
            event_name = str(event.get("event") or "").strip().upper()
            plan_name = str(event.get("plan_name") or "").strip()
            if not plan_name:
                continue
            state = pressure_by_plan.setdefault(
                plan_name,
                {"stall_events": 0, "total_resamples": 0, "failed_solutions": 0},
            )
            if event_name == "STALL_DETECTED":
                state["stall_events"] += 1
            elif event_name == "RESAMPLE_TRIGGERED":
                state["total_resamples"] += 1

    if not source_tokens:
        return {}, None
    unique_tokens: list[str] = []
    for token in source_tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    return pressure_by_plan, "/".join(unique_tokens)


def _build_outcome_payload(
    *,
    plan_order: Sequence[str],
    plan_quotas: Mapping[str, int],
    generated_by_plan: Mapping[str, int],
    pressure_by_plan: Mapping[str, Mapping[str, int]],
    created_at: str | None,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    ordered_names = _ordered_plan_names(
        plan_order,
        generated_by_plan=generated_by_plan,
        pressure_by_plan=pressure_by_plan,
    )
    for plan_name in ordered_names:
        pressure = dict(pressure_by_plan.get(plan_name, {}))
        items.append(
            {
                "plan_name": plan_name,
                "generated": int(generated_by_plan.get(plan_name, 0)),
                "quota": int(plan_quotas.get(plan_name, 0)),
                "stall_events": int(pressure.get("stall_events", 0)),
                "total_resamples": int(pressure.get("total_resamples", 0)),
                "failed_solutions": int(pressure.get("failed_solutions", 0)),
            }
        )
    quota_total = sum(int(plan_quotas.get(name, 0)) for name in ordered_names)
    generated_total = sum(int(item.get("generated", 0)) for item in items)
    payload: dict[str, object] = {
        "total_generated": int(generated_total),
        "total_quota": int(quota_total),
        "items": items,
    }
    if quota_total > 0:
        payload["quota_progress_pct"] = float(generated_total) * 100.0 / float(quota_total)
    if created_at:
        payload["created_at"] = str(created_at)
    return payload


def _build_payload_from_run_state(
    contract: RunContractSummary,
    *,
    run_root: Path,
    pressure_by_plan: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, object] | None, str | None]:
    path = run_state_path(run_root)
    if not path.exists():
        return None, None
    try:
        state = load_run_state(path)
    except Exception:
        return None, None
    generated_by_plan: dict[str, int] = {}
    for item in state.items:
        plan_name = str(item.plan_name or "").strip()
        if not plan_name:
            continue
        generated_by_plan[plan_name] = generated_by_plan.get(plan_name, 0) + int(item.generated)
    if not generated_by_plan:
        return None, None
    payload = _build_outcome_payload(
        plan_order=[plan.name for plan in contract.plans],
        plan_quotas=_quota_by_plan(contract),
        generated_by_plan=generated_by_plan,
        pressure_by_plan=pressure_by_plan,
        created_at=str(state.updated_at or state.created_at or "").strip() or None,
    )
    return payload, "run_state"


def _build_payload_from_records(
    contract: RunContractSummary,
    *,
    records_path: Path,
    pressure_by_plan: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, object] | None, str | None]:
    if not records_path.exists():
        return None, None
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except Exception:
        return None, None
    try:
        available_columns = set(pq.read_schema(records_path).names)
    except Exception:
        return None, None
    plan_column = None
    for candidate in ("densegen__plan", "plan_name", "plan"):
        if candidate in available_columns:
            plan_column = candidate
            break
    if plan_column is None:
        return None, None
    try:
        frame = pd.read_parquet(records_path, columns=[plan_column])
    except Exception:
        return None, None
    if frame.empty:
        return None, None
    generated_by_plan = frame[plan_column].dropna().astype(str).value_counts(dropna=True).sort_index().to_dict()
    if not generated_by_plan:
        return None, None
    created_at = None
    try:
        created_at = str(records_path.stat().st_mtime)
    except Exception:
        created_at = None
    payload = _build_outcome_payload(
        plan_order=[plan.name for plan in contract.plans],
        plan_quotas=_quota_by_plan(contract),
        generated_by_plan={str(key): int(value) for key, value in generated_by_plan.items()},
        pressure_by_plan=pressure_by_plan,
        created_at=created_at,
    )
    return payload, "records"


def resolve_workspace_outcome(
    contract: RunContractSummary,
    *,
    run_root: Path,
    records_path: Path | None = None,
) -> RunOutcomeSummary:
    manifest_payload, manifest_error = _load_manifest_payload(run_manifest_path(run_root))
    if manifest_payload is not None:
        manifest_outcome = extract_outcome(
            manifest_payload,
            plan_order=[plan.name for plan in contract.plans],
            outcome_source="manifest",
            pressure_source="manifest",
        )
        if manifest_outcome.available:
            return manifest_outcome

    pressure_by_plan, pressure_source = _load_pressure_by_plan(run_root)
    notes: list[str] = []
    if manifest_error:
        notes.append(manifest_error.rstrip("."))

    run_state_payload, run_state_source = _build_payload_from_run_state(
        contract,
        run_root=run_root,
        pressure_by_plan=pressure_by_plan,
    )
    if run_state_payload is not None:
        notes.append(
            "finalized run manifest is unavailable; showing checkpointed progress from `outputs/meta/run_state.json`"
        )
        return extract_outcome(
            run_state_payload,
            plan_order=[plan.name for plan in contract.plans],
            outcome_source=str(run_state_source or "run_state"),
            pressure_source=str(pressure_source or "analysis"),
            notes=tuple(dict.fromkeys(notes)),
            pressure_available=bool(pressure_source),
            pressure_message=(
                ""
                if pressure_source
                else "workspace pressure counters are unavailable; attempts/events artifacts are missing"
            ),
        )

    if records_path is not None:
        records_payload, records_source = _build_payload_from_records(
            contract,
            records_path=records_path,
            pressure_by_plan=pressure_by_plan,
        )
        if records_payload is not None:
            notes.append(
                "finalized run manifest is unavailable; showing output counts from the selected notebook records source"
            )
            return extract_outcome(
                records_payload,
                plan_order=[plan.name for plan in contract.plans],
                outcome_source=str(records_source or "records"),
                pressure_source=str(pressure_source or "analysis"),
                notes=tuple(dict.fromkeys(notes)),
                pressure_available=bool(pressure_source),
                pressure_message=(
                    ""
                    if pressure_source
                    else "workspace pressure counters are unavailable; attempts/events artifacts are missing"
                ),
            )

    return extract_outcome(
        None,
        plan_order=[plan.name for plan in contract.plans],
        error_message=(
            "run outcomes are not available yet; no `run_manifest.json`, `run_state.json`, "
            "or records-derived plan counts were found for this workspace"
        ),
        outcome_source="analysis",
        pressure_source="analysis",
        notes=tuple(dict.fromkeys(notes)),
        pressure_available=False,
        pressure_message="workspace pressure counters are unavailable; attempts/events artifacts are missing",
    )
