"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/run_state_manager.py

Run-state helpers for pipeline orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..run_paths import run_state_path
from ..run_state import RunState, load_run_state


@dataclass(frozen=True)
class RunStateContext:
    path: Path
    created_at: str
    accepted_config_sha256: list[str]


@dataclass(frozen=True)
class RunStateReconciliation:
    updated: bool
    state_total: int
    durable_total: int


def init_run_state(
    *,
    run_root: Path,
    run_id: str,
    schema_version: str,
    config_sha256: str,
    allow_config_mismatch: bool = False,
) -> RunStateContext:
    state_path = run_state_path(run_root)
    state_created_at = datetime.now(timezone.utc).isoformat()
    accepted_hashes = [str(config_sha256)]
    if state_path.exists():
        existing_state = load_run_state(state_path)
        if existing_state.run_id and existing_state.run_id != run_id:
            raise RuntimeError(
                "Existing run_state.json was created with a different run_id. "
                "Remove run_state.json or stage a new run root to start fresh."
            )
        if existing_state.config_sha256 and existing_state.config_sha256 != config_sha256:
            if not allow_config_mismatch:
                raise RuntimeError(
                    "Existing run_state.json was created with a different config. "
                    "Remove run_state.json or stage a new run root to start fresh."
                )
        accepted_hashes = sorted(
            set(
                [
                    *[str(v) for v in existing_state.accepted_config_sha256 if str(v)],
                    str(existing_state.config_sha256 or ""),
                    str(config_sha256),
                ]
            )
        )
        accepted_hashes = [h for h in accepted_hashes if h]
        if existing_state.created_at:
            state_created_at = existing_state.created_at
    return RunStateContext(path=state_path, created_at=state_created_at, accepted_config_sha256=accepted_hashes)


def reconcile_run_state_with_outputs(
    *,
    path: Path,
    run_id: str,
    schema_version: str,
    config_sha256: str,
    accepted_config_sha256: list[str],
    run_root: str,
    created_at: str,
    existing_counts: dict[tuple[str, str], int],
) -> RunStateReconciliation:
    durable_counts = {
        (str(input_name), str(plan_name)): int(count)
        for (input_name, plan_name), count in existing_counts.items()
        if int(count) > 0
    }
    durable_total = int(sum(durable_counts.values()))
    if not path.exists():
        return RunStateReconciliation(updated=False, state_total=0, durable_total=durable_total)

    existing_state = load_run_state(path)
    if existing_state.run_id and existing_state.run_id != str(run_id):
        raise RuntimeError(
            "Existing run_state.json was created with a different run_id. "
            "Remove run_state.json or stage a new run root to start fresh."
        )

    state_counts: dict[tuple[str, str], int] = {}
    for item in existing_state.items:
        input_name = str(item.input_name).strip()
        plan_name = str(item.plan_name).strip()
        if not input_name or not plan_name:
            continue
        key = (input_name, plan_name)
        state_counts[key] = int(state_counts.get(key, 0)) + max(0, int(item.generated))
    state_total = int(sum(state_counts.values()))
    if state_counts == durable_counts:
        return RunStateReconciliation(updated=False, state_total=state_total, durable_total=durable_total)

    state_created_at = str(existing_state.created_at or created_at)
    accepted_hashes = sorted(
        set(
            [
                *[str(v) for v in existing_state.accepted_config_sha256 if str(v)],
                *[str(v) for v in accepted_config_sha256 if str(v)],
                str(existing_state.config_sha256 or ""),
                str(config_sha256),
            ]
        )
    )
    accepted_hashes = [h for h in accepted_hashes if h]
    write_run_state(
        path=path,
        run_id=str(run_id),
        schema_version=str(schema_version),
        config_sha256=str(config_sha256),
        accepted_config_sha256=accepted_hashes,
        run_root=str(run_root),
        counts=durable_counts,
        created_at=state_created_at,
    )
    return RunStateReconciliation(updated=True, state_total=state_total, durable_total=durable_total)


def write_run_state(
    *,
    path: Path,
    run_id: str,
    schema_version: str,
    config_sha256: str,
    accepted_config_sha256: list[str],
    run_root: str,
    counts: dict[tuple[str, str], int],
    created_at: str,
) -> None:
    state = RunState.from_counts(
        run_id=run_id,
        schema_version=schema_version,
        config_sha256=config_sha256,
        accepted_config_sha256=accepted_config_sha256,
        run_root=run_root,
        counts=counts,
        created_at=created_at,
    )
    state.write_json(path)
