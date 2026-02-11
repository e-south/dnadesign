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


def assert_state_matches_outputs(
    *,
    state_path: Path,
    existing_counts: dict[tuple[str, str], int],
) -> None:
    if state_path.exists() and not existing_counts:
        existing_state = load_run_state(state_path)
        if existing_state.items and sum(item.generated for item in existing_state.items) > 0:
            raise RuntimeError(
                "run_state.json indicates prior progress, but no outputs were found. "
                "Restore outputs or delete run_state.json before resuming."
            )


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
