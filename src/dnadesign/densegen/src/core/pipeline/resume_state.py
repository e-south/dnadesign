"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/resume_state.py

Resume-state helpers for pipeline runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...adapters.outputs import scan_records_from_config
from ...config import LoadedConfig
from .attempts import (
    _load_existing_attempt_index_by_plan,
    _load_existing_library_build_count_by_plan,
    _load_failure_counts_from_attempts,
)
from .usage_tracking import _parse_used_tfbs_detail, _update_usage_counts


@dataclass(frozen=True)
class ResumeState:
    existing_counts: dict[tuple[str, str], int]
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]]
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]]
    attempt_counters: dict[tuple[str, str], int]
    library_build_counts: dict[tuple[str, str], int]


class ResumeStateLoadError(RuntimeError):
    pass


def load_resume_state(
    *,
    resume: bool,
    loaded: LoadedConfig,
    tables_root: Path,
    config_sha: str,
    allowed_config_sha256: list[str] | None = None,
) -> ResumeState:
    existing_counts: dict[tuple[str, str], int] = {}
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] = {}
    attempt_counters: dict[tuple[str, str], int] = {}
    library_build_counts: dict[tuple[str, str], int] = {}
    if not resume:
        return ResumeState(
            existing_counts=existing_counts,
            existing_usage_by_plan=existing_usage_by_plan,
            site_failure_counts=site_failure_counts,
            attempt_counters=attempt_counters,
            library_build_counts=library_build_counts,
        )

    site_failure_counts = _load_failure_counts_from_attempts(tables_root)
    attempt_counters = _load_existing_attempt_index_by_plan(tables_root)
    library_build_counts = _load_existing_library_build_count_by_plan(tables_root)
    if loaded.root.densegen.output.targets:
        run_ids: set[str] = set()
        try:
            rows, source_label = scan_records_from_config(
                loaded.root,
                loaded.path,
                columns=[
                    "densegen__run_id",
                    "densegen__input_name",
                    "densegen__plan",
                    "densegen__used_tfbs_detail",
                ],
            )
        except Exception as exc:
            raise ResumeStateLoadError("Failed to scan existing output records while preparing resume state.") from exc
        try:
            for row in rows:
                run_id = row.get("densegen__run_id")
                if run_id is not None:
                    run_id_val = str(run_id).strip()
                    if run_id_val:
                        run_ids.add(run_id_val)

                input_name = str(row.get("densegen__input_name") or "")
                plan_name = str(row.get("densegen__plan") or "")
                if not input_name or not plan_name:
                    continue

                key = (input_name, plan_name)
                existing_counts[key] = int(existing_counts.get(key, 0)) + 1
                counts = existing_usage_by_plan.setdefault(key, {})
                used = _parse_used_tfbs_detail(row.get("densegen__used_tfbs_detail"))
                _update_usage_counts(counts, used)
        except Exception as exc:
            raise ResumeStateLoadError(
                f"Failed to parse scanned output records from `{source_label}` while preparing resume state."
            ) from exc
        if run_ids and any(val != loaded.root.densegen.run.id for val in run_ids):
            raise RuntimeError(
                "Existing outputs were produced with a different run_id. "
                "Remove outputs/tables (and outputs/meta if present) "
                "or stage a new run root to start fresh."
            )

    return ResumeState(
        existing_counts=existing_counts,
        existing_usage_by_plan=existing_usage_by_plan,
        site_failure_counts=site_failure_counts,
        attempt_counters=attempt_counters,
        library_build_counts=library_build_counts,
    )
