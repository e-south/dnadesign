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

from ...adapters.outputs import load_records_from_config
from ...config import LoadedConfig
from .attempts import _load_existing_attempt_index_by_plan, _load_failure_counts_from_attempts
from .usage_tracking import _parse_used_tfbs_detail, _update_usage_counts


@dataclass(frozen=True)
class ResumeState:
    existing_counts: dict[tuple[str, str], int]
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]]
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]]
    attempt_counters: dict[tuple[str, str], int]


def load_resume_state(
    *,
    resume: bool,
    loaded: LoadedConfig,
    tables_root: Path,
    config_sha: str,
) -> ResumeState:
    existing_counts: dict[tuple[str, str], int] = {}
    existing_usage_by_plan: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    site_failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] = {}
    attempt_counters: dict[tuple[str, str], int] = {}
    if not resume:
        return ResumeState(
            existing_counts=existing_counts,
            existing_usage_by_plan=existing_usage_by_plan,
            site_failure_counts=site_failure_counts,
            attempt_counters=attempt_counters,
        )

    site_failure_counts = _load_failure_counts_from_attempts(tables_root)
    attempt_counters = _load_existing_attempt_index_by_plan(tables_root)
    if loaded.root.densegen.output.targets:
        try:
            df_existing, _ = load_records_from_config(
                loaded.root,
                loaded.path,
                columns=[
                    "densegen__run_config_sha256",
                    "densegen__run_id",
                    "densegen__input_name",
                    "densegen__plan",
                    "densegen__used_tfbs_detail",
                ],
            )
        except Exception:
            df_existing = None
        if df_existing is not None and not df_existing.empty:
            if "densegen__run_config_sha256" in df_existing.columns:
                mismatched = df_existing["densegen__run_config_sha256"].dropna().unique().tolist()
                if mismatched and any(val != config_sha for val in mismatched):
                    raise RuntimeError(
                        "Existing outputs were produced with a different config. "
                        "Remove outputs/tables (and outputs/meta if present) "
                        "or stage a new run root to start fresh."
                    )
            if "densegen__run_id" in df_existing.columns:
                run_ids = df_existing["densegen__run_id"].dropna().unique().tolist()
                if run_ids and any(val != loaded.root.densegen.run.id for val in run_ids):
                    raise RuntimeError(
                        "Existing outputs were produced with a different run_id. "
                        "Remove outputs/tables (and outputs/meta if present) "
                        "or stage a new run root to start fresh."
                    )
            if {"densegen__input_name", "densegen__plan"} <= set(df_existing.columns):
                counts = df_existing.groupby(["densegen__input_name", "densegen__plan"]).size().astype(int).to_dict()
                existing_counts = {(str(k[0]), str(k[1])): int(v) for k, v in counts.items()}
            if "densegen__used_tfbs_detail" in df_existing.columns:
                for _, row in df_existing.iterrows():
                    input_name = str(row.get("densegen__input_name") or "")
                    plan_name = str(row.get("densegen__plan") or "")
                    if not input_name or not plan_name:
                        continue
                    key = (input_name, plan_name)
                    counts = existing_usage_by_plan.setdefault(key, {})
                    used = _parse_used_tfbs_detail(row.get("densegen__used_tfbs_detail"))
                    _update_usage_counts(counts, used)

    return ResumeState(
        existing_counts=existing_counts,
        existing_usage_by_plan=existing_usage_by_plan,
        site_failure_counts=site_failure_counts,
        attempt_counters=attempt_counters,
    )
