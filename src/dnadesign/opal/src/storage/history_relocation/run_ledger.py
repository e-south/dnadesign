"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/run_ledger.py

Stages one canonical run ledger for relocated OPAL campaign histories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ...core.utils import file_sha256
from ..ledger import merge_run_meta_frames
from ..parquet_io import table_from_pandas, write_parquet_table
from .contracts import HistoryRelocationPlan, RunHistory
from .inspection import jsonable


def _rebase_artifact_paths(value: Any, *, source_root: str, target_root: str) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _rebase_artifact_paths(item, source_root=source_root, target_root=target_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rebase_artifact_paths(item, source_root=source_root, target_root=target_root) for item in value]
    if isinstance(value, str) and (value == source_root or value.startswith(f"{source_root}/")):
        return f"{target_root}{value[len(source_root) :]}"
    return value


def _run_frame(plan: HistoryRelocationPlan, run: RunHistory) -> pd.DataFrame:
    row = dict(run.run_row)
    if run.round_index in plan.source.rounds:
        row["artifacts"] = _rebase_artifact_paths(
            jsonable(row["artifacts"]),
            source_root=str(plan.source.workdir),
            target_root=str(plan.target.workdir),
        )
    return pd.DataFrame([row])


def stage_canonical_run_ledger(
    plan: HistoryRelocationPlan, *, staging_root: Path
) -> tuple[Path, list[dict[str, object]]]:
    runs = sorted((*plan.source.runs, *plan.target.runs), key=lambda item: item.round_index)
    frame = merge_run_meta_frames(*(_run_frame(plan, run) for run in runs))
    frame = frame.sort_values(["as_of_round", "run_id"], kind="stable").reset_index(drop=True)
    output_dir = staging_root / "outputs" / "ledger" / "runs.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"part-history-r{runs[0].round_index}-r{runs[-1].round_index}.parquet"
    write_parquet_table(output, table_from_pandas(frame))
    transformations = [
        {
            "path": run.run_part.relative_to(
                plan.source.workdir if run.round_index in plan.source.rounds else plan.target.workdir
            ).as_posix(),
            "target_path": output.relative_to(staging_root).as_posix(),
            "kind": "run_ledger_schema_union",
            "source_sha256": file_sha256(run.run_part),
            "target_sha256": file_sha256(output),
        }
        for run in runs
    ]
    return output_dir, transformations
