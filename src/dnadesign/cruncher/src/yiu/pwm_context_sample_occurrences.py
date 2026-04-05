"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/pwm_context_sample_occurrences.py

Occurrence-table loading helpers for sample-backed YIU PWM context resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.yiu.errors import YIU_PWM_CONTEXT_INVALID, raise_yiu_error


def load_selected_occurrence_rows(*, sample_workspace_root: Path, elite_id: str) -> list[dict[str, Any]]:
    path = sample_workspace_root / "outputs" / "optimize" / "tables" / "elites_occurrences.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise_yiu_error(YIU_PWM_CONTEXT_INVALID, f"sample_context occurrence loading requires pandas ({exc})")

    required = {"elite_id", "tf", "occurrence_rank", "start", "end", "strand", "selected"}
    try:
        import pyarrow.parquet as pq  # type: ignore

        columns = set(pq.read_schema(path).names)
    except Exception:
        columns = set(pd.read_parquet(path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise_yiu_error(
            YIU_PWM_CONTEXT_INVALID,
            f"sample_context occurrence table is missing required columns {missing}: {path}",
        )
    projected = ["elite_id", "tf", "occurrence_rank", "start", "end", "strand", "selected"]
    try:
        frame = pd.read_parquet(path, columns=projected, filters=[("elite_id", "==", elite_id)])
    except Exception:
        frame = pd.read_parquet(path, columns=projected)
        frame = frame.loc[frame["elite_id"].astype(str) == elite_id]
    frame = frame.loc[frame["selected"].astype(bool)]
    if frame.empty:
        return []
    frame = frame.sort_values(["tf", "occurrence_rank", "start", "end", "strand"], kind="stable")
    return frame.to_dict(orient="records")


__all__ = ["load_selected_occurrence_rows"]
