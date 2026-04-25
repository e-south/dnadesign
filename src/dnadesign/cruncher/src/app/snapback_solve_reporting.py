"""
Report-shaping helpers for preserved-site Snapback solve runs.
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.snapback.solve_models import SnapbackSolveHit, SnapbackSolveReport


def build_snapback_solve_report(
    *,
    report: SnapbackSolveReport,
    solve_id: str,
    run_dir: Path,
    materialized_hits: list[SnapbackSolveHit],
) -> SnapbackSolveReport:
    remaining_hits = report.hits[len(materialized_hits) :]
    return report.model_copy(
        update={
            "solve_id": solve_id,
            "run_dir": str(run_dir.resolve()),
            "metadata": report.metadata.model_copy(update={"materialized_hit_count": len(materialized_hits)}),
            "hits": [*materialized_hits, *remaining_hits],
        }
    )


__all__ = ["build_snapback_solve_report"]
