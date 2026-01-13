"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/elites.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def find_elites_parquet(run_dir: Path) -> Path:
    """Return the elites parquet path for the current run layout."""
    preferred = run_dir / "artifacts" / "elites.parquet"
    if preferred.exists():
        return preferred
    raise FileNotFoundError(
        "No elites parquet found in run artifacts. "
        "Expected artifacts/elites.parquet; re-run `cruncher sample` to regenerate outputs."
    )
