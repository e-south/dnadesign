"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/elites.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_elites_parquet(run_dir: Path) -> Path:
    """Return the elites parquet path, preferring the v2 flat layout."""
    preferred = run_dir / "elites.parquet"
    if preferred.exists():
        return preferred

    legacy = list(run_dir.glob("cruncher_elites_*/*.parquet"))
    if legacy:
        latest = max(legacy, key=lambda p: p.stat().st_mtime)
        logger.warning(
            "Using legacy elites layout at %s; re-run `cruncher sample` to regenerate elites.parquet.",
            latest,
        )
        return latest

    raise FileNotFoundError(f"No elites parquet found in {run_dir}")
