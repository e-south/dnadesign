"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/loading.py

Load and verify study-owned promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifact import verify_promoter_candidate_bindings
from .parquet_io import read_bindings


def load_promoter_candidate_bindings(
    bundle_dir: Path,
    *,
    allowed_root: Path | None = None,
) -> pd.DataFrame:
    """Return binding rows only after the complete bundle verifies."""

    verified = verify_promoter_candidate_bindings(bundle_dir, allowed_root=allowed_root)
    return read_bindings(verified.bindings_parquet)


__all__ = ["load_promoter_candidate_bindings"]
