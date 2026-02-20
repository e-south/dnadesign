"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_parquet_io_roundtrip.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_parquet_list_roundtrip_preserves_vectors(tmp_path: Path) -> None:
    from dnadesign.opal.src.storage.parquet_io import read_parquet_df, write_parquet_df

    path = tmp_path / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "vec": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
    )
    write_parquet_df(path, df, index=False)
    out = read_parquet_df(path)
    vals = out["vec"].tolist()
    assert all(isinstance(v, (list, tuple, np.ndarray)) for v in vals)
    assert [list(map(float, v)) for v in vals] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
