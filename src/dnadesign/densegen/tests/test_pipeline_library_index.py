from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.core.pipeline import _load_existing_library_index


def test_load_existing_library_index_reads_parts(tmp_path: Path) -> None:
    outputs = tmp_path
    df = pd.DataFrame({"sampling_library_index": [1, 2, 5]})
    part = outputs / "attempts_part-000.parquet"
    df.to_parquet(part)
    assert _load_existing_library_index(outputs) == 5
