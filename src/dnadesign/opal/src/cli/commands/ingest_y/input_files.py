"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/ingest_y/input_files.py

Input table and parameter loading for `opal ingest-y`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ....storage.parquet_io import read_parquet_df
from .._common import resolve_json_path, resolve_table_path

if TYPE_CHECKING:
    import pandas as pd


def read_label_input_table(path: Path) -> tuple[Path, "pd.DataFrame"]:
    import pandas as pd

    input_path = resolve_table_path(path, label="--csv", must_exist=True, allow_xlsx=True)
    suffix = input_path.suffix.lower()
    if suffix in (".pq", ".parquet"):
        return input_path, read_parquet_df(input_path)
    if suffix == ".xlsx":
        return input_path, pd.read_excel(input_path)
    return input_path, pd.read_csv(input_path)


def read_transform_params(path: Path) -> dict[str, Any]:
    params_path = resolve_json_path(path, label="--params", must_exist=True)
    return json.loads(params_path.read_text())
