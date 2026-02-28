"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/writers/usr.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .._logging import get_logger
from ..errors import WriteBackError

_LOG = get_logger(__name__)


def write_back_usr(
    ds,  # dnadesign.usr.Dataset
    *,
    ids: List[str],
    model_id: str,
    job_id: str,
    columnar: Dict[str, List[object]],
    overwrite: bool,
) -> None:
    if not columnar:
        _LOG.info("write_back_usr: nothing to write (empty outputs).")
        return

    N = len(ids)
    for out_id, col in columnar.items():
        if len(col) != N:
            raise WriteBackError(f"Output column '{out_id}' length={len(col)} doesn't match ids length={N}")

    out_cols = {}
    for out_id, col in columnar.items():
        col_name = f"infer__{model_id}__{job_id}__{out_id}"
        out_cols[col_name] = col

    df = pd.DataFrame({"id": ids, **out_cols})

    with tempfile.TemporaryDirectory() as tmpd:
        p = Path(tmpd) / "infer_attach.parquet"
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, p)
        _LOG.info(
            "Attaching to USR: rows=%d cols=%s overwrite=%s",
            len(ids),
            list(out_cols.keys()),
            overwrite,
        )
        ds.attach(
            p,
            namespace="infer",
            key="id",
            key_col="id",
            columns=list(out_cols.keys()),
            allow_overwrite=True,
            note=f"dnadesign.infer job={job_id} model={model_id}",
        )
