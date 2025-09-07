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

from ..errors import WriteBackError
from ..logging import get_logger

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
    """
    Attach results to a USR dataset as namespaced columns.

    Column naming:
      infer__<model_id>__<job_id>__<out_id>

    Implementation:
      - Build a small in-memory frame with id + out columns.
      - Write to a temp Parquet.
      - Call ds.attach(path, namespace="infer", id_col="id", columns=[...]).
    """
    if not columnar:
        _LOG.info("write_back_usr: nothing to write (empty outputs).")
        return

    N = len(ids)
    for out_id, col in columnar.items():
        if len(col) != N:
            raise WriteBackError(
                f"Output column '{out_id}' length={len(col)} doesn't match ids length={N}"
            )

    # Build column names *without* namespace; attach() will prefix "infer__"
    out_cols = {}
    for out_id, col in columnar.items():
        col_name = f"{model_id}__{job_id}__{out_id}"
        out_cols[col_name] = col

    df = pd.DataFrame({"id": ids, **out_cols})

    # Persist to a temp Parquet and attach
    with tempfile.TemporaryDirectory() as tmpd:
        p = Path(tmpd) / "infer_attach.parquet"
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, p)
        ds.attach(
            p,
            namespace="infer",
            id_col="id",
            columns=list(out_cols.keys()),
            allow_overwrite=bool(overwrite),
            note=f"dnadesign.infer job={job_id} model={model_id}",
        )
