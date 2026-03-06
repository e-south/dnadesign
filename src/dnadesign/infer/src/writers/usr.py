"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/writers/usr.py

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
from ..contracts import infer_usr_column_name
from ..errors import WriteBackError

_LOG = get_logger(__name__)


def _existing_infer_overlay_path(ds) -> Path | None:
    if not hasattr(ds, "list_overlays"):
        return None
    try:
        overlays = ds.list_overlays()
    except Exception as error:  # pragma: no cover - defensive conversion at boundary
        raise WriteBackError(f"Unable to inspect existing infer overlay: {error}") from error
    for overlay in overlays:
        if getattr(overlay, "namespace", None) == "infer":
            path = getattr(overlay, "path", None)
            if path is not None:
                return Path(path)
    return None


def _guard_usr_overwrite(ds, *, ids: List[str], out_cols: List[str], overwrite: bool) -> None:
    if overwrite:
        return
    overlay_path = _existing_infer_overlay_path(ds)
    if overlay_path is None or not overlay_path.exists():
        return

    read_cols = ["id", *[col for col in out_cols if col != "id"]]
    try:
        existing = pq.read_table(overlay_path, columns=read_cols).to_pandas()
    except Exception as error:
        raise WriteBackError(f"Unable to scan existing infer overlay: {error}") from error
    if existing.empty:
        return

    id_filter = {str(id_value) for id_value in ids}
    scoped = existing[existing["id"].astype(str).isin(id_filter)]
    if scoped.empty:
        return

    for col_name in out_cols:
        if col_name not in scoped.columns:
            continue
        occupied = scoped[col_name].notna()
        if occupied.any():
            collision_ids = scoped.loc[occupied, "id"].astype(str).tolist()
            sample = ", ".join(collision_ids[:5])
            raise WriteBackError(
                f"Refusing overwrite for existing infer values in column '{col_name}' (sample ids: {sample}). "
                "Re-run with overwrite=true."
            )


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
        col_name = infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=out_id)
        out_cols[col_name] = col

    df = pd.DataFrame({"id": ids, **out_cols})
    _guard_usr_overwrite(ds, ids=ids, out_cols=list(out_cols.keys()), overwrite=overwrite)

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
