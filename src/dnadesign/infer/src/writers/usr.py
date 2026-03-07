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
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

from .._logging import get_logger
from ..contracts import infer_usr_column_name
from ..errors import WriteBackError

_LOG = get_logger(__name__)
_OVERLAY_GUARD_FILTER_CHUNK_SIZE = 10_000


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


def _dedupe_ids(ids: List[str]) -> List[str]:
    unique_ids: List[str] = []
    seen: set[str] = set()
    for row_id in ids:
        value = str(row_id).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique_ids.append(value)
    return unique_ids


def _read_overlay_subset(*, overlay_path: Path, read_cols: List[str], ids: List[str]) -> pd.DataFrame:
    unique_ids = _dedupe_ids(ids)
    if not unique_ids:
        return pd.DataFrame(columns=read_cols)

    frames: List[pd.DataFrame] = []
    for start in range(0, len(unique_ids), _OVERLAY_GUARD_FILTER_CHUNK_SIZE):
        id_chunk = unique_ids[start : start + _OVERLAY_GUARD_FILTER_CHUNK_SIZE]
        try:
            table = pq.read_table(overlay_path, columns=read_cols, filters=[("id", "in", id_chunk)])
        except Exception as error:
            raise WriteBackError(f"Unable to scan existing infer overlay: {error}") from error
        if table.num_rows == 0:
            continue
        frames.append(table.to_pandas())

    if not frames:
        return pd.DataFrame(columns=read_cols)
    return pd.concat(frames, ignore_index=True)


def _guard_usr_overwrite(ds, *, ids: List[str], out_cols: List[str], overwrite: bool) -> None:
    if overwrite:
        return
    overlay_path = _existing_infer_overlay_path(ds)
    if overlay_path is None or not overlay_path.exists():
        return

    try:
        schema_names = set(pa_dataset.dataset(overlay_path, format="parquet").schema.names)
    except Exception as error:
        raise WriteBackError(f"Unable to inspect existing infer overlay schema: {error}") from error
    if "id" not in schema_names:
        raise WriteBackError("Existing infer overlay is missing required 'id' column.")

    read_cols = ["id", *[col for col in out_cols if col != "id" and col in schema_names]]
    if len(read_cols) == 1:
        return
    existing = _read_overlay_subset(overlay_path=overlay_path, read_cols=read_cols, ids=ids)
    if existing.empty:
        return

    for col_name in out_cols:
        if col_name not in existing.columns:
            continue
        occupied = existing[col_name].notna()
        if occupied.any():
            collision_ids = existing.loc[occupied, "id"].astype(str).tolist()
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
