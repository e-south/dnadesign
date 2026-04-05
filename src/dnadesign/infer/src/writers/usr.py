"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/writers/usr.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

from dnadesign.usr.src.dataset import MUTATION_RESERVED_NAMESPACES
from dnadesign.usr.src.dataset_overlay_ops import _attach_frame_dataset
from dnadesign.usr.src.errors import SchemaError

from .._logging import get_logger
from ..contracts import infer_usr_column_name
from ..errors import WriteBackError

_LOG = get_logger(__name__)
_OVERLAY_GUARD_FILTER_CHUNK_SIZE = 10_000


def _infer_actor(job_id: str) -> dict[str, object]:
    run_id = str(os.getenv("USR_ACTOR_RUN_ID") or "").strip() or f"infer-{job_id}"
    return {
        "tool": "infer",
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


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


def _infer_output_event_context(out_id: str) -> dict[str, object]:
    output_id = str(out_id).strip()
    family, _, remainder = output_id.partition("__")
    payload: dict[str, object] = {
        "id": output_id,
        "family": family or output_id,
        "kind": "metadata" if output_id.startswith("metadata__") else "feature",
    }
    if family == "log_likelihood" and remainder:
        payload["reduction"] = remainder
    elif family == "output_layer_mean" and remainder:
        payload["pool_scope"] = remainder
    elif family == "intermediate_embedding" and remainder:
        selector, _, pool_scope = remainder.partition("__")
        if selector:
            payload["intermediate_selector"] = selector
        if pool_scope:
            payload["pool_scope"] = pool_scope
    return {"infer_output": payload}


def _merge_infer_event_args(
    *,
    columnar: Dict[str, List[object]],
    event_args: Mapping[str, object] | None,
) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    if len(columnar) == 1:
        only_out_id = next(iter(columnar))
        merged.update(_infer_output_event_context(only_out_id))
    if event_args is not None:
        merged.update(dict(event_args))
    return merged or None


def _supports_inprocess_attach(ds) -> bool:
    required_methods = (
        "_auto_freeze_registry",
        "_record_event",
        "_registry",
        "_registry_hash",
        "_validate_registry_schema",
    )
    return all(callable(getattr(ds, name, None)) for name in required_methods) and all(
        hasattr(ds, attr_name) for attr_name in ("dir", "records_path")
    )


def _supports_overlay_part_write(ds) -> bool:
    return callable(getattr(ds, "write_overlay_part", None))


def _attach_usr_frame(
    ds,
    *,
    df: pd.DataFrame,
    model_id: str,
    job_id: str,
    columnar: Dict[str, List[object]],
    out_cols: Dict[str, List[object]],
    overwrite: bool,
    event_args: Mapping[str, object] | None,
) -> None:
    actor = _infer_actor(job_id)
    resolved_event_args = _merge_infer_event_args(columnar=columnar, event_args=event_args)
    _attach_frame_dataset(
        dataset=ds,
        incoming=df,
        namespace="infer",
        key="id",
        key_col="id",
        columns=list(out_cols.keys()),
        allow_overwrite=True,
        allow_missing=False,
        parse_json=True,
        fail_on_non_null_overwrite=not overwrite,
        note=f"dnadesign.infer job={job_id} model={model_id}",
        actor=actor,
        event_args=resolved_event_args,
        reserved_namespaces=MUTATION_RESERVED_NAMESPACES,
    )


def _write_usr_overlay_part(
    ds,
    *,
    df: pd.DataFrame,
    job_id: str,
    columnar: Dict[str, List[object]],
    event_args: Mapping[str, object] | None,
) -> None:
    actor = _infer_actor(job_id)
    resolved_event_args = _merge_infer_event_args(columnar=columnar, event_args=event_args)
    ds.write_overlay_part(
        "infer",
        df,
        key="id",
        allow_missing=False,
        actor=actor,
        event_args=resolved_event_args,
    )


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
    event_args: Mapping[str, object] | None = None,
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

    if _supports_overlay_part_write(ds):
        _guard_usr_overwrite(ds, ids=ids, out_cols=list(out_cols.keys()), overwrite=overwrite)
        _write_usr_overlay_part(
            ds,
            df=df,
            job_id=job_id,
            columnar=columnar,
            event_args=event_args,
        )
        return

    if _supports_inprocess_attach(ds):
        try:
            _attach_usr_frame(
                ds,
                df=df,
                model_id=model_id,
                job_id=job_id,
                columnar=columnar,
                out_cols=out_cols,
                overwrite=overwrite,
                event_args=event_args,
            )
        except SchemaError as error:
            if str(error).startswith("Refusing overwrite for existing values in column "):
                raise WriteBackError(str(error)) from error
            raise
        return

    _guard_usr_overwrite(ds, ids=ids, out_cols=list(out_cols.keys()), overwrite=overwrite)

    with tempfile.TemporaryDirectory() as tmpd:
        p = Path(tmpd) / "infer_attach.parquet"
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, p)
        actor = _infer_actor(job_id)
        _LOG.info(
            "Attaching to USR: rows=%d cols=%s overwrite=%s",
            len(ids),
            list(out_cols.keys()),
            overwrite,
        )
        resolved_event_args = _merge_infer_event_args(columnar=columnar, event_args=event_args)
        ds.attach(
            p,
            namespace="infer",
            key="id",
            key_col="id",
            columns=list(out_cols.keys()),
            allow_overwrite=True,
            note=f"dnadesign.infer job={job_id} model={model_id}",
            actor=actor,
            event_args=resolved_event_args,
        )
