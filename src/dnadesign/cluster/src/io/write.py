"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/io/write.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def _backup_file(p: Path, backup_suffix: str = ".bak") -> Path:
    b = p.with_suffix(p.suffix + backup_suffix)
    if not b.exists():
        shutil.copy2(p, b)
    return b


def attach_usr(
    usr_root: Path, dataset: str, cols_df: pd.DataFrame, allow_overwrite: bool = False
) -> None:
    """Attach columns to a USR dataset using its Python API. We require dnadesign.usr.Dataset."""
    try:
        from dnadesign.usr import Dataset
    except Exception as e:
        raise RuntimeError(
            "dnadesign.usr is required to attach columns to a USR dataset. "
            "Please install the dnadesign package."
        ) from e
    ds = Dataset(usr_root, dataset)
    # We assume 'cols_df' contains 'id' plus one or more *namespaced* columns
    # (e.g., 'cluster__ldn_v1', 'cluster__ldn_v1__meta', ...).
    # USR requires an explicit namespace; infer it and *fail fast* if ambiguous.
    non_id = [c for c in cols_df.columns if c != "id"]
    if not non_id:
        return
    ns_tokens = {c.split("__", 1)[0] for c in non_id if "__" in c}
    if len(ns_tokens) != 1:
        raise RuntimeError(
            "attach_usr() requires all columns (except 'id') to share the same '<namespace>__' prefix. "
            f"Columns seen: {sorted(non_id)[:5]}..."
        )
    namespace = next(iter(ns_tokens))
    tmp = usr_root / dataset / ".__cluster_attach_temp.parquet"
    cols_df.to_parquet(tmp, index=False)
    try:
        # Pass the explicit namespace; Dataset.attach() will keep already
        # namespaced columns intact (it won't double-prefix).
        ds.attach_columns(
            tmp,
            namespace=namespace,
            id_col="id",
            columns=[c for c in cols_df.columns if c != "id"],
            allow_overwrite=allow_overwrite,
        )
    finally:
        if tmp.exists():
            tmp.unlink()
    # Optionally append to .events.log (best-effort)
    try:
        event_log = usr_root / dataset / ".events.log"
        with event_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "cluster_attach",
                        "columns": [c for c in cols_df.columns if c != "id"],
                    }
                )
                + "\n"
            )
    except Exception:
        pass


def drop_usr_columns(usr_root: Path, dataset: str, columns: list[str]) -> None:
    """
    Delete namespaced columns from a USR dataset by editing records.parquet atomically.
    ─ Only 'cluster__*' columns are allowed (defensive guard).
    ─ Uses dnadesign.usr I/O helpers for snapshots + metadata preservation + event log.
    """
    if not columns:
        return
    # Normalize dotted leaf paths (e.g., 'cluster__ldn_v1__meta.algo') to their
    # top‑level column ('cluster__ldn_v1__meta') and de‑duplicate.
    columns = list(dict.fromkeys(c.split(".", 1)[0] for c in columns))
    non_cluster = [c for c in columns if not c.startswith("cluster__")]
    if non_cluster:
        raise RuntimeError(
            "Refusing to delete non-namespaced or non-cluster columns. "
            f"Only 'cluster__*' allowed; offending: {non_cluster[:5]}..."
        )
    # Resolve dataset paths deterministically
    ds_dir = (usr_root / dataset).resolve()
    records = ds_dir / "records.parquet"
    snapshots = ds_dir / "_snapshots"
    events = ds_dir / ".events.log"
    if not records.exists():
        raise FileNotFoundError(f"USR dataset not found: {records}")

    # Use USR I/O primitives so we inherit atomic write + snapshot + metadata preservation.
    try:
        from dnadesign.usr.src.io import (  # type: ignore
            append_event,
            read_parquet,
            write_parquet_atomic,
        )
    except Exception as e:
        raise RuntimeError(
            "dnadesign.usr.src.io is required to modify USR datasets. "
            "Please ensure the dnadesign.usr package is installed and up to date."
        ) from e

    tbl = read_parquet(records)
    names = set(tbl.schema.names)  # top‑level Arrow names
    drop_now = [c for c in columns if c in names]
    if not drop_now:
        return
    keep = [c for c in tbl.schema.names if c not in drop_now]
    # Build new table by selection (pyarrow Table.select keeps metadata on preserved columns)
    new_tbl = tbl.select(keep)

    # Persist atomically and snapshot
    write_parquet_atomic(new_tbl, records, snapshots, preserve_metadata_from=tbl)

    # Append a concise event
    try:
        append_event(events, {"action": "cluster_delete", "columns": drop_now})
    except Exception:
        pass


def write_generic(
    src_file: Path,
    df: pd.DataFrame,
    *,
    inplace: bool,
    out: Path | None,
    backup_suffix: str,
) -> Path:
    if inplace and out is not None:
        raise ValueError("Pass either --inplace or --out, not both.")
    if not inplace and out is None:
        raise ValueError("Provide --out when not using --inplace.")
    if inplace:
        _backup_file(src_file, backup_suffix=backup_suffix)
        if src_file.suffix.lower() == ".parquet":
            df.to_parquet(src_file, index=False)
        elif src_file.suffix.lower() == ".csv":
            df.to_csv(src_file, index=False)
        else:
            raise ValueError(f"Unsupported file type: {src_file}")
        return src_file
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
        elif out.suffix.lower() == ".csv":
            df.to_csv(out, index=False)
        else:
            raise ValueError(f"Unsupported out path: {out}")
        return out
