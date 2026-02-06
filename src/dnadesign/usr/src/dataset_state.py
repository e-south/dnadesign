"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_state.py

State and tombstone lifecycle operations for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
import pyarrow as pa

from .errors import SchemaError
from .events import record_event
from .overlays import overlay_dir_path, overlay_path
from .storage.locking import dataset_write_lock


class DatasetStateHost(Protocol):
    dir: Path
    events_path: Path
    name: str
    records_path: Path
    root: Path

    def _require_exists(self) -> None: ...

    def _write_reserved_overlay(
        self,
        namespace: str,
        key: str,
        overlay_df: pd.DataFrame,
        *,
        validate_registry: bool = False,
        schema_types: dict[str, pa.DataType] | None = None,
    ) -> int: ...


def _normalize_ids(ids: Sequence[str], *, empty_message: str, duplicate_message: str) -> list[str]:
    ids_list = [str(i).strip() for i in ids if str(i).strip()]
    if not ids_list:
        raise SchemaError(empty_message)
    if len(ids_list) != len(set(ids_list)):
        raise SchemaError(duplicate_message)
    return ids_list


def ensure_ids_exist(dataset: DatasetStateHost, ids: list[str]) -> None:
    from .storage.parquet import iter_parquet_batches

    targets = set(ids)
    found: set[str] = set()
    for batch in iter_parquet_batches(dataset.records_path, columns=["id"]):
        for rid in batch.column("id").to_pylist():
            value = str(rid)
            if value in targets:
                found.add(value)
        if len(found) == len(targets):
            break
    missing = sorted(targets - found)
    if missing:
        sample = ", ".join(missing[:5])
        raise SchemaError(f"{len(missing)} id(s) not found in dataset (sample: {sample}).")


def tombstone(
    dataset: DatasetStateHost,
    ids: Sequence[str],
    *,
    reason: str | None = None,
    deleted_at: str | None = None,
    allow_missing: bool = False,
    tombstone_namespace: str,
) -> int:
    dataset._require_exists()
    ids_list = _normalize_ids(
        ids,
        empty_message="Provide at least one id to tombstone.",
        duplicate_message="Duplicate ids provided to tombstone.",
    )
    if not allow_missing:
        ensure_ids_exist(dataset, ids_list)

    timestamp = pd.to_datetime(deleted_at, utc=True) if deleted_at is not None else pd.Timestamp.now(tz="UTC")
    ts_series = pd.Series([timestamp] * len(ids_list), dtype="datetime64[ns, UTC]")
    overlay_df = pd.DataFrame(
        {
            "id": ids_list,
            "usr__deleted": [True] * len(ids_list),
            "usr__deleted_at": ts_series,
            "usr__deleted_reason": [reason] * len(ids_list),
        }
    )

    with dataset_write_lock(dataset.dir):
        rows = dataset._write_reserved_overlay(tombstone_namespace, "id", overlay_df)
        record_event(
            dataset.events_path,
            "tombstone",
            dataset=dataset.name,
            args={"rows": rows, "reason": reason or "", "allow_missing": allow_missing},
            target_path=dataset.records_path,
            dataset_root=dataset.root,
        )
        return rows


def restore(
    dataset: DatasetStateHost,
    ids: Sequence[str],
    *,
    allow_missing: bool = False,
    tombstone_namespace: str,
) -> int:
    dataset._require_exists()
    ids_list = _normalize_ids(
        ids,
        empty_message="Provide at least one id to restore.",
        duplicate_message="Duplicate ids provided to restore.",
    )
    if not allow_missing:
        ensure_ids_exist(dataset, ids_list)

    ts_series = pd.Series([pd.NaT] * len(ids_list), dtype="datetime64[ns, UTC]")
    overlay_df = pd.DataFrame(
        {
            "id": ids_list,
            "usr__deleted": [False] * len(ids_list),
            "usr__deleted_at": ts_series,
            "usr__deleted_reason": [None] * len(ids_list),
        }
    )

    with dataset_write_lock(dataset.dir):
        rows = dataset._write_reserved_overlay(tombstone_namespace, "id", overlay_df)
        record_event(
            dataset.events_path,
            "restore",
            dataset=dataset.name,
            args={"rows": rows, "allow_missing": allow_missing},
            target_path=dataset.records_path,
            dataset_root=dataset.root,
        )
        return rows


def set_state(
    dataset: DatasetStateHost,
    ids: Sequence[str],
    *,
    masked: bool | None = None,
    qc_status: str | None = None,
    split: str | None = None,
    supersedes: str | None = None,
    lineage: Sequence[str] | str | None = None,
    allow_missing: bool = False,
    state_namespace: str,
    state_schema_types: dict[str, pa.DataType],
    state_qc_status_allowed: set[str],
    state_split_allowed: set[str],
) -> int:
    dataset._require_exists()
    ids_list = _normalize_ids(
        ids,
        empty_message="Provide at least one id to update usr_state.",
        duplicate_message="Duplicate ids provided to update usr_state.",
    )
    if not allow_missing:
        ensure_ids_exist(dataset, ids_list)

    if masked is None and qc_status is None and split is None and supersedes is None and lineage is None:
        raise SchemaError("Provide at least one usr_state field to update.")
    if masked is not None and not isinstance(masked, bool):
        raise SchemaError("usr_state__masked must be a boolean.")

    def _normalize_text(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise SchemaError(f"{field} cannot be empty.")
        return text

    qc_status = _normalize_text(qc_status, "usr_state__qc_status")
    split = _normalize_text(split, "usr_state__split")
    supersedes = _normalize_text(supersedes, "usr_state__supersedes")
    if qc_status is not None and qc_status not in state_qc_status_allowed:
        allowed = ", ".join(sorted(state_qc_status_allowed))
        raise SchemaError(f"usr_state__qc_status must be one of: {allowed}.")
    if split is not None and split not in state_split_allowed:
        allowed = ", ".join(sorted(state_split_allowed))
        raise SchemaError(f"usr_state__split must be one of: {allowed}.")

    lineage_vals: list[list[str]] | None = None
    if lineage is not None:
        if isinstance(lineage, str):
            items = [lineage.strip()]
        else:
            items = [str(v).strip() for v in lineage]
        items = [value for value in items if value]
        if not items:
            raise SchemaError("usr_state__lineage cannot be empty.")
        lineage_vals = [items] * len(ids_list)

    data: dict[str, list] = {"id": ids_list}
    if masked is not None:
        data["usr_state__masked"] = [bool(masked)] * len(ids_list)
    if qc_status is not None:
        data["usr_state__qc_status"] = [qc_status] * len(ids_list)
    if split is not None:
        data["usr_state__split"] = [split] * len(ids_list)
    if supersedes is not None:
        data["usr_state__supersedes"] = [supersedes] * len(ids_list)
    if lineage_vals is not None:
        data["usr_state__lineage"] = lineage_vals

    overlay_df = pd.DataFrame(data)
    schema_types = {k: state_schema_types[k] for k in overlay_df.columns if k in state_schema_types}

    with dataset_write_lock(dataset.dir):
        rows = dataset._write_reserved_overlay(
            state_namespace,
            "id",
            overlay_df,
            validate_registry=True,
            schema_types=schema_types,
        )
        record_event(
            dataset.events_path,
            "state_set",
            dataset=dataset.name,
            args={
                "rows": rows,
                "masked": masked,
                "qc_status": qc_status or "",
                "split": split or "",
                "supersedes": supersedes or "",
                "lineage_count": len(lineage_vals[0]) if lineage_vals else 0,
                "allow_missing": allow_missing,
            },
            metrics={"rows": rows},
            target_path=dataset.records_path,
            dataset_root=dataset.root,
        )
        return rows


def clear_state(
    dataset: DatasetStateHost,
    ids: Sequence[str],
    *,
    allow_missing: bool = False,
    state_namespace: str,
    state_schema_types: dict[str, pa.DataType],
) -> int:
    dataset._require_exists()
    ids_list = _normalize_ids(
        ids,
        empty_message="Provide at least one id to clear usr_state.",
        duplicate_message="Duplicate ids provided to clear usr_state.",
    )
    if not allow_missing:
        ensure_ids_exist(dataset, ids_list)

    overlay_df = pd.DataFrame(
        {
            "id": ids_list,
            "usr_state__masked": [False] * len(ids_list),
            "usr_state__qc_status": [None] * len(ids_list),
            "usr_state__split": [None] * len(ids_list),
            "usr_state__supersedes": [None] * len(ids_list),
            "usr_state__lineage": [None] * len(ids_list),
        }
    )
    schema_types = {k: state_schema_types[k] for k in overlay_df.columns if k in state_schema_types}

    with dataset_write_lock(dataset.dir):
        rows = dataset._write_reserved_overlay(
            state_namespace,
            "id",
            overlay_df,
            validate_registry=True,
            schema_types=schema_types,
        )
        record_event(
            dataset.events_path,
            "state_clear",
            dataset=dataset.name,
            args={"rows": rows, "allow_missing": allow_missing},
            metrics={"rows": rows},
            target_path=dataset.records_path,
            dataset_root=dataset.root,
        )
        return rows


def get_state(
    dataset: DatasetStateHost,
    ids: Sequence[str],
    *,
    allow_missing: bool = False,
    state_namespace: str,
) -> pd.DataFrame:
    dataset._require_exists()
    ids_list = _normalize_ids(
        ids,
        empty_message="Provide at least one id to fetch usr_state.",
        duplicate_message="Duplicate ids provided to fetch usr_state.",
    )
    if not allow_missing:
        ensure_ids_exist(dataset, ids_list)

    state_path = overlay_path(dataset.dir, state_namespace)
    state_dir = overlay_dir_path(dataset.dir, state_namespace)
    overlay_exists = state_path.exists() or state_dir.exists()

    def _defaults() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": ids_list,
                "usr_state__masked": [False] * len(ids_list),
                "usr_state__qc_status": [None] * len(ids_list),
                "usr_state__split": [None] * len(ids_list),
                "usr_state__supersedes": [None] * len(ids_list),
                "usr_state__lineage": [None] * len(ids_list),
            }
        )

    if not overlay_exists:
        return _defaults()

    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SchemaError("get_state requires duckdb (install duckdb).") from exc

    ids_tbl = pa.table({"idx": list(range(len(ids_list))), "id": ids_list})
    overlay_sql = (str(state_dir / "part-*.parquet") if state_dir.exists() else str(state_path)).replace("'", "''")
    con = duckdb.connect()
    try:
        con.register("ids_tbl", ids_tbl)
        con.execute(f"CREATE TEMP VIEW state AS SELECT * FROM read_parquet('{overlay_sql}')")
        cols = {row[1] for row in con.execute("PRAGMA table_info('state')").fetchall()}

        def _sel(name: str) -> str:
            if name in cols:
                return f"s.{name} AS {name}"
            return f"NULL AS {name}"

        select_cols = ", ".join(
            [
                _sel("usr_state__masked"),
                _sel("usr_state__qc_status"),
                _sel("usr_state__split"),
                _sel("usr_state__supersedes"),
                _sel("usr_state__lineage"),
            ]
        )
        out = con.execute(
            "SELECT i.idx, i.id, "
            + select_cols
            + " FROM ids_tbl i LEFT JOIN state s ON i.id = s.id "
            + "ORDER BY i.idx"
        ).fetch_arrow_table()
    finally:
        con.close()

    df = out.to_pandas()
    df["usr_state__masked"] = df["usr_state__masked"].fillna(False).astype(bool)
    return df.drop(columns=["idx"])
