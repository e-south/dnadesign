"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_ingest.py

Dataset ingest and append helpers for USR canonical records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from .errors import AlphabetError, DuplicateGroup, DuplicateIDError, SchemaError
from .normalize import compute_id, normalize_sequence, validate_alphabet, validate_bio_type
from .schema import ARROW_SCHEMA
from .storage.locking import dataset_write_lock
from .storage.parquet import iter_parquet_batches, now_utc, write_parquet_atomic_batches
from .types import AddSequencesResult

if TYPE_CHECKING:
    from .dataset import Dataset


def prepare_import_rows_dataset(
    dataset: Dataset,
    rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
    *,
    default_bio_type: str,
    default_alphabet: str,
    source: Optional[str],
    strict_id_check: bool,
    created_at_override: Optional[str] = None,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        df_in = rows.copy()
    else:
        df_in = pd.DataFrame(list(rows) if rows else [])

    if "sequence" not in df_in.columns:
        raise SchemaError("Missing required column: 'sequence'.")

    def _is_missing_scalar(x: object) -> bool:
        if x is None:
            return True
        if isinstance(x, (list, tuple, dict, np.ndarray)):
            return False
        try:
            res = pd.isna(x)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"Unable to check missingness for value: {x!r}") from e
        if isinstance(res, (list, tuple, np.ndarray)):
            return False
        return bool(res)

    seq_raw = df_in["sequence"].tolist()
    bad_seq = [i for i, s in enumerate(seq_raw, start=1) if _is_missing_scalar(s) or str(s).strip() == ""]
    if bad_seq:
        sample = ", ".join(str(i) for i in bad_seq[:5])
        raise SchemaError(
            f"{len(bad_seq)} row(s) have missing/empty 'sequence' (rows: {sample}). "
            "Provide a non-empty sequence string."
        )

    if "bio_type" in df_in.columns:
        bio_raw = df_in["bio_type"].tolist()
        bad_bt = [i for i, v in enumerate(bio_raw, start=1) if _is_missing_scalar(v) or str(v).strip() == ""]
        if bad_bt:
            sample = ", ".join(str(i) for i in bad_bt[:5])
            raise SchemaError(
                f"{len(bad_bt)} row(s) have missing/empty 'bio_type' (rows: {sample}). "
                "Either provide values or omit the column to use the default."
            )
        bio_vals = [str(v).strip() for v in bio_raw]
    else:
        bio_vals = [default_bio_type] * len(df_in)

    if "alphabet" in df_in.columns:
        alph_raw = df_in["alphabet"].tolist()
        bad_ab = [i for i, v in enumerate(alph_raw, start=1) if _is_missing_scalar(v) or str(v).strip() == ""]
        if bad_ab:
            sample = ", ".join(str(i) for i in bad_ab[:5])
            raise SchemaError(
                f"{len(bad_ab)} row(s) have missing/empty 'alphabet' (rows: {sample}). "
                "Either provide values or omit the column to use the default."
            )
        alph_vals = [str(v).strip() for v in alph_raw]
    else:
        alph_vals = [default_alphabet] * len(df_in)

    ids, seqs, lens, bio_out, alph_out = [], [], [], [], []
    for i, (s, bt, ab) in enumerate(zip(seq_raw, bio_vals, alph_vals), start=1):
        try:
            bt_norm = validate_bio_type(str(bt))
        except ValueError as e:
            raise SchemaError(f"Row {i}: {e}") from e
        try:
            ab_norm = validate_alphabet(bt_norm, str(ab))
        except ValueError as e:
            raise AlphabetError(f"Row {i}: {e}") from e
        try:
            s_norm = normalize_sequence(str(s), bt_norm, ab_norm, validate=False)
        except ValueError as e:
            raise AlphabetError(f"Row {i}: {e}") from e
        seqs.append(s_norm)
        ids.append(compute_id(bt_norm, s_norm))
        lens.append(len(s_norm))
        bio_out.append(bt_norm)
        alph_out.append(ab_norm)

    if "id" in df_in.columns and strict_id_check:
        bad_id = [
            i for i, v in enumerate(df_in["id"].tolist(), start=1) if _is_missing_scalar(v) or str(v).strip() == ""
        ]
        if bad_id:
            sample = ", ".join(str(i) for i in bad_id[:5])
            raise SchemaError(
                f"{len(bad_id)} row(s) have missing/empty 'id' (rows: {sample}). Drop the column or provide valid ids."
            )
        bad_idx = [i for i, (given, want) in enumerate(zip(df_in["id"].astype(str), ids)) if str(given) != str(want)]
        if bad_idx:
            raise SchemaError(f"'id' mismatch for {len(bad_idx)} row(s); ids must equal sha1(bio_type|sequence_norm).")

    if created_at_override is not None:
        created = pd.to_datetime([created_at_override] * len(df_in), utc=True)
    elif "created_at" in df_in.columns:
        created = pd.to_datetime(df_in["created_at"], utc=True)
    else:
        ts_utc = pd.Timestamp.now(tz="UTC")
        created = pd.Series([ts_utc] * len(df_in), dtype="datetime64[ns, UTC]")

    src_str = source if source is not None else ""
    if "source" in df_in.columns:
        src_col = df_in["source"].astype(str)
        src_vals = [src_str if src_str else s for s in src_col.tolist()]
    else:
        src_vals = [src_str] * len(df_in)

    out_df = pd.DataFrame(
        {
            "id": ids,
            "bio_type": bio_out,
            "sequence": seqs,
            "alphabet": alph_out,
            "length": lens,
            "source": src_vals,
            "created_at": created,
        }
    )

    by_id: Dict[str, List[int]] = defaultdict(list)
    for i, rid in enumerate(ids, start=1):
        by_id[str(rid)].append(i)
    exact_groups = [
        DuplicateGroup(
            id=rid,
            count=len(rows_idx),
            rows=rows_idx,
            sequence=seqs[rows_idx[0] - 1],
        )
        for rid, rows_idx in by_id.items()
        if len(rows_idx) > 1
    ]
    exact_groups.sort(key=lambda g: (-g.count, g.id))

    if exact_groups:
        raise DuplicateIDError(
            "Duplicate sequences detected in incoming data (same canonical id).",
            groups=exact_groups[:5],
            hint=(
                "Remove repeated rows before importing. If you need to keep a single copy, "
                "deduplicate by the 'sequence' column (case preserved)."
            ),
        )
    return out_df


def write_import_df_dataset(
    dataset: Dataset,
    out_df: pd.DataFrame,
    *,
    source: Optional[str],
    on_conflict: str,
    actor: Optional[dict] = None,
    return_ids: bool = False,
    write_lock=dataset_write_lock,
) -> int | tuple[int, list[str], list[str]]:
    incoming = pa.Table.from_pandas(out_df, schema=ARROW_SCHEMA, preserve_index=False)
    ids_all = out_df["id"].astype(str).tolist()

    def _write_dataset() -> int | tuple[int, list[str], list[str]]:
        if dataset.records_path.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "import.sqlite"
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE seen (val TEXT PRIMARY KEY)")
                    for batch in iter_parquet_batches(dataset.records_path, columns=["id"]):
                        for rid in batch.column("id").to_pylist():
                            conn.execute("INSERT OR IGNORE INTO seen(val) VALUES (?)", (str(rid),))

                    conflicts = []
                    keep_mask: List[bool] = []
                    for rid in ids_all:
                        cur = conn.execute("INSERT OR IGNORE INTO seen(val) VALUES (?)", (str(rid),))
                        if cur.rowcount == 0:
                            if on_conflict == "ignore":
                                keep_mask.append(False)
                            else:
                                conflicts.append(str(rid))
                                if len(conflicts) >= 5:
                                    break
                        else:
                            keep_mask.append(True)
                    if conflicts and on_conflict != "ignore":
                        sample = ", ".join(conflicts[:5])
                        raise DuplicateIDError(
                            f"Duplicate ids already present in dataset (sample: {sample}).",
                            hint=(
                                "These sequences already exist in this dataset. "
                                "Remove them from your import file or put new rows in a separate dataset."
                            ),
                        )
                    if on_conflict == "ignore":
                        if not any(keep_mask):
                            if return_ids:
                                return 0, [], ids_all
                            return 0
                        out_df_local = out_df.loc[keep_mask].reset_index(drop=True)
                        incoming_local = pa.Table.from_pandas(out_df_local, schema=ARROW_SCHEMA, preserve_index=False)
                        ids_added = [rid for rid, keep in zip(ids_all, keep_mask) if keep]
                        ids_skipped = [rid for rid, keep in zip(ids_all, keep_mask) if not keep]
                    else:
                        out_df_local = out_df
                        incoming_local = incoming
                        ids_added = list(ids_all)
                        ids_skipped = []
                finally:
                    conn.close()

            def _batch_iter():
                for batch in iter_parquet_batches(dataset.records_path):
                    yield batch
                for batch in incoming_local.to_batches():
                    yield batch

            metadata = dataset._base_metadata(created_at=now_utc())  # noqa: SLF001
            write_parquet_atomic_batches(
                _batch_iter(),
                ARROW_SCHEMA,
                dataset.records_path,
                dataset.snapshot_dir,
                metadata=metadata,
            )
            out_count = int(len(out_df_local))
        else:
            metadata = dataset._base_metadata(created_at=now_utc())  # noqa: SLF001
            write_parquet_atomic_batches(
                incoming.to_batches(),
                ARROW_SCHEMA,
                dataset.records_path,
                dataset.snapshot_dir,
                metadata=metadata,
            )
            out_count = int(len(out_df))
            ids_added = list(ids_all)
            ids_skipped = []

        dataset._auto_freeze_registry()  # noqa: SLF001
        dataset._record_event(  # noqa: SLF001
            "import_rows",
            args={"n": int(out_count), "source_param": source or "", "on_conflict": on_conflict},
            metrics={"rows_written": int(out_count), "rows_skipped": len(ids_skipped)},
            actor=actor,
        )
        if return_ids:
            return int(out_count), ids_added, ids_skipped
        return int(out_count)

    with write_lock(dataset.dir):
        return _write_dataset()


def import_rows_dataset(
    dataset: Dataset,
    rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
    *,
    default_bio_type: str = "dna",
    default_alphabet: str = "dna_4",
    source: Optional[str] = None,
    strict_id_check: bool = True,
    actor: Optional[dict] = None,
) -> int:
    dataset._require_exists()  # noqa: SLF001
    out_df = dataset._prepare_import_rows(  # noqa: SLF001
        rows,
        default_bio_type=default_bio_type,
        default_alphabet=default_alphabet,
        source=source,
        strict_id_check=strict_id_check,
    )
    return dataset._write_import_df(  # noqa: SLF001
        out_df,
        source=source,
        on_conflict="error",
        actor=actor,
    )


def add_sequences_dataset(
    dataset: Dataset,
    rows_or_sequences: Union[pd.DataFrame, Sequence[Dict[str, object]], Sequence[str]],
    *,
    bio_type: str,
    alphabet: str,
    source: str = "",
    created_at: Optional[str] = None,
    on_conflict: str = "error",
    actor: Optional[dict] = None,
) -> AddSequencesResult:
    if on_conflict not in {"error", "ignore"}:
        raise SchemaError(f"Unsupported on_conflict '{on_conflict}'.")

    if isinstance(rows_or_sequences, pd.DataFrame):
        rows = rows_or_sequences
    else:
        rows_list = list(rows_or_sequences) if rows_or_sequences is not None else []
        if rows_list and all(isinstance(v, str) for v in rows_list):
            rows = [{"sequence": s} for s in rows_list]
        else:
            rows = rows_list

    out_df = dataset._prepare_import_rows(  # noqa: SLF001
        rows,
        default_bio_type=bio_type,
        default_alphabet=alphabet,
        source=source,
        strict_id_check=True,
        created_at_override=created_at,
    )
    out_count, ids_added, ids_skipped = dataset._write_import_df(  # noqa: SLF001
        out_df,
        source=source,
        on_conflict=on_conflict,
        actor=actor,
        return_ids=True,
    )
    return AddSequencesResult(
        added=int(out_count),
        skipped=len(ids_skipped),
        ids=list(ids_added),
    )


def import_csv_dataset(
    dataset: Dataset,
    csv_path: Path,
    default_bio_type: str = "dna",
    default_alphabet: str = "dna_4",
    source: Optional[str] = None,
) -> int:
    df = pd.read_csv(csv_path)
    return import_rows_dataset(
        dataset,
        df,
        default_bio_type=default_bio_type,
        default_alphabet=default_alphabet,
        source=source or str(csv_path),
    )


def import_jsonl_dataset(
    dataset: Dataset,
    jsonl_path: Path,
    default_bio_type: str = "dna",
    default_alphabet: str = "dna_4",
    source: Optional[str] = None,
) -> int:
    df = pd.read_json(jsonl_path, lines=True)
    return import_rows_dataset(
        dataset,
        df,
        default_bio_type=default_bio_type,
        default_alphabet=default_alphabet,
        source=source or str(jsonl_path),
    )
