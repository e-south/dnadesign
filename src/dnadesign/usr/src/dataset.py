"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/src/dataset.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import (
    AlphabetError,
    DuplicateGroup,
    DuplicateIDError,
    NamespaceError,
    SchemaError,
    SequencesError,
)
from .io import append_event, now_utc, read_parquet, write_parquet_atomic
from .normalize import compute_id, normalize_sequence  # case-preserving
from .schema import ARROW_SCHEMA, REQUIRED_COLUMNS

RECORDS = "records.parquet"
SNAPDIR = "_snapshots"  # standardized
META_MD = "meta.md"
EVENTS_LOG = ".events.log"

_NS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
LEGACY_ALLOW = {
    "logits_mean",
    "logits_mean_dim",
    "logits_mean_model_id",
    "logits_mean_version",
}


@dataclass
class Dataset:
    """A concrete, local dataset located at `<root>/<name>/`."""

    root: Path  # the usr/datasets/ folder
    name: str

    @property
    def dir(self) -> Path:
        return self.root / self.name

    @property
    def records_path(self) -> Path:
        return self.dir / RECORDS

    @property
    def snapshot_dir(self) -> Path:
        return self.dir / SNAPDIR

    @property
    def meta_path(self) -> Path:
        return self.dir / META_MD

    @property
    def events_path(self) -> Path:
        return self.dir / EVENTS_LOG

    # ---- quick stats for CLI and sync ----
    def stats(self):
        """Return local primary FileStat (sha/size/rows/cols/mtime)."""
        from .diff import parquet_stats

        return parquet_stats(self.records_path)

    # ---- lifecycle ----

    def init(self, source: str = "", notes: str = "") -> None:
        """Create a new, empty dataset directory with canonical schema."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.records_path.exists():
            raise SequencesError(f"Dataset already initialized: {self.records_path}")
        empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in ARROW_SCHEMA], schema=ARROW_SCHEMA)
        write_parquet_atomic(empty, self.records_path, self.snapshot_dir)
        ts = now_utc()
        date = ts.split("T")[0]
        meta_md = (
            f"name: {self.name}\n"
            f"created_at: {ts}\n"
            f"source: {source}\n"
            f"notes: {notes}\n"
            f"schema: USR v1\n\n"
            f"### Updates ({date})\n"
            f"- {ts}: initialized dataset.\n"
        )
        self.meta_path.write_text(meta_md, encoding="utf-8")
        append_event(self.events_path, {"action": "init", "dataset": self.name, "source": source})

    # --- lightweight, best-effort scratch-pad logging in meta.md ---
    def append_meta_note(self, title: str, code_block: Optional[str] = None) -> None:
        ts = now_utc()
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            # create minimal header if missing
            hdr = f"name: {self.name}\ncreated_at: {ts}\nsource: \nnotes: \nschema: USR v1\n\n### Updates ({ts.split('T')[0]})\n"  # noqa
            self.meta_path.write_text(hdr, encoding="utf-8")
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(f"- {ts}: {title}\n")
            if code_block:
                f.write("```bash\n")
                f.write(code_block.strip() + "\n")
                f.write("```\n")

    def _require_exists(self) -> None:
        if not self.records_path.exists():
            raise SequencesError(f"Dataset not initialized: {self.records_path}")

    # ---- info ----

    def info(self) -> dict:
        """Basic dataset metadata plus discovered namespaces."""
        self._require_exists()
        tbl = read_parquet(self.records_path)
        cols = list(tbl.schema.names)
        namespaces = sorted(
            {
                c.split("__", 1)[0]
                for c in cols
                if c not in {k for k, _ in REQUIRED_COLUMNS} and c not in LEGACY_ALLOW and "__" in c
            }
        )
        return {
            "name": self.name,
            "path": str(self.records_path),
            "rows": tbl.num_rows,
            "columns": cols,
            "namespaces": namespaces,
        }

    def schema(self):
        """Return the Arrow schema of the current table."""
        self._require_exists()
        return read_parquet(self.records_path).schema

    def head(self, n: int = 10):
        """Return the first N rows as a pandas DataFrame."""
        self._require_exists()
        return read_parquet(self.records_path).to_pandas().head(n)

    # ---- ingest ----

    def import_rows(
        self,
        rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
        *,
        default_bio_type: str = "dna",
        default_alphabet: str = "dna_4",
        source: Optional[str] = None,
        strict_id_check: bool = True,
    ) -> int:
        """
        Import sequence rows (DataFrame or sequence of dicts).

        Expectations per row (case-preserving):
          - 'sequence' (required): string; trimmed; validated by (bio_type, alphabet)
          - 'bio_type'  (optional): defaults to `default_bio_type`
          - 'alphabet'  (optional): defaults to `default_alphabet`
          - 'id'        (optional): if present and `strict_id_check`, must equal
                                   sha1(bio_type|sequence_norm)
          - 'created_at'(optional): if missing, set to now (UTC)
          - 'source'    (optional): defaults to `source` param or ""

        Behavior:
          - computes 'length' from normalized sequence
          - rejects duplicate ids within incoming
          - rejects ids that already exist on disk (append-only semantics)
          - atomic write + snapshot + event log
        """
        self._require_exists()

        # Normalize input → DataFrame
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
            except Exception:
                return False
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

        # Default columns (fail fast on missing/empty values if provided)
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

        # Compute normalized sequences + ids + lengths
        ids, seqs, lens = [], [], []
        for i, (s, bt, ab) in enumerate(zip(seq_raw, bio_vals, alph_vals), start=1):
            try:
                s_norm = normalize_sequence(str(s), str(bt), str(ab))
            except ValueError as e:
                raise AlphabetError(f"Row {i}: {e}") from e
            seqs.append(s_norm)
            ids.append(compute_id(str(bt), s_norm))
            lens.append(len(s_norm))

        # Optional user-supplied ids must match
        if "id" in df_in.columns and strict_id_check:
            bad_id = [
                i for i, v in enumerate(df_in["id"].tolist(), start=1) if _is_missing_scalar(v) or str(v).strip() == ""
            ]
            if bad_id:
                sample = ", ".join(str(i) for i in bad_id[:5])
                raise SchemaError(
                    f"{len(bad_id)} row(s) have missing/empty 'id' (rows: {sample}). "
                    "Drop the column or provide valid ids."
                )
            bad_idx = [
                i for i, (given, want) in enumerate(zip(df_in["id"].astype(str), ids)) if str(given) != str(want)
            ]
            if bad_idx:
                raise SchemaError(
                    f"'id' mismatch for {len(bad_idx)} row(s); ids must equal sha1(bio_type|sequence_norm)."
                )

        # created_at: robust tz-aware timestamp
        if "created_at" in df_in.columns:
            created = pd.to_datetime(df_in["created_at"], utc=True)
        else:
            ts_utc = pd.Timestamp.now(tz="UTC")
            created = pd.Series([ts_utc] * len(df_in), dtype="datetime64[ns, UTC]")

        # source column: prefer explicit param; fall back to per-row source or ""
        src_str = source if source is not None else ""
        if "source" in df_in.columns:
            src_col = df_in["source"].astype(str)
            src_vals = [src_str if src_str else s for s in src_col.tolist()]
        else:
            src_vals = [src_str] * len(df_in)

        # Build outgoing Pandas frame in canonical order/types
        out_df = pd.DataFrame(
            {
                "id": ids,
                "bio_type": bio_vals,
                "sequence": seqs,
                "alphabet": alph_vals,
                "length": lens,
                "source": src_vals,
                "created_at": created,
            }
        )

        # ---------- Duplicate diagnostics (EXACT & CASE-INSENSITIVE) ----------
        # EXACT duplicate groups (same canonical id → byte-for-byte duplicates)
        by_id: Dict[str, List[int]] = defaultdict(list)  # id -> [1-based row idx]
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

        # CASE-INSENSITIVE duplicate groups
        # Treat sequences that are the same *ignoring case* as duplicates too
        # (helps catch accidental duplicates when case encodes no biology).
        # Key: (bio_type_lower, uppercase(sequence))
        bio_list = [str(v) for v in bio_vals]
        casefold_map: Dict[tuple, List[int]] = defaultdict(list)
        for i, (bt, s) in enumerate(zip(bio_list, seqs), start=1):
            casefold_map[(bt.lower(), s.upper())].append(i)
        casefold_groups_all = [(key, rows_idx) for key, rows_idx in casefold_map.items() if len(rows_idx) > 1]
        # Only show groups that are NOT already exact dup groups
        exact_row_sets = {tuple(g.rows) for g in exact_groups}
        casefold_groups = []
        for (_, _), rows_idx in casefold_groups_all:
            rows_idx_sorted = sorted(rows_idx)
            # If this set is identical to an exact dup, skip (we already report it)
            if tuple(rows_idx_sorted) in exact_row_sets:
                continue
            rid_example = ids[rows_idx_sorted[0] - 1]
            seq_example = seqs[rows_idx_sorted[0] - 1]
            casefold_groups.append(
                DuplicateGroup(
                    id=rid_example,
                    count=len(rows_idx_sorted),
                    rows=rows_idx_sorted,
                    sequence=seq_example,
                )
            )
        casefold_groups.sort(key=lambda g: (-g.count, g.id))

        if exact_groups:
            raise DuplicateIDError(
                "Duplicate sequences detected in incoming data (same canonical id).",
                groups=exact_groups[:5],
                hint=(
                    "Remove repeated rows before importing. If you need to keep a single copy, "
                    "deduplicate by the 'sequence' column (case preserved)."
                ),
            )
        if casefold_groups:
            raise DuplicateIDError(
                "Case-insensitive duplicate sequences detected in incoming data "
                "(same letters, different capitalization).",
                casefold_groups=casefold_groups[:5],
                hint=("Sequences that differ only by case are treated as duplicates at import time. "),
            )
        incoming = pa.Table.from_pandas(out_df, schema=ARROW_SCHEMA, preserve_index=False)

        # Merge with existing (append-only; reject collisions)
        if self.records_path.exists():
            existing = read_parquet(self.records_path)
            # -------- case-insensitive duplicate check vs existing on disk --------
            ex_bt = [str(x) for x in existing.column("bio_type").to_pylist()]
            ex_sq = [str(x) for x in existing.column("sequence").to_pylist()]
            existing_cf = {(bt.lower(), sq.upper()) for bt, sq in zip(ex_bt, ex_sq)}
            conflicts = [
                i
                for i, (bt, s) in enumerate(zip(bio_vals, seqs), start=1)
                if (str(bt).lower(), s.upper()) in existing_cf
            ]
            if conflicts:
                preview = [
                    DuplicateGroup(id=ids[i - 1], count=1, rows=[i], sequence=seqs[i - 1]) for i in conflicts[:5]
                ]
                raise DuplicateIDError(
                    "Case-insensitive duplicate sequences already exist in this dataset.",
                    casefold_groups=preview,
                    hint="Run 'usr dedupe-sequences <dataset>' or remove duplicates from the import file.",
                )
            existing_ids = set(existing.column("id").to_pylist())
            inter = existing_ids.intersection(set(out_df["id"].tolist()))
            if inter:
                # Build a compact report for the first few conflicts, with input row indices
                row_index_by_id = defaultdict(list)
                for i, rid in enumerate(ids, start=1):
                    row_index_by_id[rid].append(i)
                preview = []
                for rid in list(inter)[:5]:
                    rows_idx = row_index_by_id.get(rid, [])
                    seq_example = seqs[rows_idx[0] - 1] if rows_idx else ""
                    preview.append(
                        DuplicateGroup(
                            id=rid,
                            count=max(1, len(rows_idx)),
                            rows=rows_idx,
                            sequence=seq_example,
                        )
                    )
                raise DuplicateIDError(
                    f"Duplicate ids already present in dataset: {len(inter)} conflict(s).",
                    groups=preview,
                    hint=(
                        "These sequences already exist in this dataset. "
                        "Remove them from your import file or put new rows in a separate dataset."
                    ),
                )
            combined = pa.concat_tables([existing, incoming], promote_options="default")
            write_parquet_atomic(combined, self.records_path, self.snapshot_dir)
        else:
            write_parquet_atomic(incoming, self.records_path, self.snapshot_dir)

        append_event(
            self.events_path,
            {
                "action": "import_rows",
                "dataset": self.name,
                "n": len(out_df),
                "source_param": source or "",
            },
        )
        return int(len(out_df))

    # Legacy file import entry points now route to import_rows (no special logic)
    def import_csv(
        self,
        csv_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        df = pd.read_csv(csv_path)
        return self.import_rows(
            df,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source or str(csv_path),
        )

    def import_jsonl(
        self,
        jsonl_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        df = pd.read_json(jsonl_path, lines=True)
        return self.import_rows(
            df,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source or str(jsonl_path),
        )

    # ---- generic attach ----

    def attach(
        self,
        path: Path,
        namespace: str,
        *,
        id_col: str = "id",
        columns: Optional[Iterable[str]] = None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        note: str = "",
    ) -> int:
        """
        Attach arbitrary columns under a tool-safe namespace: `<namespace>__<column>`.

        Rules:
        - `namespace` must match `^[a-z][a-z0-9_]*$`
        - Essential columns are immutable
        - Overwriting existing columns requires `allow_overwrite=True`
        - Values align by `id`; missing/ambiguous ids raise unless `allow_missing=True`
        - Strings that look like JSON arrays/objects are parsed when `parse_json=True`

        Supported input formats: .parquet, .csv, .jsonl
        """
        self._require_exists()
        if not _NS_RE.match(namespace):
            raise NamespaceError(
                "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
            )

        # Load incoming attachment (usually small)
        if path.suffix.lower() == ".parquet":
            inc = pq.read_table(path).to_pandas()
        elif path.suffix.lower() in {".csv"}:
            inc = pd.read_csv(path)
        elif path.suffix.lower() in {".jsonl", ".json"}:
            inc = pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
        else:
            raise SchemaError("Unsupported input format. Use parquet|csv|jsonl.")
        if id_col not in inc.columns:
            raise SchemaError(f"Missing id column '{id_col}' in incoming data.")

        rows_incoming = int(len(inc))
        row_nums = list(range(1, rows_incoming + 1))

        # Choose + prefix targets
        attach_cols = [c for c in inc.columns if c != id_col] if columns is None else list(columns)
        if not attach_cols:
            return 0

        work = inc[[id_col] + attach_cols].copy()

        def _normalize_optional_str(x: object) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip()
            if not s:
                return None
            if s.lower() in {"nan", "none"}:
                return None
            return s

        def _parse_jsonish(v: object, col: str, row_idx: int) -> object:
            if not parse_json:
                return v
            if not isinstance(v, str):
                return v
            s = v.strip()
            if not s:
                return v
            if s.startswith("[") or s.startswith("{"):
                try:
                    return json.loads(s)
                except Exception:
                    # try a small fix for single-quoted CSV-y lists
                    if s.startswith("[") and ("'" in s) and ('"' not in s):
                        try:
                            return json.loads(s.replace("'", '"'))
                        except Exception:
                            pass
                    raise SchemaError(
                        f"Column '{col}' row {row_idx}: invalid JSON-like value. "
                        "Provide valid JSON or pass --no-parse-json."
                    )
            return v

        if parse_json:
            for col in attach_cols:
                vals = work[col].tolist()
                parsed = [_parse_jsonish(v, col, i) for i, v in enumerate(vals, start=1)]
                work[col] = parsed

        def target_name(c: str) -> str:
            return c if c.startswith(namespace + "__") else f"{namespace}__{c}"

        targets = [target_name(c) for c in attach_cols]

        rows_missing = 0
        rows_ambiguous = 0

        # --- Map sequence → id if requested ---
        existing_tbl = read_parquet(self.records_path, columns=["id", "sequence", "bio_type"])
        if id_col.lower() == "sequence":
            seq_to_ids: Dict[str, List[str]] = {}
            pair_to_id: Dict[tuple, str] = {}
            # build maps from current dataset
            ds_ids = existing_tbl.column("id").to_pylist()
            ds_seqs = [str(s).strip() for s in existing_tbl.column("sequence").to_pylist()]
            ds_types = [str(bt) for bt in existing_tbl.column("bio_type").to_pylist()]
            for rid, bt, seq in zip(ds_ids, ds_types, ds_seqs):
                pair_to_id[(bt, seq)] = rid
                seq_to_ids.setdefault(seq, []).append(rid)

            incoming_seq = [_normalize_optional_str(s) for s in work[id_col].tolist()]
            incoming_type = (
                [_normalize_optional_str(bt) for bt in inc["bio_type"].tolist()]
                if "bio_type" in inc.columns
                else [None] * len(incoming_seq)
            )
            resolved: List[Optional[str]] = []
            missing_rows: List[int] = []
            ambiguous_rows: List[int] = []
            for idx, (bt, seq) in enumerate(zip(incoming_type, incoming_seq), start=1):
                if seq is None:
                    missing_rows.append(idx)
                    resolved.append(None)
                    continue
                if bt is not None:
                    rid = pair_to_id.get((bt, seq))
                    if rid is None:
                        missing_rows.append(idx)
                        resolved.append(None)
                    else:
                        resolved.append(rid)
                    continue
                ids = seq_to_ids.get(seq, [])
                if len(ids) == 1:
                    resolved.append(ids[0])
                elif len(ids) == 0:
                    missing_rows.append(idx)
                    resolved.append(None)
                else:
                    ambiguous_rows.append(idx)
                    resolved.append(None)

            if ambiguous_rows:
                sample = ", ".join(str(i) for i in ambiguous_rows[:5])
                raise SchemaError(
                    f"Ambiguous sequence→id mapping for {len(ambiguous_rows)} row(s) (rows: {sample}). "
                    f"Run 'usr dedupe-sequences {self.name}' or supply a bio_type column."
                )
            if missing_rows and not allow_missing:
                sample = ", ".join(str(i) for i in missing_rows[:5])
                raise SchemaError(
                    f"{len(missing_rows)} row(s) could not be matched by sequence (rows: {sample}). "
                    "Provide a valid bio_type or pass --allow-missing to skip unmatched rows."
                )

            rows_missing = len(missing_rows)
            rows_ambiguous = len(ambiguous_rows)
            if missing_rows:
                keep_mask = [rid is not None for rid in resolved]
                row_nums = [rn for rn, keep in zip(row_nums, keep_mask) if keep]
                work = work[keep_mask].reset_index(drop=True)
                resolved = [rid for rid in resolved if rid is not None]

            work.insert(0, "id", resolved)
            work = work.drop(columns=[id_col])  # remove 'sequence' identifier column
            # Rename incoming columns to namespaced targets (mirror the --id-col 'id' path)
            work.columns = ["id"] + targets
        else:
            # keep provided 'id' as-is
            work.columns = ["id"] + targets

        # Read existing once; check policy (full schema)
        existing_tbl = read_parquet(self.records_path)
        existing_names = set(existing_tbl.schema.names)

        essential = {k for k, _ in REQUIRED_COLUMNS}
        clobbers = [c for c in targets if c in existing_names]
        if clobbers and not allow_overwrite:
            raise NamespaceError(f"Columns already exist: {', '.join(clobbers)}. Use --allow-overwrite to replace.")
        for t in targets:
            if t in essential:
                raise NamespaceError(f"Refusing to write essential column: {t}")
            if "__" not in t and t not in LEGACY_ALLOW:
                raise NamespaceError(f"Derived columns must be namespaced (got '{t}').")

        # Fast id index (no pandas round-trip)
        id_existing = existing_tbl.column("id").to_pylist()
        nrows = len(id_existing)
        pos = {rid: i for i, rid in enumerate(id_existing)}

        id_series_raw = [str(r) for r in work["id"].tolist()]
        missing_mask = [rid not in pos for rid in id_series_raw]
        missing_ids = [rid for rid, missing in zip(id_series_raw, missing_mask) if missing]
        if missing_ids:
            if not allow_missing:
                sample = ", ".join(missing_ids[:5])
                raise SchemaError(
                    f"{len(missing_ids)} row(s) reference ids not present in the dataset (sample: {sample}). "
                    "Fix the input file or pass --allow-missing to skip unmatched rows."
                )
            rows_missing += len(missing_ids)
            keep_mask = [not m for m in missing_mask]
            row_nums = [rn for rn, keep in zip(row_nums, keep_mask) if keep]
            work = work[keep_mask].reset_index(drop=True)
            id_series = [rid for rid, missing in zip(id_series_raw, missing_mask) if not missing]
        else:
            id_series = id_series_raw

        # Reject duplicate ids in attachment input (avoid silent last-write wins)
        dup_map: Dict[str, List[int]] = defaultdict(list)
        for rid, row_num in zip(id_series, row_nums):
            dup_map[str(rid)].append(row_num)
        dup = {rid: rows for rid, rows in dup_map.items() if len(rows) > 1}
        if dup:
            preview = []
            for rid, rows in list(dup.items())[:3]:
                rows_str = ",".join(str(r) for r in rows[:5])
                preview.append(f"{rid} (rows {rows_str})")
            sample = "; ".join(preview)
            raise SchemaError(
                f"Duplicate ids in attachment input: {len(dup)} id(s) repeated. Sample: {sample}. "
                "Ensure each id appears once per attachment file."
            )

        # ---------- robust null/typing helpers ----------

        def _is_nanlike(x) -> bool:
            """
            Treat only scalar NaN/NA/NaT as null. Lists/arrays/dicts are data, not nulls.
            """
            if x is None:
                return True
            if isinstance(x, (list, tuple, dict)):
                return False
            if isinstance(x, np.ndarray):
                return False
            try:
                res = pd.isna(x)
            except Exception:
                return False
            if isinstance(res, (list, tuple, np.ndarray)):
                return False
            return bool(res)

        def _to_arrow_typed(values: List, target_type: pa.DataType) -> pa.Array:
            cleaned = [None if _is_nanlike(v) else v for v in values]
            if pa.types.is_integer(target_type):
                coerced = [None if v is None else int(v) for v in cleaned]
                return pa.array(coerced, type=target_type)
            if pa.types.is_floating(target_type):
                coerced = [None if v is None else float(v) for v in cleaned]
                return pa.array(coerced, type=target_type)
            if pa.types.is_boolean(target_type):
                coerced = [None if v is None else bool(v) for v in cleaned]
                return pa.array(coerced, type=target_type)
            if pa.types.is_timestamp(target_type):
                return pa.array(cleaned, type=target_type)
            if pa.types.is_string(target_type):
                coerced = [None if v is None else str(v) for v in cleaned]
                return pa.array(coerced, type=target_type)
            return pa.array(cleaned, type=target_type)

        def _infer_arrow(values: List) -> pa.Array:
            cleaned = [None if _is_nanlike(v) else v for v in values]
            if all(v is None for v in cleaned):
                return pa.array(cleaned, type=pa.string())
            if any(isinstance(v, str) for v in cleaned):
                return pa.array([None if v is None else str(v) for v in cleaned], type=pa.string())
            return pa.array(cleaned)

        # id->value map per target (latest wins)
        maps: Dict[str, Dict[str, object]] = {}
        for col in targets:
            colvals = work[col].tolist()
            m: Dict[str, object] = {}
            for rid, val in zip(id_series, colvals):
                m[str(rid)] = val
            maps[col] = m

        # Update only changed columns
        new_tbl = existing_tbl
        for col in targets:
            incoming_map = maps[col]
            if col in existing_names:
                field = new_tbl.schema.field(col)
                base_values = new_tbl.column(col).to_pylist()
                for rid, v in incoming_map.items():
                    idx = pos.get(str(rid))
                    if idx is not None:
                        base_values[idx] = v
                arr = _to_arrow_typed(base_values, field.type)
                col_idx = new_tbl.schema.get_field_index(col)
                new_tbl = new_tbl.set_column(col_idx, pa.field(col, arr.type, nullable=True), arr)
            else:
                base_values = [None] * nrows
                for rid, v in incoming_map.items():
                    idx = pos.get(str(rid))
                    if idx is not None:
                        base_values[idx] = v
                arr = _infer_arrow(base_values)
                new_tbl = new_tbl.add_column(new_tbl.num_columns, pa.field(col, arr.type, nullable=True), arr)

        # Atomic write
        write_parquet_atomic(new_tbl, self.records_path, self.snapshot_dir)

        rows_matched = int(work.shape[0])
        append_event(
            self.events_path,
            {
                "action": "attach",
                "dataset": self.name,
                "path": str(path),
                "namespace": namespace,
                "columns": targets,
                "allow_overwrite": allow_overwrite,
                "rows_incoming": rows_incoming,
                "rows_matched": rows_matched,
                "rows_missing": rows_missing,
                "rows_ambiguous": rows_ambiguous,
                "rows_seen": rows_matched,  # legacy alias
                "note": note,
            },
        )
        return rows_matched

    # Friendly alias for didactic API name in README/examples
    def attach_columns(
        self,
        path: Path,
        namespace: str,
        *,
        id_col: str = "id",
        columns: Optional[Iterable[str]] = None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        note: str = "",
    ) -> int:
        return self.attach(
            path,
            namespace,
            id_col=id_col,
            columns=columns,
            allow_overwrite=allow_overwrite,
            allow_missing=allow_missing,
            parse_json=parse_json,
            note=note,
        )

    # ---- validation & utils ----

    def validate(self, strict: bool = False) -> None:
        """
        Validate schema, ID uniqueness, alphabet constraints, and namespacing policy.
        In strict mode, warnings become errors for alphabet/namespacing issues.
        """
        self._require_exists()
        tbl = read_parquet(self.records_path)
        names = set(tbl.schema.names)

        # required columns present
        for req, _ in REQUIRED_COLUMNS:
            if req not in names:
                raise SchemaError(f"Missing required column: {req}")

        # id uniqueness
        ids = tbl.column("id").to_pylist()
        if len(ids) != len(set(ids)):
            raise DuplicateIDError("Duplicate ids detected.")

        # id<->sequence bijection (case-insensitive on sequence)
        seqs = tbl.column("sequence").to_pylist()
        bt = tbl.column("bio_type").to_pylist()
        # id -> sequence consistency (paranoid; impossible if ids are sha of (bio_type|sequence_norm))
        # and sequence (casefold) -> one id
        seq_key = [(str(b).lower(), str(s).upper()) for b, s in zip(bt, seqs)]
        k_to_ids = defaultdict(set)
        for rid, k in zip(ids, seq_key):
            k_to_ids[k].add(rid)
        bad = [k for k, s in k_to_ids.items() if len(s) > 1]
        if bad:
            msg = "Case-insensitive duplicate sequences found (same letters map to different ids)."
            if strict:
                raise DuplicateIDError(msg)
            else:
                print(f"WARNING: {msg} Run 'usr dedupe-sequences {self.name}'.")

        # alphabet check for dna_4 (case-insensitive)
        if "dna_4" in set(tbl.column("alphabet").to_pylist()):
            seqs = tbl.column("sequence").to_pylist()
            non_acgt = [s for s in seqs if s and re.search(r"[^ACGTacgt]", s)]
            if non_acgt:
                msg = "Non-ACGT characters found under dna_4."
                if strict:
                    raise AlphabetError(msg)
                else:
                    print(f"WARNING: {msg}")

        # namespacing policy (allow legacy columns)
        essential = {k for k, _ in REQUIRED_COLUMNS}
        derived = [c for c in names if c not in essential and c not in LEGACY_ALLOW]
        bad_ns = [c for c in derived if "__" not in c or c.split("__", 1)[0] == ""]
        if bad_ns:
            msg = (
                "Derived columns must be namespaced as '<tool>__<field>'. "
                f"Offending columns: {', '.join(sorted(bad_ns))}"
            )
            if strict:
                raise NamespaceError(msg)
            else:
                print(f"WARNING: {msg}")

    def get(self, record_id: str, columns: Optional[List[str]] = None):
        """Return a single record by id (as a pandas DataFrame row)."""
        self._require_exists()
        df = read_parquet(self.records_path).to_pandas()
        row = df.loc[df["id"] == record_id]
        return row if (columns is None or row.empty) else row[columns]

    def grep(self, pattern: str, limit: int = 20):
        """Regex search across sequences, returning first `limit` hits."""
        self._require_exists()
        df = read_parquet(self.records_path, columns=["id", "sequence", "length"]).to_pandas()
        hits = df[df["sequence"].str.contains(pattern, case=False, regex=True, na=False)]
        return hits.head(limit)

    def export(self, fmt: str, out_path: Path) -> None:
        """Export current table to CSV or JSONL."""
        self._require_exists()
        df = read_parquet(self.records_path).to_pandas()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out_path, index=False)
        elif fmt == "jsonl":
            df.to_json(out_path, orient="records", lines=True)
        else:
            raise SequencesError("Unsupported export format. Use csv|jsonl.")

    def snapshot(self) -> None:
        """Write a timestamped snapshot and atomically persist current table."""
        self._require_exists()
        tbl = read_parquet(self.records_path)
        write_parquet_atomic(tbl, self.records_path, self.snapshot_dir)
        append_event(self.events_path, {"action": "snapshot", "dataset": self.name})
