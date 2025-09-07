"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/dataset.py

Parquet-backed dataset with fail-fast schema checks, atomic writes, bounded snapshots,
and an append-only event log. Datasets have immutable essential columns and are
extended by namespaced derived columns via `Dataset.attach(...)`.

Key ideas:
- One `records.parquet` per dataset directory (single source of truth)
- Metadata embedded in Parquet schema (and optional meta.yaml for human read)
- Case-preserving sequences; `id = sha1(bio_type|sequence)` on trimmed text
- Strict namespacing: derived columns must look like `<tool>__<field>`
- Pragmatic safety: atomic writes + bounded snapshots + event log

Module-level controls (edit to taste; no env vars required):
  - SNAPDIR: subdirectory name for snapshots (default comes from io.SNAPSHOT_DIR_NAME)
  - WRITE_META_FILE: whether to write a human-readable meta.yaml (default True)

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import (
    AlphabetError,
    DuplicateIDError,
    NamespaceError,
    SchemaError,
    SequencesError,
)
from .io import (
    append_event,
    now_utc,
    read_parquet,
    write_parquet_atomic,
    SNAPSHOT_DIR_NAME,   # default snapshot dir name
)
from .normalize import compute_id, normalize_sequence  # case-preserving
from .schema import ARROW_SCHEMA, REQUIRED_COLUMNS


# ---------------------------
# Module-level configuration
# ---------------------------
RECORDS = "records.parquet"
SNAPDIR = SNAPSHOT_DIR_NAME
META_YAML = "meta.yaml"
EVENTS_LOG = ".events.log"
WRITE_META_FILE: bool = True  # set False to skip writing meta.yaml


_NS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
LEGACY_ALLOW = {
    "logits_mean",
    "logits_mean_dim",
    "logits_mean_model_id",
    "logits_mean_version",
}


def _embed_meta(schema: pa.Schema, meta_dict: Dict[str, str]) -> pa.Schema:
    """
    Merge `meta_dict` into Arrow schema metadata (keys under 'usr:*').
    Arrow requires bytes->bytes dict.
    """
    meta = dict(schema.metadata or {})
    for k, v in meta_dict.items():
        meta_key = f"usr:{k}".encode("utf-8")
        meta_val = (v if isinstance(v, str) else json.dumps(v)).encode("utf-8")
        meta[meta_key] = meta_val
    return schema.with_metadata(meta)


@dataclass
class Dataset:
    """A concrete, local dataset located at `<root>/<name>/`."""

    root: Path  # the usr/datasets/ folder
    name: str

    @property
    def dir(self) -> Path:
        return Path(self.root) / self.name

    @property
    def records_path(self) -> Path:
        return self.dir / RECORDS

    @property
    def snapshot_dir(self) -> Path:
        return self.dir / SNAPDIR

    @property
    def meta_path(self) -> Path:
        return self.dir / META_YAML

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
        """Create a new, empty dataset directory with canonical schema + embedded metadata."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.records_path.exists():
            raise SequencesError(f"Dataset already initialized: {self.records_path}")

        # Prepare embedded metadata
        meta = {
            "name": self.name,
            "schema": "USR v1",
            "created_at": now_utc(),
            "source": source,
            "notes": notes,
        }
        schema_with_meta = _embed_meta(ARROW_SCHEMA, meta)

        empty = pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in schema_with_meta], schema=schema_with_meta
        )

        # First write persists metadata into records.parquet (and snapshot per policy)
        write_parquet_atomic(empty, self.records_path, self.snapshot_dir)

        # Optional plain-text meta.yaml for human-readability
        if WRITE_META_FILE:
            yaml = (
                f"name: {self.name}\n"
                f"created_at: {meta['created_at']}\n"
                f"source: {source}\n"
                f"notes: {notes}\n"
                f"schema: {meta['schema']}\n"
            )
            self.meta_path.write_text(yaml, encoding="utf-8")

        append_event(
            self.events_path, {"action": "init", "dataset": self.name, "source": source}
        )

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
                if c not in {k for k, _ in REQUIRED_COLUMNS}
                and c not in LEGACY_ALLOW
                and "__" in c
            }
        )

        # Extract embedded metadata if present
        md = tbl.schema.metadata or {}
        meta = {}
        for bkey, bval in md.items():
            k = bkey.decode("utf-8")
            if k.startswith("usr:"):
                try:
                    v = bval.decode("utf-8")
                    meta[k[4:]] = v
                except Exception:
                    pass

        return {
            "name": self.name,
            "path": str(self.records_path),
            "rows": tbl.num_rows,
            "columns": cols,
            "namespaces": namespaces,
            "meta": meta,
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

    def import_csv(
        self,
        csv_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        """Import sequences from CSV; trims, validates, computes ids, appends."""
        df = pd.read_csv(csv_path)
        n = self._import_df(
            df, default_bio_type, default_alphabet, source or str(csv_path)
        )
        append_event(
            self.events_path,
            {
                "action": "import_csv",
                "dataset": self.name,
                "path": str(csv_path),
                "n": n,
            },
        )
        return n

    def import_jsonl(
        self,
        jsonl_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        """Import sequences from JSONL (records)."""
        df = pd.read_json(jsonl_path, lines=True)
        n = self._import_df(
            df, default_bio_type, default_alphabet, source or str(jsonl_path)
        )
        append_event(
            self.events_path,
            {
                "action": "import_jsonl",
                "dataset": self.name,
                "path": str(jsonl_path),
                "n": n,
            },
        )
        return n

    def _import_df(
        self,
        df: pd.DataFrame,
        default_bio_type: str,
        default_alphabet: str,
        source: str,
    ) -> int:
        if "sequence" not in df.columns:
            raise SchemaError("Missing required column: sequence")
        bio = (
            df["bio_type"]
            if "bio_type" in df.columns
            else pd.Series([default_bio_type] * len(df))
        ).astype(str)
        alph = (
            df["alphabet"]
            if "alphabet" in df.columns
            else pd.Series([default_alphabet] * len(df))
        ).astype(str)

        ids, seqs, lens = [], [], []
        for s, bt, ab in zip(df["sequence"], bio, alph):
            s_norm = normalize_sequence(str(s), bt, ab)
            seqs.append(s_norm)
            ids.append(compute_id(bt, s_norm))
            lens.append(len(s_norm))

        # robust tz-aware timestamp for all pandas versions
        ts_utc = pd.Timestamp.now(tz="UTC")
        created = pd.Series([ts_utc] * len(df), dtype="datetime64[ns, UTC]")

        out_df = pd.DataFrame(
            {
                "id": ids,
                "bio_type": bio.tolist(),
                "sequence": seqs,
                "alphabet": alph.tolist(),
                "length": lens,
                "source": [source] * len(df),
                "created_at": created,
            }
        )
        incoming = pa.Table.from_pandas(
            out_df, schema=ARROW_SCHEMA, preserve_index=False
        )

        if len(ids) != len(set(ids)):
            raise DuplicateIDError(
                "Duplicate ids in incoming data (likely duplicate sequences)."
            )

        if self.records_path.exists():
            existing = read_parquet(self.records_path)
            existing_ids = set(existing.column("id").to_pylist())
            inter = existing_ids.intersection(set(ids))
            if inter:
                raise DuplicateIDError(
                    f"Duplicate ids already present in dataset: {len(inter)} conflict(s)."
                )
            combined = pa.concat_tables([existing, incoming], promote_options="default")
            # Preserve embedded metadata from the existing table
            write_parquet_atomic(
                combined,
                self.records_path,
                self.snapshot_dir,
                preserve_metadata_from=existing,
            )
        else:
            # First write; ensure metadata is embedded (init usually handles this)
            write_parquet_atomic(incoming, self.records_path, self.snapshot_dir)
        return len(ids)

    # ---- generic attach ----

    def attach(
        self,
        path: Path,
        namespace: str,
        *,
        id_col: str = "id",
        columns: Optional[Iterable[str]] = None,
        allow_overwrite: bool = False,
        note: str = "",
    ) -> int:
        """
        Attach arbitrary columns under a tool-safe namespace: `<namespace>__<column>`.

        Rules:
        - `namespace` must match `^[a-z][a-z0-9_]*$`
        - Essential columns are immutable
        - Overwriting existing columns requires `allow_overwrite=True`
        - Values align by `id`; unknown ids are ignored; missing values become NULL
        - Strings that look like JSON arrays/objects are parsed (generic vector support)

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
            inc = pd.read_json(path, lines=path.suffix.lower().endswith("jsonl"))
        else:
            raise SchemaError("Unsupported input format. Use parquet|csv|jsonl.")
        if id_col not in inc.columns:
            raise SchemaError(f"Missing id column '{id_col}' in incoming data.")

        # Choose + prefix targets
        attach_cols = (
            [c for c in inc.columns if c != id_col]
            if columns is None
            else list(columns)
        )
        if not attach_cols:
            return 0

        # Parse JSON arrays/objects if given as strings (robust to NaNs etc.)
        def _maybe_parse_json_value(x):
            if isinstance(x, str):
                s = x.strip()
                if s.startswith("[") or s.startswith("{"):
                    try:
                        return json.loads(s)
                    except Exception:
                        return x
            return x

        work = inc[[id_col] + attach_cols].copy()
        for c in attach_cols:
            work[c] = work[c].map(_maybe_parse_json_value)

        def target_name(c: str) -> str:
            return c if c.startswith(namespace + "__") else f"{namespace}__{c}"

        targets = [target_name(c) for c in attach_cols]
        work.columns = ["id"] + targets

        # Read existing once; check policy
        existing_tbl = read_parquet(self.records_path)
        existing_names = set(existing_tbl.schema.names)

        essential = {k for k, _ in REQUIRED_COLUMNS}
        clobbers = [c for c in targets if c in existing_names]
        if clobbers and not allow_overwrite:
            raise NamespaceError(
                f"Columns already exist: {', '.join(clobbers)}. Use --allow-overwrite to replace."
            )
        for t in targets:
            if t in essential:
                raise NamespaceError(f"Refusing to write essential column: {t}")
            if "__" not in t and t not in LEGACY_ALLOW:
                raise NamespaceError(f"Derived columns must be namespaced (got '{t}').")

        # Fast id index (no pandas round-trip)
        id_existing = existing_tbl.column("id").to_pylist()
        nrows = len(id_existing)
        pos = {rid: i for i, rid in enumerate(id_existing)}

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
            # list/struct or other complex: let Arrow accept cleaned Python values
            return pa.array(cleaned, type=target_type)

        def _infer_arrow(values: List) -> pa.Array:
            cleaned = [None if _is_nanlike(v) else v for v in values]
            if all(v is None for v in cleaned):
                return pa.array(cleaned, type=pa.string())
            if any(isinstance(v, str) for v in cleaned):
                return pa.array(
                    [None if v is None else str(v) for v in cleaned], type=pa.string()
                )
            return pa.array(cleaned)

        # id->value map per target (latest wins)
        id_series = work["id"].tolist()
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
                # Overwrite existing column
                field = new_tbl.schema.field(col)
                base_values = new_tbl.column(col).to_pylist()
                for rid, v in incoming_map.items():
                    idx = pos.get(str(rid))
                    if idx is not None:
                        base_values[idx] = v
                arr = _to_arrow_typed(base_values, field.type)
                col_idx = new_tbl.schema.get_field_index(col)
                new_tbl = new_tbl.set_column(
                    col_idx, pa.field(col, arr.type, nullable=True), arr
                )
            else:
                # Brand new column
                base_values = [None] * nrows
                for rid, v in incoming_map.items():
                    idx = pos.get(str(rid))
                    if idx is not None:
                        base_values[idx] = v
                arr = _infer_arrow(base_values)
                new_tbl = new_tbl.add_column(
                    new_tbl.num_columns, pa.field(col, arr.type, nullable=True), arr
                )

        # Atomic write, preserving embedded metadata
        write_parquet_atomic(
            new_tbl, self.records_path, self.snapshot_dir, preserve_metadata_from=existing_tbl
        )

        n_attached = int(work.shape[0])
        append_event(
            self.events_path,
            {
                "action": "attach",
                "dataset": self.name,
                "path": str(path),
                "namespace": namespace,
                "columns": targets,
                "allow_overwrite": allow_overwrite,
                "rows_seen": n_attached,
                "note": note,
            },
        )
        return n_attached

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

        # alphabet check for dna_4 (case-insensitive)
        if "dna_4" in set(tbl.column("alphabet").to_pylist()):
            seqs = tbl.column("sequence").to_pylist()
            bad = [s for s in seqs if s and re.search(r"[^ACGTacgt]", s)]
            if bad:
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
        df = read_parquet(
            self.records_path, columns=["id", "sequence", "length"]
        ).to_pandas()
        hits = df[
            df["sequence"].str.contains(pattern, case=False, regex=True, na=False)
        ]
        return hits.head(limit)

    def export(self, fmt: str, out_path: Path) -> None:
        """Export current table to CSV or JSONL."""
        self._require_exists()
        df = read_parquet(self.records_path).to_pandas()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out_path, index=False)
        elif fmt == "jsonl":
            df.to_json(out_path, orient="records", lines=True)
        else:
            raise SequencesError("Unsupported export format. Use csv|jsonl.")

    def snapshot(self) -> None:
        """Write a timestamped snapshot (per policy) and atomically persist current table."""
        self._require_exists()
        tbl = read_parquet(self.records_path)
        write_parquet_atomic(
            tbl, self.records_path, self.snapshot_dir, preserve_metadata_from=tbl
        )
        append_event(self.events_path, {"action": "snapshot", "dataset": self.name})
