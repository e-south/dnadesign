"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/mock.py

Mock dataset helpers for USR.

Two entry points:
- make-mock: create records.parquet with essential columns and demo vectors
- add-demo-cols: add/overwrite demo vector/label columns on an existing dataset

Demo columns (namespaced):
- <ns>__x_representation : list<float32>[x_dim] per row (default 512)
- <ns>__label_vec8       : list<float32>[y_dim] per row (default 8)

Usage (CLI):
  usr make-mock <dataset> [--n 100] [--length 60] [--x-dim 512] [--y-dim 8]
                     [--seed 7] [--namespace demo]
                     [--from-csv template_demo/template_sequences.csv]
  usr add-demo-cols <dataset> [--x-dim 512] [--y-dim 8] [--seed 7]
                        [--namespace demo] [--allow-overwrite]

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from .dataset import Dataset
from .io import append_event, read_parquet, write_parquet_atomic
from .normalize import compute_id, normalize_sequence
from .schema import ARROW_SCHEMA


@dataclass
class MockSpec:
    n: int = 100
    length: int = 60
    x_dim: int = 512
    y_dim: int = 8
    seed: int = 7
    namespace: str = "demo"  # derived columns will be <ns>__*
    csv_path: Optional[Path] = None  # if provided, sequences come from here


_DNA = np.array(list("ACGT"))


def _random_dna_sequences(n: int, length: int, rng: np.random.Generator) -> List[str]:
    ix = rng.integers(0, 4, size=(n, length), dtype=np.int8)
    return ["".join(_DNA[row].tolist()) for row in ix]


def _load_sequences_from_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"{path} must contain a 'sequence' column.")
    return [str(s) for s in df["sequence"].tolist()]


def make_mock_table(spec: MockSpec) -> pa.Table:
    rng = np.random.default_rng(spec.seed)

    # ----- Essential columns -----
    if spec.csv_path is not None:
        # Harmonize with template_demo: use exact sequences from CSV
        seqs_raw = _load_sequences_from_csv(Path(spec.csv_path))
        # If user passed --n, honor it by truncating; otherwise use all
        n_rows = spec.n if spec.n and spec.n <= len(seqs_raw) else len(seqs_raw)
        sequences = [s.strip() for s in seqs_raw[:n_rows]]
        source_str = f"make-mock from-csv:{Path(spec.csv_path).as_posix()}"
    else:
        n_rows = spec.n
        sequences = _random_dna_sequences(n_rows, spec.length, rng)
        source_str = f"make-mock (n={spec.n}, L={spec.length}, seed={spec.seed})"

    bio_type = ["dna"] * n_rows
    alphabet = ["dna_4"] * n_rows

    # Normalize, enforce alphabet, compute ids/lengths
    seqs_norm = [normalize_sequence(s, "dna", "dna_4") for s in sequences]
    ids = [compute_id("dna", s) for s in seqs_norm]
    lengths = [len(s) for s in seqs_norm]
    created_at = [datetime.now(timezone.utc) for _ in range(n_rows)]
    source = [source_str] * n_rows

    # ----- Derived demo columns -----
    x_vals = (rng.normal(loc=-20.0, scale=4.0, size=(n_rows, spec.x_dim))).astype(
        np.float32
    )
    y_vals = rng.random(size=(n_rows, spec.y_dim), dtype=np.float32)
    x_list = [row.tolist() for row in x_vals]
    y_list = [row.tolist() for row in y_vals]

    # Build Arrow table
    fields = list(ARROW_SCHEMA)
    arrays = [
        pa.array(ids, type=pa.string()),
        pa.array(bio_type, type=pa.string()),
        pa.array(seqs_norm, type=pa.string()),
        pa.array(alphabet, type=pa.string()),
        pa.array(lengths, type=pa.int32()),
        pa.array(source, type=pa.string()),
        pa.array(created_at, type=pa.timestamp("us", tz="UTC")),
    ]

    demo_x_name = f"{spec.namespace}__x_representation"
    demo_y_name = f"{spec.namespace}__label_vec8"

    demo_x_arr = pa.array(x_list, type=pa.list_(pa.float32()))
    demo_y_arr = pa.array(y_list, type=pa.list_(pa.float32()))

    schema = (
        pa.schema(fields)
        .append(pa.field(demo_x_name, demo_x_arr.type, nullable=True))
        .append(pa.field(demo_y_name, demo_y_arr.type, nullable=True))
    )
    arrays.extend([demo_x_arr, demo_y_arr])
    return pa.Table.from_arrays(arrays, schema=schema)


def create_mock_dataset(
    root: Path, dataset: str, spec: MockSpec, *, force: bool = False
) -> int:
    ds = Dataset(root, dataset)

    # Initialize folder if not present
    if not ds.records_path.exists():
        ds.init(source=f"make-mock (seed={spec.seed})")
    elif ds.records_path.exists() and not force:
        raise FileExistsError(
            f"Dataset '{dataset}' already exists at {ds.records_path}. Use --force to overwrite."
        )

    tbl = make_mock_table(spec)
    write_parquet_atomic(tbl, ds.records_path, ds.snapshot_dir)

    append_event(
        ds.events_path,
        {
            "action": "make_mock",
            "dataset": ds.name,
            "n": tbl.num_rows,
            "length": (
                int(pa.array(tbl.column("length")).to_pylist()[0])
                if tbl.num_rows
                else 0
            ),
            "x_dim": spec.x_dim,
            "y_dim": spec.y_dim,
            "seed": spec.seed,
            "namespace": spec.namespace,
            "from_csv": bool(spec.csv_path is not None),
            "csv_path": str(spec.csv_path) if spec.csv_path else "",
        },
    )
    return tbl.num_rows


def add_demo_columns(
    root: Path,
    dataset: str,
    *,
    x_dim: int = 512,
    y_dim: int = 8,
    seed: int = 7,
    namespace: str = "demo",
    allow_overwrite: bool = False,
) -> int:
    """
    Add (or overwrite with --allow-overwrite) demo vector/label columns for ALL rows.
    Returns number of rows updated.
    """
    ds = Dataset(root, dataset)
    if not ds.records_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds.records_path}")

    tbl = read_parquet(ds.records_path)
    n = tbl.num_rows
    if n == 0:
        # no-op but still safe
        return 0

    rng = np.random.default_rng(int(seed))
    x = (rng.normal(loc=-20.0, scale=4.0, size=(n, int(x_dim)))).astype(np.float32)
    y = rng.random(size=(n, int(y_dim)), dtype=np.float32)
    x_arr = pa.array([row.tolist() for row in x], type=pa.list_(pa.float32()))
    y_arr = pa.array([row.tolist() for row in y], type=pa.list_(pa.float32()))

    x_name = f"{namespace}__x_representation"
    y_name = f"{namespace}__label_vec8"

    existing_names = set(tbl.schema.names)
    if not allow_overwrite:
        clobbers = [c for c in (x_name, y_name) if c in existing_names]
        if clobbers:
            raise FileExistsError(
                f"Columns already exist: {', '.join(clobbers)}. Use --allow-overwrite to replace."
            )

    new_tbl = tbl
    # set/replace X
    if x_name in existing_names:
        idx = new_tbl.schema.get_field_index(x_name)
        new_tbl = new_tbl.set_column(idx, pa.field(x_name, x_arr.type, True), x_arr)
    else:
        new_tbl = new_tbl.append_column(pa.field(x_name, x_arr.type, True), x_arr)
    # set/replace Y
    if y_name in existing_names:
        idx = new_tbl.schema.get_field_index(y_name)
        new_tbl = new_tbl.set_column(idx, pa.field(y_name, y_arr.type, True), y_arr)
    else:
        new_tbl = new_tbl.append_column(pa.field(y_name, y_arr.type, True), y_arr)

    write_parquet_atomic(
        new_tbl, ds.records_path, ds.snapshot_dir, preserve_metadata_from=tbl
    )

    append_event(
        ds.events_path,
        {
            "action": "add_demo_cols",
            "dataset": ds.name,
            "rows": n,
            "x_dim": int(x_dim),
            "y_dim": int(y_dim),
            "seed": int(seed),
            "namespace": namespace,
            "overwrote": bool(any(c in existing_names for c in (x_name, y_name))),
        },
    )
    return n
