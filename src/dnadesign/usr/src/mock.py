"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/mock.py

Mock dataset generation helpers for USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import Dataset
from .events import record_event
from .io import read_parquet, write_parquet_atomic
from .locks import dataset_write_lock
from .normalize import compute_id, normalize_sequence
from .overlays import overlay_metadata, overlay_path, with_overlay_metadata
from .registry import registry_hash
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


def make_mock_tables(spec: MockSpec) -> tuple[pa.Table, pa.Table]:
    rng = np.random.default_rng(spec.seed)

    # ----- Essential columns -----
    if spec.csv_path is not None:
        # Harmonize with demo_material: use exact sequences from CSV
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
    x_vals = (rng.normal(loc=-20.0, scale=4.0, size=(n_rows, spec.x_dim))).astype(np.float32)
    y_vals = rng.random(size=(n_rows, spec.y_dim), dtype=np.float32)
    x_list = [row.tolist() for row in x_vals]
    y_list = [row.tolist() for row in y_vals]

    # Build base table
    base_arrays = [
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

    base_tbl = pa.Table.from_arrays(base_arrays, schema=ARROW_SCHEMA)
    overlay_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field(demo_x_name, demo_x_arr.type, nullable=True),
            pa.field(demo_y_name, demo_y_arr.type, nullable=True),
        ]
    )
    overlay_tbl = pa.Table.from_arrays([pa.array(ids), demo_x_arr, demo_y_arr], schema=overlay_schema)
    return base_tbl, overlay_tbl


def create_mock_dataset(root: Path, dataset: str, spec: MockSpec, *, force: bool = False) -> int:
    ds = Dataset(root, dataset)

    # Initialize folder if not present
    if not ds.records_path.exists():
        ds.init(source=f"make-mock (seed={spec.seed})")
    elif ds.records_path.exists() and not force:
        raise FileExistsError(f"Dataset '{dataset}' already exists at {ds.records_path}. Use --force to overwrite.")

    with dataset_write_lock(ds.dir):
        base_tbl, overlay_tbl = make_mock_tables(spec)
        write_parquet_atomic(base_tbl, ds.records_path, ds.snapshot_dir)

        ds._validate_registry_schema(namespace=spec.namespace, schema=overlay_tbl.schema, key="id")
        reg_hash = registry_hash(ds.root, required=True)
        overlay_tbl = with_overlay_metadata(
            overlay_tbl,
            namespace=spec.namespace,
            key="id",
            created_at=datetime.now(timezone.utc).isoformat(),
            registry_hash=reg_hash,
        )
        out_path = overlay_path(ds.dir, spec.namespace)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        pq.write_table(overlay_tbl, tmp)
        os.replace(tmp, out_path)

        record_event(
            ds.events_path,
            "make_mock",
            dataset=ds.name,
            args={
                "n": base_tbl.num_rows,
                "length": (int(pa.array(base_tbl.column("length")).to_pylist()[0]) if base_tbl.num_rows else 0),
                "x_dim": spec.x_dim,
                "y_dim": spec.y_dim,
                "seed": spec.seed,
                "namespace": spec.namespace,
                "from_csv": bool(spec.csv_path is not None),
                "csv_path": str(spec.csv_path) if spec.csv_path else "",
            },
            target_path=ds.records_path,
            dataset_root=ds.root,
        )
        return base_tbl.num_rows


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

    with dataset_write_lock(ds.dir):
        overlay_tbl = pa.Table.from_arrays(
            [tbl.column("id"), x_arr, y_arr],
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field(x_name, x_arr.type, nullable=True),
                    pa.field(y_name, y_arr.type, nullable=True),
                ]
            ),
        )
        ds._validate_registry_schema(namespace=namespace, schema=overlay_tbl.schema, key="id")
        reg_hash = registry_hash(ds.root, required=True)
        overlay_tbl = with_overlay_metadata(
            overlay_tbl,
            namespace=namespace,
            key="id",
            created_at=datetime.now(timezone.utc).isoformat(),
            registry_hash=reg_hash,
        )

        out_path = overlay_path(ds.dir, namespace)
        overwrote = out_path.exists()
        if out_path.exists():
            meta = overlay_metadata(out_path)
            existing_key = meta.get("key")
            if existing_key and existing_key != "id":
                raise FileExistsError(f"Overlay key mismatch (expected id, got {existing_key}).")
            existing_tbl = pq.read_table(out_path)
            existing_cols = set(existing_tbl.schema.names)
            clobbers = [c for c in (x_name, y_name) if c in existing_cols]
            if clobbers and not allow_overwrite:
                raise FileExistsError(
                    f"Columns already exist: {', '.join(clobbers)}. Use --allow-overwrite to replace."
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        pq.write_table(overlay_tbl, tmp)
        os.replace(tmp, out_path)

        record_event(
            ds.events_path,
            "add_demo_cols",
            dataset=ds.name,
            args={
                "rows": n,
                "x_dim": int(x_dim),
                "y_dim": int(y_dim),
                "seed": int(seed),
                "namespace": namespace,
                "overwrote": bool(overwrote),
            },
            target_path=out_path,
            dataset_root=ds.root,
        )
        return n
