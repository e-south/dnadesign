"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/loader.py

Load output records for plotting/analysis.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Tuple

if TYPE_CHECKING:
    import pandas as pd

from ...config import RootConfig, resolve_run_root, resolve_run_scoped_path
from .base import DEFAULT_NAMESPACE
from .parquet import validate_parquet_schema


def _maybe_json_load(val):
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                raise ValueError(f"Failed to parse JSON field: {s[:80]}")
    return val


def load_records_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    columns: Iterable[str] | None = None,
    *,
    max_rows: int | None = None,
) -> Tuple["pd.DataFrame", str]:
    """
    Load output records based on output.targets and plots.source (when multiple sinks).
    Returns (df, source_label), where source_label is 'parquet:<path>' or 'usr:<dataset>'.
    """
    out_cfg = root_cfg.densegen.output
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    targets = out_cfg.targets

    if len(targets) > 1:
        plots = root_cfg.plots
        if plots is None or plots.source is None:
            raise ValueError("plots.source must be set when output.targets has multiple sinks")
        source = plots.source
    else:
        source = targets[0]

    if source == "usr":
        usr_cfg = out_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when source='usr'")
        root = resolve_run_scoped_path(cfg_path, run_root, usr_cfg.root, label="output.usr.root")
        try:
            from dnadesign.usr.src.dataset import Dataset
        except Exception as e:
            raise RuntimeError(f"USR support is not available: {e}") from e

        ds = Dataset(root, usr_cfg.dataset)
        rp = ds.records_path
        if not rp.exists():
            raise FileNotFoundError(f"USR records not found at: {rp}")
        import pyarrow.parquet as pq

        tbl = pq.read_table(rp, columns=list(columns) if columns else None)
        if max_rows is not None and tbl.num_rows > max_rows:
            tbl = tbl.slice(0, max_rows)
        df = tbl.to_pandas()
        for col in [c for c in df.columns if "__" in c]:
            df[col] = df[col].map(_maybe_json_load)
        return df, f"usr:{usr_cfg.dataset}"

    if source == "parquet":
        pq_cfg = out_cfg.parquet
        if pq_cfg is None:
            raise ValueError("output.parquet is required when source='parquet'")
        root = resolve_run_scoped_path(cfg_path, run_root, pq_cfg.path, label="output.parquet.path")
        if not root.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {root}")
        if root.is_file():
            raise ValueError(f"Parquet path must be a directory, got file: {root}")
        if not any(root.glob("*.parquet")):
            raise RuntimeError(f"Parquet dataset has no files: {root}")
        import pyarrow.dataset as ds

        validate_parquet_schema(root, namespace=DEFAULT_NAMESPACE)
        dataset = ds.dataset(root, format="parquet")
        if dataset.count_rows() == 0:
            raise RuntimeError(f"Parquet dataset has no rows: {root}")
        if max_rows is not None:
            tbl = dataset.head(max_rows, columns=list(columns) if columns else None)
        else:
            scanner = ds.Scanner.from_dataset(dataset, columns=list(columns) if columns else None)
            tbl = scanner.to_table()
        df = tbl.to_pandas()
        return df, f"parquet:{root}"

    raise ValueError(f"Unknown plot source: {source}")
