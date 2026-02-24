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

import contextlib
import json
import logging
import warnings
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Tuple

if TYPE_CHECKING:
    import pandas as pd

from ...config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from .base import DEFAULT_NAMESPACE
from .parquet import validate_parquet_schema

log = logging.getLogger(__name__)
DEFAULT_RECORD_LOAD_LIMIT = 100_000


@contextlib.contextmanager
def _suppressed_pyarrow_sysctl_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sysctlbyname.*", category=UserWarning)
        yield


def _maybe_json_load(val):
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                raise ValueError(f"Failed to parse JSON field: {s[:80]}")
    return val


def _resolve_source(root_cfg: RootConfig, cfg_path: Path) -> tuple[str, Path]:
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
    return source, run_root


def _iter_record_dicts_from_batches(
    batches: Iterable[Any],
    *,
    parse_json_in_namespaced_columns: bool,
) -> Iterator[dict[str, Any]]:
    for batch in batches:
        names = list(batch.schema.names)
        values = {name: batch.column(i).to_pylist() for i, name in enumerate(names)}
        for row_idx in range(batch.num_rows):
            row: dict[str, Any] = {}
            for name in names:
                val = values[name][row_idx]
                if parse_json_in_namespaced_columns and "__" in name:
                    val = _maybe_json_load(val)
                row[name] = val
            yield row


def _limit_rows(rows: Iterable[dict[str, Any]], *, max_rows: int | None) -> Iterator[dict[str, Any]]:
    if max_rows is None:
        yield from rows
        return
    limit = int(max_rows)
    if limit <= 0:
        return
    for idx, row in enumerate(rows):
        if idx >= limit:
            return
        yield row


def _require_non_empty_rows(
    rows: Iterable[dict[str, Any]],
    *,
    empty_error: str,
) -> Iterator[dict[str, Any]]:
    it = iter(rows)
    try:
        first = next(it)
    except StopIteration as exc:
        raise RuntimeError(empty_error) from exc
    return chain([first], it)


def scan_records_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    columns: Iterable[str] | None = None,
    *,
    max_rows: int | None = None,
    batch_size: int = 65536,
) -> Tuple[Iterable[dict[str, Any]], str]:
    """
    Stream output records based on output.targets and plots.source (when multiple sinks).
    Returns (rows, source_label), where rows yields dict records.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out_cfg = root_cfg.densegen.output
    source, run_root = _resolve_source(root_cfg, cfg_path)
    requested = list(columns) if columns else None

    if source == "usr":
        usr_cfg = out_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when source='usr'")
        root = resolve_outputs_scoped_path(cfg_path, run_root, usr_cfg.root, label="output.usr.root")
        try:
            from dnadesign.usr import Dataset
        except Exception as e:
            raise RuntimeError(f"USR support is not available: {e}") from e

        ds = Dataset(root, usr_cfg.dataset)
        rp = ds.records_path
        if not rp.exists():
            raise FileNotFoundError(f"USR records not found at: {rp}")
        rows = _iter_record_dicts_from_batches(
            ds.scan(
                columns=requested,
                include_overlays=True,
                include_deleted=False,
                batch_size=int(batch_size),
            ),
            parse_json_in_namespaced_columns=True,
        )
        rows = _limit_rows(rows, max_rows=max_rows)
        rows = _require_non_empty_rows(rows, empty_error=f"USR output has no rows: {rp}")
        return rows, f"usr:{usr_cfg.dataset}"

    if source == "parquet":
        pq_cfg = out_cfg.parquet
        if pq_cfg is None:
            raise ValueError("output.parquet is required when source='parquet'")
        root = resolve_outputs_scoped_path(cfg_path, run_root, pq_cfg.path, label="output.parquet.path")
        if root.exists() and root.is_dir():
            raise ValueError(f"Parquet path must be a file, got directory: {root}")

        if root.exists():
            import pyarrow.parquet as pq

            validate_parquet_schema(root, namespace=DEFAULT_NAMESPACE)
            with _suppressed_pyarrow_sysctl_warnings():
                pf = pq.ParquetFile(root)
            if pf.metadata is not None and pf.metadata.num_rows == 0:
                raise RuntimeError(f"Parquet output has no rows: {root}")
            with _suppressed_pyarrow_sysctl_warnings():
                rows = _iter_record_dicts_from_batches(
                    pf.iter_batches(batch_size=int(batch_size), columns=requested),
                    parse_json_in_namespaced_columns=False,
                )
            rows = _limit_rows(rows, max_rows=max_rows)
            rows = _require_non_empty_rows(rows, empty_error=f"Parquet output has no rows: {root}")
            return rows, f"parquet:{root}"

        parts = sorted(root.parent.glob(f"{root.stem}__part-*.parquet"))
        if parts:
            import pyarrow.dataset as ds

            with _suppressed_pyarrow_sysctl_warnings():
                dataset = ds.dataset([str(p) for p in parts], format="parquet")
            if dataset.count_rows() == 0:
                raise RuntimeError(f"Parquet parts have no rows: {root.parent}")
            with _suppressed_pyarrow_sysctl_warnings():
                scanner = ds.Scanner.from_dataset(dataset, columns=requested, batch_size=int(batch_size))
            rows = _iter_record_dicts_from_batches(
                scanner.to_batches(),
                parse_json_in_namespaced_columns=False,
            )
            rows = _limit_rows(rows, max_rows=max_rows)
            rows = _require_non_empty_rows(rows, empty_error=f"Parquet parts have no rows: {root.parent}")
            return rows, f"parquet:{root} (parts)"

        raise FileNotFoundError(f"Parquet output not found: {root}")

    raise ValueError(f"Unknown plot source: {source}")


def load_records_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    columns: Iterable[str] | None = None,
    *,
    max_rows: int | None = None,
    allow_truncated: bool = False,
) -> Tuple["pd.DataFrame", str]:
    """
    Load output records based on output.targets and plots.source (when multiple sinks).
    Returns (df, source_label), where source_label is 'parquet:<path>' or 'usr:<dataset>'.
    """
    import pandas as pd

    resolved_max_rows = int(max_rows) if max_rows is not None else int(DEFAULT_RECORD_LOAD_LIMIT)
    if resolved_max_rows < 1:
        raise ValueError("max_rows must be >= 1 when loading output records")

    rows, source_label = scan_records_from_config(
        root_cfg,
        cfg_path,
        columns=columns,
        max_rows=resolved_max_rows + 1,
    )
    materialized_rows: list[dict[str, Any]] = []
    truncated = False
    for row in rows:
        if len(materialized_rows) >= resolved_max_rows:
            truncated = True
            break
        materialized_rows.append(row)

    if not materialized_rows:
        raise RuntimeError("Output records could not be materialized into a dataframe.")

    df = pd.DataFrame.from_records(materialized_rows)
    if df.empty:
        raise RuntimeError("Output records dataframe is empty after materialization.")

    if truncated:
        message = (
            "Output records rows were truncated to "
            f"{resolved_max_rows} (source={source_label}). "
            "Increase plots.sample_rows or pass allow_truncated=True to proceed with sampled rows."
        )
        if not allow_truncated:
            raise RuntimeError(message)
        log.warning(message)
    return df, source_label
