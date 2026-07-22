"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/outputs/loader.py

Loads DenseGen output records for reporting and analysis.

Module Author(s): Eric J. South
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

from ...config import RootConfig, resolve_outputs_scoped_path, resolve_run_root, resolve_usr_root_scoped_path
from ...core.record_metadata_recovery import recover_densegen_metadata_from_source
from ...core.record_values import coerce_list_of_dicts
from .base import DEFAULT_NAMESPACE
from .parquet import is_legacy_used_tfbs_detail_schema_compatible, validate_parquet_schema

log = logging.getLogger(__name__)
DEFAULT_RECORD_LOAD_LIMIT = 100_000
_RECOVERABLE_METADATA_COLUMNS = {"densegen__plan", "densegen__input_name"}


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


def _materialize_batches_to_dataframe(
    batches: Iterable[Any],
    *,
    row_limit: int | None,
    parse_json_in_namespaced_columns: bool,
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    limit = None if row_limit is None else max(1, int(row_limit))
    collected: list[Any] = []
    rows = 0
    for batch in batches:
        batch_rows = int(getattr(batch, "num_rows", 0) or 0)
        if batch_rows <= 0:
            continue
        if limit is not None:
            remaining = limit - rows
            if remaining <= 0:
                break
            if batch_rows > remaining:
                collected.append(batch.slice(0, remaining))
                rows += remaining
                break
        collected.append(batch)
        rows += batch_rows
    if not collected:
        return pd.DataFrame()

    frame = pa.Table.from_batches(collected).to_pandas()
    for name in frame.columns:
        values = frame[name].tolist()
        if not any(isinstance(value, np.ndarray) for value in values):
            continue
        frame[name] = [value.tolist() if isinstance(value, np.ndarray) else value for value in values]
    if not parse_json_in_namespaced_columns or frame.empty:
        return frame
    for name in [column for column in frame.columns if "__" in str(column)]:
        values = frame[name].tolist()
        if not any(isinstance(value, str) for value in values):
            continue
        frame[name] = [_maybe_json_load(value) for value in values]
    return frame


def _iter_record_batches_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    columns: Iterable[str] | None = None,
    *,
    batch_size: int = 65536,
) -> tuple[Iterable[Any], str, bool]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out_cfg = root_cfg.densegen.output
    source, run_root = _resolve_source(root_cfg, cfg_path)
    requested = list(columns) if columns else None

    if source == "usr":
        usr_cfg = out_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when source='usr'")
        root = resolve_usr_root_scoped_path(
            cfg_path,
            usr_cfg.root,
            label="output.usr.root",
            scope=usr_cfg.root_scope,
        )
        try:
            from dnadesign.usr import Dataset
        except Exception as e:
            raise RuntimeError(f"USR support is not available: {e}") from e

        ds = Dataset(root, usr_cfg.dataset)
        rp = ds.records_path
        if not rp.exists():
            raise FileNotFoundError(f"USR records not found at: {rp}")
        return (
            ds.scan(
                columns=requested,
                include_overlays=True,
                include_deleted=False,
                batch_size=int(batch_size),
            ),
            f"usr:{usr_cfg.dataset}",
            True,
        )

    if source == "parquet":
        pq_cfg = out_cfg.parquet
        if pq_cfg is None:
            raise ValueError("output.parquet is required when source='parquet'")
        root = resolve_outputs_scoped_path(cfg_path, run_root, pq_cfg.path, label="output.parquet.path")
        if root.exists() and root.is_dir():
            raise ValueError(f"Parquet path must be a file, got directory: {root}")

        if root.exists():
            import pyarrow.parquet as pq

            try:
                validate_parquet_schema(root, namespace=DEFAULT_NAMESPACE)
            except RuntimeError:
                if not is_legacy_used_tfbs_detail_schema_compatible(root, namespace=DEFAULT_NAMESPACE):
                    raise
                log.warning(
                    "Loading legacy DenseGen parquet schema for compatibility: %s. "
                    "Only densegen__used_tfbs_detail uses the legacy field layout.",
                    root,
                )
            with _suppressed_pyarrow_sysctl_warnings():
                pf = pq.ParquetFile(root)
            if pf.metadata is not None and pf.metadata.num_rows == 0:
                raise RuntimeError(f"Parquet output has no rows: {root}")
            with _suppressed_pyarrow_sysctl_warnings():
                return (
                    pf.iter_batches(batch_size=int(batch_size), columns=requested),
                    f"parquet:{root}",
                    False,
                )

        parts = sorted(root.parent.glob(f"{root.stem}__part-*.parquet"))
        if parts:
            import pyarrow.dataset as ds

            with _suppressed_pyarrow_sysctl_warnings():
                dataset = ds.dataset([str(p) for p in parts], format="parquet")
            with _suppressed_pyarrow_sysctl_warnings():
                scanner = ds.Scanner.from_dataset(dataset, columns=requested, batch_size=int(batch_size))
            return (
                scanner.to_batches(),
                f"parquet:{root} (parts)",
                False,
            )

        raise FileNotFoundError(f"Parquet output not found: {root}")

    raise ValueError(f"Unknown plot source: {source}")


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
        root = resolve_usr_root_scoped_path(
            cfg_path,
            usr_cfg.root,
            label="output.usr.root",
            scope=usr_cfg.root_scope,
        )
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

            try:
                validate_parquet_schema(root, namespace=DEFAULT_NAMESPACE)
            except RuntimeError:
                if not is_legacy_used_tfbs_detail_schema_compatible(root, namespace=DEFAULT_NAMESPACE):
                    raise
                log.warning(
                    "Loading legacy DenseGen parquet schema for compatibility: %s. "
                    "Only densegen__used_tfbs_detail uses the legacy field layout.",
                    root,
                )
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
    normalize_used_tfbs_detail: bool = True,
) -> Tuple["pd.DataFrame", str]:
    """
    Load output records based on output.targets and plots.source (when multiple sinks).
    Returns (df, source_label), where source_label is 'parquet:<path>' or 'usr:<dataset>'.
    """
    resolved_max_rows = int(max_rows) if max_rows is not None else int(DEFAULT_RECORD_LOAD_LIMIT)
    if resolved_max_rows < 1:
        raise ValueError("max_rows must be >= 1 when loading output records")
    requested_columns = list(columns) if columns is not None else None
    scan_columns = list(requested_columns) if requested_columns is not None else None
    source_added_for_recovery = False
    if (
        scan_columns is not None
        and "source" not in scan_columns
        and _RECOVERABLE_METADATA_COLUMNS.intersection(scan_columns)
    ):
        scan_columns = [*scan_columns, "source"]
        source_added_for_recovery = True

    batches, source_label, parse_json_in_namespaced_columns = _iter_record_batches_from_config(
        root_cfg,
        cfg_path,
        columns=scan_columns,
    )
    df = _materialize_batches_to_dataframe(
        batches,
        row_limit=resolved_max_rows + 1,
        parse_json_in_namespaced_columns=parse_json_in_namespaced_columns,
    )
    if df.empty:
        raise RuntimeError("Output records could not be materialized into a dataframe.")
    truncated = len(df) > resolved_max_rows
    if truncated:
        df = df.iloc[:resolved_max_rows].reset_index(drop=True)
    if normalize_used_tfbs_detail and "densegen__used_tfbs_detail" in df.columns:
        df["densegen__used_tfbs_detail"] = [
            coerce_list_of_dicts(value) for value in df["densegen__used_tfbs_detail"].tolist()
        ]
    df = recover_densegen_metadata_from_source(df)
    if source_added_for_recovery and requested_columns is not None and "source" not in requested_columns:
        df = df.drop(columns=["source"], errors="ignore")

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
