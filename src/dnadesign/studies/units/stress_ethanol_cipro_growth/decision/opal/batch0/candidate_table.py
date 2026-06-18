"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/candidate_table.py

Candidate feature-table contract for the stress/ethanol/ciprofloxacin OPAL handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.opal import validate_x_parquet_column

EXPECTED_OPAL_CANDIDATE_ROLE = "opal_candidate_feature_table"
REQUIRED_OPAL_COLUMNS: tuple[str, ...] = ("id", "bio_type", "sequence", "alphabet")
VIEW_PROVENANCE_COLUMNS: tuple[str, ...] = ("source_class", "design_family")
DENSEGEN_KEY_COLUMNS: tuple[str, ...] = (
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
)
REQUIRED_CANDIDATE_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "opal_candidate__role",
    "opal_candidate__x_source_view_id",
    "opal_candidate__source_class",
    "opal_candidate__design_family",
    "opal_candidate__sfxi_ref__collection_id",
    *DENSEGEN_KEY_COLUMNS,
)
REQUIRED_NON_NULL_CANDIDATE_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "opal_candidate__role",
    "opal_candidate__x_source_view_id",
    "opal_candidate__source_class",
    "opal_candidate__design_family",
    *DENSEGEN_KEY_COLUMNS,
)


def _repo_root_from(path: Path) -> Path:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not resolve repo root from {path}")


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _normal_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _blank_text_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.isna() | text.eq("")


def _read_required_parquet(path: str | Path, *, label: str, columns: list[str] | None = None) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise ValueError(f"{label} not found: {parquet_path}")
    try:
        return pd.read_parquet(parquet_path, columns=columns)
    except Exception as exc:
        raise ValueError(f"failed to read {label} at {parquet_path}: {exc}") from exc


def _candidate_table_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate_table = config.get("candidate_feature_table")
    if not isinstance(candidate_table, Mapping):
        raise ValueError("sampling config is missing candidate_feature_table")
    return candidate_table


def _x_source_config(candidate_table: Mapping[str, Any]) -> Mapping[str, Any]:
    x_source = candidate_table.get("x_source")
    if not isinstance(x_source, Mapping):
        raise ValueError("candidate_feature_table is missing x_source mapping")
    return x_source


def _materialization_config(candidate_table: Mapping[str, Any]) -> Mapping[str, Any]:
    materialization = candidate_table.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ValueError("candidate_feature_table is missing materialization mapping")
    return materialization


def _load_sampling_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"sampling config must be a mapping: {path}")
    return loaded


def _validate_view_ordered_subset(
    *,
    record_ids: list[str],
    view_rows_path: Path,
    view_row_id_column: str,
) -> None:
    view_rows = _read_required_parquet(view_rows_path, label="LatentDNA view rows", columns=[view_row_id_column])
    if view_row_id_column not in view_rows.columns:
        raise ValueError(f"LatentDNA view rows missing id column {view_row_id_column!r}")
    view_ids = view_rows[view_row_id_column].astype(str).tolist()
    if len(view_ids) != len(set(view_ids)):
        raise ValueError(f"LatentDNA view rows id column {view_row_id_column!r} contains duplicates")
    view_index = {row_id: idx for idx, row_id in enumerate(view_ids)}
    missing = [row_id for row_id in record_ids if row_id not in view_index]
    if missing:
        raise ValueError(f"candidate feature table ids are missing from LatentDNA view rows (sample={missing[:5]})")
    positions = [view_index[row_id] for row_id in record_ids]
    if positions != sorted(positions):
        raise ValueError("candidate feature table ids do not align with LatentDNA view rows")


def _validate_required_opal_values(records: pd.DataFrame, *, label: str) -> None:
    missing_columns = [column for column in REQUIRED_OPAL_COLUMNS if column not in records.columns]
    if missing_columns:
        raise ValueError(f"{label} missing required OPAL columns: {missing_columns}")
    for column in REQUIRED_OPAL_COLUMNS:
        missing_mask = _blank_text_mask(records[column])
        if missing_mask.any():
            if column == "id":
                sample = records.index[missing_mask].tolist()[:5]
                raise ValueError(f"{label} required column 'id' has null/blank values (row_indexes={sample})")
            sample_ids = records.loc[missing_mask, "id"].astype(str).tolist()[:5]
            raise ValueError(f"{label} required column {column!r} has null/blank values (sample_ids={sample_ids})")


def _allowed_value_set(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    allowed = {_normal_text(value) for value in values if _normal_text(value)}
    if not allowed:
        raise ValueError("candidate feature table allowed provenance values must not be empty")
    return allowed


def _validate_column_allowed_values(records: pd.DataFrame, *, column: str, allowed_values: set[str] | None) -> None:
    if allowed_values is None:
        return
    values = records[column].astype("string").str.strip()
    bad_mask = ~values.isin(sorted(allowed_values))
    if bad_mask.any():
        sample_values = sorted({str(value) for value in values[bad_mask].dropna().head(5).tolist()})
        sample_ids = records.loc[bad_mask, "id"].astype(str).tolist()[:5]
        raise ValueError(
            f"candidate feature table provenance column {column!r} contains values outside "
            f"{sorted(allowed_values)} (sample_values={sample_values}, sample_ids={sample_ids})"
        )


def _validate_required_null_columns(records: pd.DataFrame, *, columns: Sequence[str]) -> None:
    for column in columns:
        non_null_mask = ~_blank_text_mask(records[column])
        if non_null_mask.any():
            sample_ids = records.loc[non_null_mask, "id"].astype(str).tolist()[:5]
            raise ValueError(
                f"candidate feature table provenance column {column!r} must be null/blank "
                f"for the OPAL candidate universe (sample_ids={sample_ids})"
            )


def _validate_candidate_provenance_values(
    records: pd.DataFrame,
    *,
    allowed_source_classes: Sequence[str] | None = None,
    allowed_design_families: Sequence[str] | None = None,
    required_null_provenance_columns: Sequence[str] = (),
) -> None:
    for column in REQUIRED_NON_NULL_CANDIDATE_PROVENANCE_COLUMNS:
        missing_mask = _blank_text_mask(records[column])
        if missing_mask.any():
            sample_ids = records.loc[missing_mask, "id"].astype(str).tolist()[:5]
            raise ValueError(
                f"candidate feature table provenance column {column!r} has null/blank values (sample_ids={sample_ids})"
            )

    bad_role = records["opal_candidate__role"].astype(str) != EXPECTED_OPAL_CANDIDATE_ROLE
    if bad_role.any():
        sample_ids = records.loc[bad_role, "id"].astype(str).tolist()[:5]
        raise ValueError(
            "candidate feature table provenance column 'opal_candidate__role' must be "
            f"'{EXPECTED_OPAL_CANDIDATE_ROLE}' (sample_ids={sample_ids})"
        )
    _validate_column_allowed_values(
        records,
        column="opal_candidate__source_class",
        allowed_values=_allowed_value_set(allowed_source_classes),
    )
    _validate_column_allowed_values(
        records,
        column="opal_candidate__design_family",
        allowed_values=_allowed_value_set(allowed_design_families),
    )
    _validate_required_null_columns(records, columns=required_null_provenance_columns)


def _validate_required_opal_table_values(table: pa.Table, *, label: str) -> None:
    missing = [column for column in REQUIRED_OPAL_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"{label} missing required OPAL columns: {missing}")
    _validate_required_opal_values(table.select(REQUIRED_OPAL_COLUMNS).to_pandas(), label=label)


def validate_candidate_feature_table(
    *,
    records_path: str | Path,
    x_column: str,
    expected_rows: int | None = None,
    allowed_source_classes: Sequence[str] | None = None,
    allowed_design_families: Sequence[str] | None = None,
    required_null_provenance_columns: Sequence[str] = (),
    view_rows_path: str | Path | None = None,
    view_row_id_column: str = "construct__anchor_id",
) -> dict[str, int]:
    """Validate the OPAL candidate feature table contract."""

    if expected_rows is not None and int(expected_rows) <= 0:
        raise ValueError("candidate feature table expected_rows must be a positive integer")

    parquet_path = Path(records_path)
    if not parquet_path.exists():
        raise ValueError(f"candidate feature table records_path not found: {parquet_path}")
    try:
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    except Exception as exc:
        raise ValueError(f"failed to read candidate feature table schema at {parquet_path}: {exc}") from exc

    required_null_columns = tuple(dict.fromkeys(str(column) for column in required_null_provenance_columns))
    required_columns = (
        *REQUIRED_OPAL_COLUMNS,
        *REQUIRED_CANDIDATE_PROVENANCE_COLUMNS,
        *required_null_columns,
        x_column,
    )
    missing = [column for column in required_columns if column not in schema_names]
    if missing:
        raise ValueError(f"candidate feature table missing required columns: {missing}")

    records_columns = list(
        dict.fromkeys((*REQUIRED_OPAL_COLUMNS, *REQUIRED_CANDIDATE_PROVENANCE_COLUMNS, *required_null_columns))
    )
    records = _read_required_parquet(
        parquet_path,
        label="candidate feature table records_path",
        columns=records_columns,
    )
    if expected_rows is not None and int(len(records)) != int(expected_rows):
        raise ValueError(
            f"candidate feature table row count {len(records)} does not equal expected {int(expected_rows)}"
        )
    _validate_required_opal_values(records, label="candidate feature table")
    _validate_candidate_provenance_values(
        records,
        allowed_source_classes=allowed_source_classes,
        allowed_design_families=allowed_design_families,
        required_null_provenance_columns=required_null_columns,
    )
    ids = records["id"].astype(str)
    if ids.duplicated().any():
        sample = ids[ids.duplicated()].unique().tolist()[:5]
        raise ValueError(f"candidate feature table ids must be unique; duplicates={sample}")

    try:
        x_report = validate_x_parquet_column(parquet_path, x_column=x_column, id_column="id")
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if int(x_report.row_count) != int(len(records)):
        raise ValueError(
            f"candidate feature table X row count does not match records: x={x_report.row_count} records={len(records)}"
        )

    if view_rows_path is not None:
        _validate_view_ordered_subset(
            record_ids=ids.tolist(),
            view_rows_path=Path(view_rows_path),
            view_row_id_column=view_row_id_column,
        )

    return {"row_count": int(len(records)), "x_dim": int(x_report.x_dim)}


def _configured_records_path(config: Mapping[str, Any], *, repo_root: str | Path) -> Path:
    candidate_table = _candidate_table_config(config)
    records_path = _normal_text(candidate_table.get("records_path"))
    if not records_path:
        raise ValueError(
            "candidate feature table config is missing required field: candidate_feature_table.records_path"
        )
    return _resolve_repo_path(Path(repo_root), records_path)


def _configured_x_column(config: Mapping[str, Any]) -> str:
    candidate_table = _candidate_table_config(config)
    x_column = _normal_text(candidate_table.get("x_column"))
    if not x_column:
        raise ValueError("candidate feature table config is missing required field: candidate_feature_table.x_column")
    return x_column


def _configured_view_rows_path(config: Mapping[str, Any], *, repo_root: str | Path) -> Path | None:
    candidate_table = _candidate_table_config(config)
    x_source = candidate_table.get("x_source")
    if not isinstance(x_source, Mapping):
        return None
    rows_path = _normal_text(x_source.get("rows_path"))
    if not rows_path:
        return None
    return _resolve_repo_path(Path(repo_root), rows_path)


def _configured_view_row_id_column(config: Mapping[str, Any]) -> str:
    materialization = _candidate_table_config(config).get("materialization")
    if isinstance(materialization, Mapping):
        return _normal_text(materialization.get("view_row_id_column")) or "construct__anchor_id"
    return "construct__anchor_id"


def _configured_expected_rows(config: Mapping[str, Any]) -> int | None:
    candidate_table = _candidate_table_config(config)
    raw_rows = candidate_table.get("expected_rows")
    if raw_rows is None:
        return None
    if isinstance(raw_rows, bool) or not isinstance(raw_rows, int) or raw_rows <= 0:
        raise ValueError("candidate_feature_table.expected_rows must be a positive integer")
    return int(raw_rows)


def _configured_allowed_source_classes(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    materialization = _candidate_table_config(config).get("materialization")
    if not isinstance(materialization, Mapping):
        return None
    values = materialization.get("include_source_class")
    if values is None:
        return None
    return tuple(_normal_text(value) for value in values if _normal_text(value))


def _configured_allowed_design_families(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    materialization = _candidate_table_config(config).get("materialization")
    if not isinstance(materialization, Mapping):
        return None
    values = materialization.get("allowed_design_families")
    if values is None:
        return None
    return tuple(_normal_text(value) for value in values if _normal_text(value))


def _configured_required_null_provenance_columns(config: Mapping[str, Any]) -> tuple[str, ...]:
    materialization = _candidate_table_config(config).get("materialization")
    if not isinstance(materialization, Mapping):
        return ()
    return tuple(
        f"opal_candidate__{_normal_text(column)}"
        for column in materialization.get("exclude_non_null_columns") or ()
        if _normal_text(column)
    )


def _configured_candidate_feature_table_ids(config: Mapping[str, Any], *, repo_root: str | Path) -> set[str]:
    records = _read_required_parquet(
        _configured_records_path(config, repo_root=repo_root),
        label="candidate feature table records_path",
        columns=["id"],
    )
    return set(records["id"].astype(str).tolist())


def validate_selected_ids_against_candidate_feature_table(
    selected: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, int]:
    """Ensure selected handoff rows are valid OPAL candidate-table rows."""

    candidate_ids = _configured_candidate_feature_table_ids(config, repo_root=repo_root)
    selected_ids = selected["id"].astype(str)
    missing_ids = sorted(set(selected_ids.tolist()) - candidate_ids)
    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        raise ValueError(
            "selected batch-0 rows are missing from the OPAL candidate feature table: "
            f"{sample}. Refresh the configured records.parquet before selecting batch-0 rows."
        )
    return {"selected_row_count": int(len(selected)), "candidate_row_count": int(len(candidate_ids))}


def validate_configured_candidate_feature_table(config: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, int]:
    candidate_table = _candidate_table_config(config)
    records_path = _normal_text(candidate_table.get("records_path"))
    x_column = _normal_text(candidate_table.get("x_column"))
    missing = [
        field
        for field, value in {
            "candidate_feature_table.records_path": records_path,
            "candidate_feature_table.x_column": x_column,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"candidate feature table config is missing required field(s): {', '.join(missing)}")

    root = Path(repo_root)
    return validate_candidate_feature_table(
        records_path=_resolve_repo_path(root, records_path),
        x_column=x_column,
        expected_rows=_configured_expected_rows(config),
        allowed_source_classes=_configured_allowed_source_classes(config),
        allowed_design_families=_configured_allowed_design_families(config),
        required_null_provenance_columns=_configured_required_null_provenance_columns(config),
        view_rows_path=_configured_view_rows_path(config, repo_root=root),
        view_row_id_column=_configured_view_row_id_column(config),
    )


def _mask_candidate_population(view_rows: pd.DataFrame, materialization: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=view_rows.index)
    source_classes = list(materialization.get("include_source_class") or [])
    if source_classes:
        if "source_class" not in view_rows.columns:
            raise ValueError("candidate materialization filter requires source_class in LatentDNA view rows")
        mask &= view_rows["source_class"].astype(str).isin([str(value) for value in source_classes])

    design_families = list(materialization.get("allowed_design_families") or [])
    if design_families:
        if "design_family" not in view_rows.columns:
            raise ValueError("candidate materialization filter requires design_family in LatentDNA view rows")
        mask &= view_rows["design_family"].astype(str).isin([str(value) for value in design_families])

    for column in materialization.get("exclude_non_null_columns") or []:
        column_name = str(column)
        if column_name not in view_rows.columns:
            raise ValueError(f"candidate materialization filter requires {column_name!r} in LatentDNA view rows")
        mask &= view_rows[column_name].isna()
    return mask


def _source_table_for_ids(source_records_path: Path, ids: Sequence[str]) -> pa.Table:
    if not source_records_path.exists():
        raise ValueError(f"candidate source records not found: {source_records_path}")
    source_table = pq.read_table(source_records_path)
    if "id" not in source_table.column_names:
        raise ValueError(f"candidate source records missing id column: {source_records_path}")
    source_ids = [str(value) for value in source_table["id"].to_pylist()]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("candidate source records contain duplicate ids")
    source_index = {row_id: idx for idx, row_id in enumerate(source_ids)}
    missing = [row_id for row_id in ids if row_id not in source_index]
    if missing:
        raise ValueError(f"candidate ids are missing from source records (sample={missing[:5]})")
    return source_table.take(pa.array([source_index[row_id] for row_id in ids], type=pa.int64()))


def _configured_densegen_sidecar_columns(materialization: Mapping[str, Any]) -> list[str]:
    raw_columns = materialization.get("densegen_sidecar_columns") or DENSEGEN_KEY_COLUMNS
    columns = [str(column) for column in raw_columns]
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise ValueError("candidate materialization densegen_sidecar_columns must not be empty")
    return columns


def _densegen_sidecar_table_for_ids(
    *,
    sidecar_path: Path,
    ids: Sequence[str],
    columns: Sequence[str],
) -> pa.Table:
    if not sidecar_path.exists():
        raise ValueError(f"candidate DenseGen sidecar not found: {sidecar_path}")
    schema_names = set(pq.ParquetFile(sidecar_path).schema_arrow.names)
    missing_columns = [column for column in ("id", *columns) if column not in schema_names]
    if missing_columns:
        raise ValueError(f"candidate DenseGen sidecar missing required column(s): {missing_columns}")
    sidecar = pq.read_table(sidecar_path, columns=["id", *columns])
    sidecar_ids = [str(value) for value in sidecar["id"].to_pylist()]
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for row_id in sidecar_ids:
        if row_id in seen_ids:
            duplicate_ids.add(row_id)
        seen_ids.add(row_id)
    if duplicate_ids:
        raise ValueError(f"candidate DenseGen sidecar contains duplicate ids (sample={sorted(duplicate_ids)[:5]})")
    sidecar_index = {row_id: idx for idx, row_id in enumerate(sidecar_ids)}
    missing = [row_id for row_id in ids if row_id not in sidecar_index]
    if missing:
        raise ValueError(f"candidate ids are missing from DenseGen sidecar (sample={missing[:5]})")

    aligned = sidecar.take(pa.array([sidecar_index[row_id] for row_id in ids], type=pa.int64()))
    null_columns = [column for column in columns if aligned[column].null_count]
    if null_columns:
        raise ValueError(f"candidate DenseGen sidecar has null required column(s): {null_columns}")
    return aligned


def _replace_columns(table: pa.Table, replacement: pa.Table, *, columns: Sequence[str]) -> pa.Table:
    for column in columns:
        if column in table.column_names:
            table = table.drop([column])
        table = table.append_column(column, replacement[column])
    return table


def _append_constant_column(table: pa.Table, name: str, value: str) -> pa.Table:
    if name in table.column_names:
        table = table.drop([name])
    return table.append_column(name, pa.array([value] * table.num_rows, type=pa.string()))


def _append_string_column(table: pa.Table, name: str, values: Sequence[Any]) -> pa.Table:
    if len(values) != table.num_rows:
        raise ValueError(f"candidate provenance column {name!r} has {len(values)} rows, expected {table.num_rows}")
    if name in table.column_names:
        table = table.drop([name])
    normalized = [None if _is_missing(value) else str(value) for value in values]
    return table.append_column(name, pa.array(normalized, type=pa.string()))


def _append_view_provenance_columns(
    table: pa.Table,
    selected_view_rows: pd.DataFrame,
    *,
    exclude_non_null_columns: Sequence[Any],
) -> pa.Table:
    provenance_columns = [*VIEW_PROVENANCE_COLUMNS, *(str(column) for column in exclude_non_null_columns)]
    for column in dict.fromkeys(provenance_columns):
        if column not in selected_view_rows.columns:
            raise ValueError(f"candidate materialization provenance requires {column!r} in LatentDNA view rows")
        table = _append_string_column(
            table,
            f"opal_candidate__{column}",
            selected_view_rows[column].tolist(),
        )
    return table


def _write_candidate_records(
    *,
    records_path: Path,
    base_table: pa.Table,
    x_column: str,
    matrix: np.ndarray,
    positions: np.ndarray,
    chunk_size: int,
) -> None:
    missing = [column for column in REQUIRED_OPAL_COLUMNS if column not in base_table.column_names]
    if missing:
        raise ValueError(f"candidate source records missing required OPAL columns: {missing}")
    _validate_required_opal_table_values(base_table, label="candidate source records")
    if x_column in base_table.column_names:
        base_table = base_table.drop([x_column])

    tmp = records_path.with_name(f".{records_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        x_dim = int(matrix.shape[1])
        for start in range(0, len(positions), int(chunk_size)):
            stop = min(start + int(chunk_size), len(positions))
            chunk_positions = positions[start:stop]
            x_values = np.asarray(matrix[chunk_positions], dtype=np.float32)
            if x_values.ndim != 2 or x_values.shape[1] != x_dim:
                raise ValueError("candidate X matrix chunk has inconsistent dimensions")
            if not np.all(np.isfinite(x_values)):
                raise ValueError(f"candidate X matrix contains non-finite values in rows {start}:{stop}")
            flat = pa.array(x_values.reshape(-1), type=pa.float32())
            x_array = pa.FixedSizeListArray.from_arrays(flat, x_dim)
            chunk_table = base_table.slice(start, stop - start).append_column(x_column, x_array)
            if writer is None:
                writer = pq.ParquetWriter(tmp, chunk_table.schema)
            writer.write_table(chunk_table)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(records_path)


def materialize_configured_candidate_feature_table(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
    write: bool = False,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Dry-run or write the configured OPAL candidate feature table."""

    root = Path(repo_root)
    candidate_table = _candidate_table_config(config)
    x_source = _x_source_config(candidate_table)
    materialization = _materialization_config(candidate_table)

    records_path = _configured_records_path(config, repo_root=root)
    x_column = _configured_x_column(config)
    view_rows_path = _resolve_repo_path(root, _normal_text(x_source.get("rows_path")))
    matrix_path = _resolve_repo_path(root, _normal_text(x_source.get("matrix_path")))
    source_records_path = _resolve_repo_path(root, _normal_text(materialization.get("source_records_path")))
    view_row_id_column = _normal_text(materialization.get("view_row_id_column")) or "construct__anchor_id"
    densegen_sidecar_path_text = _normal_text(materialization.get("densegen_sidecar_path"))
    densegen_sidecar_path = _resolve_repo_path(root, densegen_sidecar_path_text) if densegen_sidecar_path_text else None
    densegen_sidecar_columns = _configured_densegen_sidecar_columns(materialization)

    view_columns = [view_row_id_column, *VIEW_PROVENANCE_COLUMNS]
    view_columns.extend(str(column) for column in materialization.get("exclude_non_null_columns") or [])
    view_columns = list(dict.fromkeys(view_columns))
    view_rows = _read_required_parquet(view_rows_path, label="LatentDNA view rows", columns=view_columns)
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"LatentDNA X matrix must be 2D: {matrix_path}")
    if int(matrix.shape[0]) != int(len(view_rows)):
        raise ValueError(
            "LatentDNA X matrix rows do not match LatentDNA view rows: "
            f"matrix={matrix.shape[0]} view_rows={len(view_rows)}"
        )

    mask = _mask_candidate_population(view_rows, materialization)
    positions = np.flatnonzero(mask.to_numpy())
    selected_view_rows = view_rows.iloc[positions].reset_index(drop=True)
    candidate_ids = selected_view_rows[view_row_id_column].astype(str).tolist()
    if not candidate_ids:
        raise ValueError("candidate materialization filter selected zero rows")
    base_table = _source_table_for_ids(source_records_path, candidate_ids)
    if densegen_sidecar_path is not None:
        densegen_sidecar = _densegen_sidecar_table_for_ids(
            sidecar_path=densegen_sidecar_path,
            ids=candidate_ids,
            columns=densegen_sidecar_columns,
        )
        base_table = _replace_columns(base_table, densegen_sidecar, columns=densegen_sidecar_columns)
    base_table = _append_constant_column(
        base_table,
        "opal_candidate__role",
        _normal_text(candidate_table.get("role")) or "opal_candidate_feature_table",
    )
    base_table = _append_constant_column(
        base_table,
        "opal_candidate__x_source_view_id",
        _normal_text(x_source.get("view_id")) or "unknown",
    )
    base_table = _append_view_provenance_columns(
        base_table,
        selected_view_rows,
        exclude_non_null_columns=materialization.get("exclude_non_null_columns") or [],
    )

    report: dict[str, Any] = {
        "records_path": str(records_path),
        "write": bool(write),
        "dataset_id": _normal_text(candidate_table.get("dataset_id")),
        "role": _normal_text(candidate_table.get("role")),
        "row_count": int(len(candidate_ids)),
        "source_population_rows": int(len(view_rows)),
        "x_column": x_column,
        "x_dim": int(matrix.shape[1]),
        "x_dtype": str(matrix.dtype),
        "view_rows_path": str(view_rows_path),
        "matrix_path": str(matrix_path),
        "source_records_path": str(source_records_path),
        "densegen_sidecar_path": str(densegen_sidecar_path) if densegen_sidecar_path is not None else None,
        "densegen_sidecar_columns": list(densegen_sidecar_columns) if densegen_sidecar_path is not None else [],
        "provenance_columns": [
            "opal_candidate__source_class",
            "opal_candidate__design_family",
            *[
                f"opal_candidate__{column}"
                for column in dict.fromkeys(
                    str(column) for column in materialization.get("exclude_non_null_columns") or []
                )
            ],
        ],
        "filter": {
            "include_source_class": list(materialization.get("include_source_class") or []),
            "allowed_design_families": list(materialization.get("allowed_design_families") or []),
            "exclude_non_null_columns": list(materialization.get("exclude_non_null_columns") or []),
        },
    }
    if write:
        _write_candidate_records(
            records_path=records_path,
            base_table=base_table,
            x_column=x_column,
            matrix=matrix,
            positions=positions,
            chunk_size=int(chunk_size),
        )
        report["validation"] = validate_configured_candidate_feature_table(config, repo_root=root)
        report["written"] = True
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the OPAL candidate feature table.")
    parser.add_argument("--config", default=Path(__file__).with_name("sampling.yaml"), type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--validate-existing", action="store_true", help="Validate the configured records.parquet.")
    parser.add_argument("--write", action="store_true", help="Write records.parquet; default is dry-run only.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Rows per parquet write chunk.")
    args = parser.parse_args(argv)
    if args.validate_existing and args.write:
        parser.error("--validate-existing cannot be combined with --write")

    config = _load_sampling_config(args.config)
    repo_root = args.repo_root or _repo_root_from(args.config)
    if args.validate_existing:
        report: dict[str, Any] = {
            "mode": "validate_existing",
            **validate_configured_candidate_feature_table(config, repo_root=repo_root),
        }
    else:
        report = materialize_configured_candidate_feature_table(
            config,
            repo_root=repo_root,
            write=bool(args.write),
            chunk_size=int(args.chunk_size),
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
