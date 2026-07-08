"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/provenance.py

Sidecar-aware provenance checks for the stress/ethanol/ciprofloxacin OPAL handoff.

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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .candidate_table import (
    VIEW_PROVENANCE_COLUMNS,
    _candidate_table_config,
    _configured_records_path,
    _configured_view_row_id_column,
    _configured_view_rows_path,
    _configured_x_column,
    _load_sampling_config,
    _mask_candidate_population,
    _materialization_config,
    _normal_text,
    _repo_root_from,
    _resolve_repo_path,
    _x_source_config,
)

DENSEGEN_KEY_COLUMNS: tuple[str, ...] = (
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
)
OPTIONAL_RECORD_COLUMNS: tuple[str, ...] = (
    "source",
    "canonical_densegen_plan",
    "regulator_composition",
    "sigma35_variant",
    "spacer_length",
    "opal_candidate__role",
    "opal_candidate__x_source_view_id",
    "opal_candidate__source_class",
    "opal_candidate__design_family",
    "opal_candidate__sfxi_ref__collection_id",
)
CONSTRUCT_VIEW_COLUMNS: tuple[str, ...] = (
    "view_id",
    "sequence_id",
    "parent_sequence_id",
    "parent_dataset_id",
    "product_kind",
    "context_kind",
    "orientation",
    "recommended_pooling",
)
INFER_ALIAS_COLUMNS: tuple[str, ...] = (
    "alias_id",
    "view_id",
    "sequence_id",
    "feature_vector_key",
    "provider",
    "model_name",
    "layer_name",
    "representation_kind",
    "pooling_operation",
    "orientation",
    "source_dataset_id",
    "feature_request_digest",
    "runtime_fingerprint_key",
)
EXPECTED_CONTEXT_ORIENTATIONS = ("forward", "reverse_complement")


def _provenance_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sampling config is missing provenance mapping")
    return provenance


def _configured_provenance_path(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
    key: str,
) -> Path:
    value = _normal_text(_provenance_config(config).get(key))
    if not value:
        raise ValueError(f"sampling config provenance is missing required field: provenance.{key}")
    return _resolve_repo_path(Path(repo_root), value)


def _schema_names(path: str | Path) -> set[str]:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise ValueError(f"provenance artifact not found: {parquet_path}")
    return set(pq.ParquetFile(parquet_path).schema_arrow.names)


def _present_columns(path: str | Path, requested: Sequence[str], *, required: Sequence[str] = ()) -> list[str]:
    names = _schema_names(path)
    missing_required = [column for column in required if column not in names]
    if missing_required:
        raise ValueError(f"{Path(path)} missing required column(s): {missing_required}")
    return [column for column in dict.fromkeys([*required, *requested]) if column in names]


def _read_parquet_columns(
    path: str | Path,
    *,
    label: str,
    columns: Sequence[str],
    required: Sequence[str] = (),
) -> pd.DataFrame:
    parquet_path = Path(path)
    selected = _present_columns(parquet_path, columns, required=required)
    try:
        return pd.read_parquet(parquet_path, columns=selected)
    except Exception as exc:
        raise ValueError(f"failed to read {label} at {parquet_path}: {exc}") from exc


def _read_one(
    path: str | Path,
    *,
    label: str,
    id_column: str,
    value: str,
    columns: Sequence[str],
) -> dict[str, Any] | None:
    parquet_path = Path(path)
    selected = _present_columns(parquet_path, columns, required=[id_column])
    try:
        table = ds.dataset(str(parquet_path)).to_table(
            columns=selected,
            filter=ds.field(id_column) == str(value),
        )
    except Exception as exc:
        raise ValueError(f"failed to scan {label} at {parquet_path}: {exc}") from exc
    if table.num_rows == 0:
        return None
    if table.num_rows > 1:
        raise ValueError(f"{label} has multiple rows for {id_column}={value!r}")
    return table.to_pylist()[0]


def _read_many_by_parent(
    path: str | Path,
    *,
    label: str,
    parent_column: str,
    value: str,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    parquet_path = Path(path)
    selected = _present_columns(parquet_path, columns, required=[parent_column])
    try:
        table = ds.dataset(str(parquet_path)).to_table(
            columns=selected,
            filter=ds.field(parent_column) == str(value),
        )
    except Exception as exc:
        raise ValueError(f"failed to scan {label} at {parquet_path}: {exc}") from exc
    return table.to_pylist()


def _filter_aliases(frame: pd.DataFrame, alias_filter: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column, expected in alias_filter.items():
        if column not in out.columns:
            raise ValueError(f"infer alias filter requires column {column!r}")
        out = out.loc[out[column].astype(str) == str(expected)].copy()
    return out


def _row_position(path: str | Path, *, id_column: str, value: str, label: str) -> int | None:
    ids = _read_parquet_columns(path, label=label, columns=[id_column], required=[id_column])[id_column].astype(str)
    matches = np.flatnonzero(ids.to_numpy() == str(value))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(f"{label} contains duplicate rows for {id_column}={value!r}")
    return int(matches[0])


def _fixed_size_list_dim(path: str | Path, column: str) -> int | None:
    schema = pq.ParquetFile(Path(path)).schema_arrow
    if column not in schema.names:
        return None
    field = schema.field(column)
    if pa.types.is_fixed_size_list(field.type):
        return int(field.type.list_size)
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _jsonable(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _clean_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {str(key): _jsonable(value) for key, value in record.items()}


def _null_counts(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in columns if column in frame.columns}


def _ids_with_nulls(frame: pd.DataFrame, columns: Sequence[str], *, id_column: str = "id") -> set[str]:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return set()
    mask = frame[present].isna().any(axis=1)
    return set(frame.loc[mask, id_column].astype(str).tolist())


def _sidecar_null_counts_for_ids(
    sidecar: pd.DataFrame,
    ids: set[str],
    columns: Sequence[str],
) -> dict[str, int]:
    if not ids:
        return {column: 0 for column in columns if column in sidecar.columns}
    sidecar_ids = sidecar["id"].astype(str)
    subset = sidecar.loc[sidecar_ids.isin(ids)]
    return _null_counts(subset, columns)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).to_dict().items()}


def _ids(frame: pd.DataFrame, *, column: str = "id") -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"required id column missing: {column}")
    return frame[column].astype(str)


def _duplicate_count(frame: pd.DataFrame, *, column: str = "id") -> int:
    ids = _ids(frame, column=column)
    return int(ids.duplicated().sum())


def _first_present(
    source_name_and_records: Sequence[tuple[str, Mapping[str, Any] | None]],
    column: str,
) -> tuple[Any, str | None]:
    for source_name, record in source_name_and_records:
        if record is None or column not in record:
            continue
        value = record.get(column)
        if not _is_missing(value):
            return value, source_name
    return None, None


def _candidate_records_columns(x_column: str) -> list[str]:
    return [
        "id",
        "bio_type",
        "sequence",
        "alphabet",
        *OPTIONAL_RECORD_COLUMNS,
        *DENSEGEN_KEY_COLUMNS,
        x_column,
    ]


def _lineage_paths(config: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Path]:
    candidate_table = _candidate_table_config(config)
    materialization = _materialization_config(candidate_table)
    source_records_path = _normal_text(materialization.get("source_records_path"))
    if not source_records_path:
        raise ValueError("candidate_feature_table.materialization.source_records_path is required")
    return {
        "opal_records": _configured_records_path(config, repo_root=repo_root),
        "latentdna_rows": _configured_view_rows_path(config, repo_root=repo_root) or Path(""),
        "latentdna_matrix": _resolve_repo_path(
            Path(repo_root),
            _normal_text(_x_source_config(candidate_table).get("matrix_path")),
        ),
        "anchor_records": _resolve_repo_path(Path(repo_root), source_records_path),
        "densegen_source_records": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="densegen_source_records_path",
        ),
        "densegen_source_sidecar": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="densegen_source_sidecar_path",
        ),
        "anchor_densegen_sidecar": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="anchor_densegen_sidecar_path",
        ),
        "construct_sequence_views": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="construct_sequence_views_path",
        ),
        "infer_feature_aliases": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="infer_feature_aliases_path",
        ),
        "infer_feature_vectors": _configured_provenance_path(
            config,
            repo_root=repo_root,
            key="infer_feature_vectors_path",
        ),
    }


def audit_candidate_lineage(config: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    """Validate all OPAL candidate IDs against study-owned provenance artifacts."""

    root = Path(repo_root)
    paths = _lineage_paths(config, repo_root=root)
    candidate_table = _candidate_table_config(config)
    materialization = _materialization_config(candidate_table)
    x_column = _configured_x_column(config)
    view_row_id_column = _configured_view_row_id_column(config)
    alias_filter = dict(_provenance_config(config).get("infer_alias_filter") or {})
    if not alias_filter:
        raise ValueError("sampling config provenance.infer_alias_filter must not be empty")

    opal = _read_parquet_columns(
        paths["opal_records"],
        label="OPAL candidate records",
        columns=_candidate_records_columns(x_column),
        required=["id", "sequence", x_column, *DENSEGEN_KEY_COLUMNS],
    )
    dense_source = _read_parquet_columns(
        paths["densegen_source_records"],
        label="DenseGen source records",
        columns=["id", "sequence", "source", *DENSEGEN_KEY_COLUMNS],
        required=["id", "sequence", *DENSEGEN_KEY_COLUMNS],
    )
    anchor = _read_parquet_columns(
        paths["anchor_records"],
        label="anchor records",
        columns=["id", "sequence", "source", *DENSEGEN_KEY_COLUMNS],
        required=["id", "sequence"],
    )
    view_columns = [view_row_id_column, *VIEW_PROVENANCE_COLUMNS]
    view_columns.extend(str(column) for column in materialization.get("exclude_non_null_columns") or [])
    latent_rows = _read_parquet_columns(
        paths["latentdna_rows"],
        label="LatentDNA selected-X rows",
        columns=view_columns,
        required=[view_row_id_column],
    )

    opal_ids = _ids(opal).tolist()
    opal_id_set = set(opal_ids)
    dense_by_id = dense_source.set_index(dense_source["id"].astype(str), drop=False)
    anchor_by_id = anchor.set_index(anchor["id"].astype(str), drop=False)
    failures: list[str] = []

    duplicate_ids = _duplicate_count(opal)
    if duplicate_ids:
        failures.append(f"OPAL candidate records contain {duplicate_ids} duplicate id rows")
    missing_dense = sorted(opal_id_set - set(dense_by_id.index))
    missing_anchor = sorted(opal_id_set - set(anchor_by_id.index))
    if missing_dense:
        failures.append(f"{len(missing_dense)} OPAL ids are missing from DenseGen source records")
    if missing_anchor:
        failures.append(f"{len(missing_anchor)} OPAL ids are missing from anchor records")

    comparable_dense_ids = [row_id for row_id in opal_ids if row_id in dense_by_id.index]
    dense_sequence_mismatch = sum(
        str(opal.loc[index, "sequence"]) != str(dense_by_id.loc[row_id, "sequence"])
        for index, row_id in enumerate(opal_ids)
        if row_id in dense_by_id.index
    )
    anchor_sequence_mismatch = sum(
        str(opal.loc[index, "sequence"]) != str(anchor_by_id.loc[row_id, "sequence"])
        for index, row_id in enumerate(opal_ids)
        if row_id in anchor_by_id.index
    )
    if dense_sequence_mismatch:
        failures.append(f"{dense_sequence_mismatch} OPAL sequences differ from DenseGen source records")
    if anchor_sequence_mismatch:
        failures.append(f"{anchor_sequence_mismatch} OPAL sequences differ from anchor records")

    mask = _mask_candidate_population(
        latent_rows,
        materialization,
        view_row_id_column=view_row_id_column,
    )
    latent_selected_ids = latent_rows.loc[mask, view_row_id_column].astype(str).tolist()
    if opal_ids != latent_selected_ids:
        failures.append("OPAL candidate id order does not match the configured LatentDNA selected-X subset")

    construct = _read_parquet_columns(
        paths["construct_sequence_views"],
        label="Construct sequence views",
        columns=CONSTRUCT_VIEW_COLUMNS,
        required=["sequence_id", "parent_sequence_id", "orientation"],
    )
    construct_subset = construct.loc[construct["parent_sequence_id"].astype(str).isin(opal_id_set)].copy()
    construct_parent_ids = set(construct_subset["parent_sequence_id"].astype(str).tolist())
    missing_construct_ids = sorted(opal_id_set - construct_parent_ids)
    if missing_construct_ids:
        failures.append(f"{len(missing_construct_ids)} OPAL ids are missing Construct context views")
    views_per_id = construct_subset.groupby(construct_subset["parent_sequence_id"].astype(str)).size()
    ids_with_not_two_views = int((views_per_id != 2).sum() + len(missing_construct_ids))
    if ids_with_not_two_views:
        failures.append(f"{ids_with_not_two_views} OPAL ids do not have exactly two Construct context views")
    missing_orientation_counts = {}
    for orientation in EXPECTED_CONTEXT_ORIENTATIONS:
        ids_with_orientation = set(
            construct_subset.loc[
                construct_subset["orientation"].astype(str) == orientation,
                "parent_sequence_id",
            ].astype(str)
        )
        missing_orientation_counts[orientation] = int(len(opal_id_set - ids_with_orientation))
    for orientation, count in missing_orientation_counts.items():
        if count:
            failures.append(f"{count} OPAL ids are missing {orientation} Construct views")

    construct_sequence_ids = set(construct_subset["sequence_id"].astype(str).tolist())
    aliases = _read_parquet_columns(
        paths["infer_feature_aliases"],
        label="Infer feature aliases",
        columns=INFER_ALIAS_COLUMNS,
        required=["sequence_id", "feature_vector_key"],
    )
    aliases = aliases.loc[aliases["sequence_id"].astype(str).isin(construct_sequence_ids)].copy()
    aliases = _filter_aliases(aliases, alias_filter)
    alias_sequence_ids = set(aliases["sequence_id"].astype(str).tolist())
    missing_alias_sequence_ids = sorted(construct_sequence_ids - alias_sequence_ids)
    if missing_alias_sequence_ids:
        failures.append(f"{len(missing_alias_sequence_ids)} Construct context views are missing expected Infer aliases")

    vectors = _read_parquet_columns(
        paths["infer_feature_vectors"],
        label="Infer feature vectors",
        columns=["feature_vector_key"],
        required=["feature_vector_key"],
    )
    vector_keys = set(vectors["feature_vector_key"].astype(str).tolist())
    alias_keys = set(aliases["feature_vector_key"].astype(str).tolist())
    missing_vector_keys = sorted(alias_keys - vector_keys)
    if missing_vector_keys:
        failures.append(f"{len(missing_vector_keys)} Infer aliases are missing feature-vector payloads")

    source_sidecar = _read_parquet_columns(
        paths["densegen_source_sidecar"],
        label="DenseGen source sidecar",
        columns=["id", *DENSEGEN_KEY_COLUMNS],
        required=["id", *DENSEGEN_KEY_COLUMNS],
    )
    anchor_sidecar = _read_parquet_columns(
        paths["anchor_densegen_sidecar"],
        label="anchor DenseGen sidecar",
        columns=["id", *DENSEGEN_KEY_COLUMNS],
        required=["id", *DENSEGEN_KEY_COLUMNS],
    )
    source_sidecar_ids = set(source_sidecar["id"].astype(str).tolist())
    anchor_sidecar_ids = set(anchor_sidecar["id"].astype(str).tolist())
    opal_nulls = _null_counts(opal, DENSEGEN_KEY_COLUMNS)
    source_nulls = _null_counts(dense_source, DENSEGEN_KEY_COLUMNS)
    opal_null_ids = _ids_with_nulls(opal, DENSEGEN_KEY_COLUMNS)
    source_null_ids = _ids_with_nulls(dense_source, DENSEGEN_KEY_COLUMNS)
    missing_source_null_sidecar_ids = sorted(source_null_ids - source_sidecar_ids)
    missing_anchor_opal_ids = sorted(opal_id_set - anchor_sidecar_ids)
    source_sidecar_null_counts_for_source_null_ids = _sidecar_null_counts_for_ids(
        source_sidecar,
        source_null_ids,
        DENSEGEN_KEY_COLUMNS,
    )
    anchor_sidecar_null_counts_for_opal_ids = _sidecar_null_counts_for_ids(
        anchor_sidecar,
        opal_id_set,
        DENSEGEN_KEY_COLUMNS,
    )
    if missing_source_null_sidecar_ids:
        failures.append(
            f"{len(missing_source_null_sidecar_ids)} DenseGen source record null ids are missing "
            "from the DenseGen source sidecar"
        )
    if any(count > 0 for count in source_sidecar_null_counts_for_source_null_ids.values()):
        failures.append("DenseGen source sidecar has null plan/run/hash values for source-record null ids")
    if any(count > 0 for count in anchor_sidecar_null_counts_for_opal_ids.values()):
        failures.append("anchor DenseGen sidecar has null plan/run/hash values for OPAL ids")
    if opal_id_set - anchor_sidecar_ids:
        failures.append(
            f"{len(opal_id_set - anchor_sidecar_ids)} OPAL ids are missing from the anchor DenseGen sidecar"
        )
    uses_sidecar_resolution = bool(opal_null_ids or source_null_ids)
    resolution_state = "complete_via_anchor_sidecar" if uses_sidecar_resolution else "complete_inline"

    matrix_shape: tuple[int, ...] | None = None
    if not paths["latentdna_matrix"].exists():
        failures.append(f"LatentDNA X matrix is missing: {paths['latentdna_matrix']}")
    else:
        matrix = np.load(paths["latentdna_matrix"], mmap_mode="r")
        matrix_shape = tuple(int(dim) for dim in matrix.shape)
        if matrix.ndim != 2:
            failures.append(f"LatentDNA X matrix must be 2D: {paths['latentdna_matrix']}")
        elif int(matrix.shape[0]) != int(len(latent_rows)):
            failures.append("LatentDNA X matrix row count does not match LatentDNA rows.parquet")

    if failures:
        raise ValueError("candidate lineage audit failed: " + "; ".join(failures[:8]))

    return {
        "status": "pass_with_sidecar" if uses_sidecar_resolution else "pass",
        "attention": [],
        "paths": {key: str(path) for key, path in paths.items()},
        "candidate_table": {
            "row_count": int(len(opal)),
            "x_column": x_column,
            "x_dim": _fixed_size_list_dim(paths["opal_records"], x_column),
            "x_source_view_id": _normal_text(_x_source_config(candidate_table).get("view_id")),
        },
        "identity": {
            "opal_duplicate_id_rows": duplicate_ids,
            "densegen_source_rows": int(len(dense_source)),
            "anchor_rows": int(len(anchor)),
            "ids_equal_densegen_source": opal_id_set == set(dense_by_id.index),
            "ids_equal_anchor_subset": len(missing_anchor) == 0,
            "sequence_mismatch_vs_densegen_source": int(dense_sequence_mismatch),
            "sequence_mismatch_vs_anchor": int(anchor_sequence_mismatch),
            "comparable_densegen_ids": int(len(comparable_dense_ids)),
        },
        "latentdna": {
            "rows": int(len(latent_rows)),
            "selected_rows": int(len(latent_selected_ids)),
            "selected_order_matches_opal": True,
            "matrix_shape": list(matrix_shape or []),
            "derive_kind": "block_normalized_concatenate",
        },
        "construct": {
            "matched_view_rows": int(len(construct_subset)),
            "ids_with_views": int(len(construct_parent_ids)),
            "ids_with_not_two_views": ids_with_not_two_views,
            "missing_orientation_counts": missing_orientation_counts,
            "product_kind_counts": _value_counts(construct_subset, "product_kind")
            if "product_kind" in construct_subset.columns
            else {},
            "context_kind_counts": _value_counts(construct_subset, "context_kind")
            if "context_kind" in construct_subset.columns
            else {},
            "recommended_pooling_counts": {
                str(key): int(value)
                for key, value in construct_subset["recommended_pooling"].value_counts(dropna=False).to_dict().items()
            }
            if "recommended_pooling" in construct_subset.columns
            else {},
        },
        "infer": {
            "alias_filter": {str(key): str(value) for key, value in alias_filter.items()},
            "matched_alias_rows": int(len(aliases)),
            "context_view_sequence_ids": int(len(construct_sequence_ids)),
            "missing_alias_sequence_ids": 0,
            "feature_vector_payload_rows": int(len(vectors)),
            "missing_feature_vector_keys": 0,
        },
        "densegen_metadata": {
            "opal_record_null_counts": opal_nulls,
            "densegen_source_record_null_counts": source_nulls,
            "densegen_source_sidecar_rows": int(len(source_sidecar)),
            "densegen_source_sidecar_missing_opal_ids": int(len(opal_id_set - source_sidecar_ids)),
            "densegen_source_sidecar_missing_source_record_null_ids": int(len(missing_source_null_sidecar_ids)),
            "densegen_source_sidecar_null_counts_for_source_record_null_ids": (
                source_sidecar_null_counts_for_source_null_ids
            ),
            "anchor_densegen_sidecar_rows": int(len(anchor_sidecar)),
            "anchor_densegen_sidecar_missing_opal_ids": int(len(missing_anchor_opal_ids)),
            "anchor_densegen_sidecar_null_counts_for_opal_ids": anchor_sidecar_null_counts_for_opal_ids,
            "resolution_state": resolution_state,
            "resolved_contract": (
                "join usr_prom_eth_cip_anchor/_derived/densegen.parquet by id "
                "for complete DenseGen plan/run/hash provenance"
            ),
        },
    }


def show_candidate_lineage(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Return a per-ID provenance trace without reading the vector-valued X payload."""

    root = Path(repo_root)
    paths = _lineage_paths(config, repo_root=root)
    candidate_table = _candidate_table_config(config)
    x_column = _configured_x_column(config)
    view_row_id_column = _configured_view_row_id_column(config)
    alias_filter = dict(_provenance_config(config).get("infer_alias_filter") or {})
    if not alias_filter:
        raise ValueError("sampling config provenance.infer_alias_filter must not be empty")

    record_columns = _candidate_records_columns(x_column)
    if x_column in record_columns:
        record_columns.remove(x_column)
    opal = _read_one(
        paths["opal_records"],
        label="OPAL candidate records",
        id_column="id",
        value=candidate_id,
        columns=record_columns,
    )
    if opal is None:
        raise ValueError(f"candidate id is not present in OPAL candidate records: {candidate_id}")
    sequence = str(opal.get("sequence") or "")
    dense_source = _read_one(
        paths["densegen_source_records"],
        label="DenseGen source records",
        id_column="id",
        value=candidate_id,
        columns=["id", "sequence", "source", *DENSEGEN_KEY_COLUMNS],
    )
    anchor = _read_one(
        paths["anchor_records"],
        label="anchor records",
        id_column="id",
        value=candidate_id,
        columns=["id", "sequence", "source", *DENSEGEN_KEY_COLUMNS, *OPTIONAL_RECORD_COLUMNS],
    )
    source_sidecar = _read_one(
        paths["densegen_source_sidecar"],
        label="DenseGen source sidecar",
        id_column="id",
        value=candidate_id,
        columns=["id", *DENSEGEN_KEY_COLUMNS],
    )
    anchor_sidecar = _read_one(
        paths["anchor_densegen_sidecar"],
        label="anchor DenseGen sidecar",
        id_column="id",
        value=candidate_id,
        columns=["id", *DENSEGEN_KEY_COLUMNS],
    )

    densegen_sources = [
        ("anchor_densegen_sidecar", anchor_sidecar),
        ("densegen_source_sidecar", source_sidecar),
        ("opal_records", opal),
        ("densegen_source_records", dense_source),
        ("anchor_records", anchor),
    ]
    resolved_densegen: dict[str, Any] = {}
    resolved_from: dict[str, str | None] = {}
    for column in DENSEGEN_KEY_COLUMNS:
        value, source_name = _first_present(densegen_sources, column)
        resolved_densegen[column] = _jsonable(value)
        resolved_from[column] = source_name

    latent_row = _read_one(
        paths["latentdna_rows"],
        label="LatentDNA selected-X rows",
        id_column=view_row_id_column,
        value=candidate_id,
        columns=[
            view_row_id_column,
            "alias_id",
            "source_class",
            "design_family",
            "sig35_variant",
            "spacer_length",
            "construct__context_id",
            "sfxi_ref__collection_id",
        ],
    )
    construct_views = _read_many_by_parent(
        paths["construct_sequence_views"],
        label="Construct sequence views",
        parent_column="parent_sequence_id",
        value=candidate_id,
        columns=CONSTRUCT_VIEW_COLUMNS,
    )
    construct_sequence_ids = [str(row["sequence_id"]) for row in construct_views if row.get("sequence_id")]
    if construct_sequence_ids:
        aliases_table = ds.dataset(str(paths["infer_feature_aliases"])).to_table(
            columns=_present_columns(paths["infer_feature_aliases"], INFER_ALIAS_COLUMNS, required=["sequence_id"]),
            filter=ds.field("sequence_id").isin(construct_sequence_ids),
        )
        aliases = _filter_aliases(aliases_table.to_pandas(), alias_filter)
        infer_aliases = aliases.to_dict(orient="records")
    else:
        infer_aliases = []

    warnings: list[str] = []
    for column, source_name in resolved_from.items():
        if source_name is None:
            warnings.append(f"could not resolve {column}")
    if len(construct_views) != 2:
        warnings.append(f"expected two Construct context views; found {len(construct_views)}")
    alias_orientations = {str(row.get("orientation")) for row in infer_aliases}
    for orientation in EXPECTED_CONTEXT_ORIENTATIONS:
        if orientation not in alias_orientations:
            warnings.append(f"missing expected {orientation} Infer alias")

    return {
        "id": str(candidate_id),
        "sequence": sequence,
        "row_positions": {
            "opal_candidate_records": _row_position(
                paths["opal_records"],
                id_column="id",
                value=candidate_id,
                label="OPAL candidate records",
            ),
            "latentdna_rows": _row_position(
                paths["latentdna_rows"],
                id_column=view_row_id_column,
                value=candidate_id,
                label="LatentDNA selected-X rows",
            ),
        },
        "opal": {
            "records_path": str(paths["opal_records"]),
            "record": _clean_record(opal),
            "x_column": x_column,
            "x_dim": _fixed_size_list_dim(paths["opal_records"], x_column),
            "x_payload_loaded": False,
        },
        "densegen": {
            "resolved": resolved_densegen,
            "resolved_from": resolved_from,
            "source_record": _clean_record(dense_source),
            "anchor_record": _clean_record(anchor),
            "densegen_source_sidecar": _clean_record(source_sidecar),
            "anchor_densegen_sidecar": _clean_record(anchor_sidecar),
        },
        "latentdna": {
            "view_id": _normal_text(_x_source_config(candidate_table).get("view_id")),
            "rows_path": str(paths["latentdna_rows"]),
            "matrix_path": str(paths["latentdna_matrix"]),
            "derive_kind": "block_normalized_concatenate",
            "row": _clean_record(latent_row),
        },
        "construct": {
            "sequence_views_path": str(paths["construct_sequence_views"]),
            "views": [
                _clean_record(row) for row in sorted(construct_views, key=lambda item: str(item.get("orientation")))
            ],
        },
        "infer": {
            "feature_aliases_path": str(paths["infer_feature_aliases"]),
            "feature_vectors_path": str(paths["infer_feature_vectors"]),
            "alias_filter": {str(key): str(value) for key, value in alias_filter.items()},
            "aliases": [_clean_record(row) for row in infer_aliases],
        },
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or show OPAL candidate provenance for the stress study.")
    parser.add_argument("--config", default=Path(__file__).with_name("sampling.yaml"), type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--id", dest="candidate_id", help="Candidate id to trace. Omit for the all-row audit.")
    args = parser.parse_args(argv)

    config = _load_sampling_config(args.config)
    repo_root = args.repo_root or _repo_root_from(args.config)
    if args.candidate_id:
        report = show_candidate_lineage(config, repo_root=repo_root, candidate_id=str(args.candidate_id))
    else:
        report = audit_candidate_lineage(config, repo_root=repo_root)
    print(json.dumps(_jsonable(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
