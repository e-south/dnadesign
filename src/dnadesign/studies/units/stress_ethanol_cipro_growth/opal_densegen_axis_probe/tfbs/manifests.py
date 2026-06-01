"""Manifest and artifact writers for the DenseGen TFBS learnability probe v1."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from ..constants import X_COLUMN
from .schema import (
    TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES,
    TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
    TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
    TFBS_LEARNABILITY_ORACLE_VERSION,
    TFBS_LEARNABILITY_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from .oracle import TfbsLearnabilityOracleBuild


def row_universe_manifest(
    *,
    candidates: pd.DataFrame,
    densegen_sidecar: pd.DataFrame | None,
    row_universe: Mapping[str, Any],
    active_row_count: int,
) -> dict[str, Any]:
    sidecar_ids = row_universe["sidecar_ids"]
    candidate_ids = row_universe["candidate_ids"]
    sidecar_only_ids = row_universe["sidecar_only_ids"]
    candidate_only_ids = row_universe["candidate_only_ids"]
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.row_universe",
        "candidate_records_path": None,
        "candidate_records_hash": _dataframe_hash(candidates),
        "candidate_records_row_count": int(len(candidates)),
        "candidate_records_schema_hash": _schema_hash(candidates),
        "densegen_sidecar_path": None,
        "densegen_sidecar_hash": None if densegen_sidecar is None else _dataframe_hash(densegen_sidecar),
        "densegen_sidecar_row_count": None if densegen_sidecar is None else int(len(densegen_sidecar)),
        "densegen_sidecar_schema_hash": None if densegen_sidecar is None else _schema_hash(densegen_sidecar),
        "candidate_id_count": int(len(candidate_ids)),
        "sidecar_id_count": int(len(sidecar_ids)),
        "candidate_sidecar_intersection_count": int(len(candidate_ids & sidecar_ids))
        if sidecar_ids
        else int(len(candidate_ids)),
        "sidecar_only_id_count": int(len(sidecar_only_ids)),
        "candidate_only_id_count": int(len(candidate_only_ids)),
        "quality_ok_count": int(active_row_count),
        "active_row_count": int(active_row_count),
        "excluded_row_count_by_reason": {
            "sidecar_only_outlier": int(len(sidecar_only_ids)),
            "candidate_only_missing_sidecar": int(len(candidate_only_ids)),
        },
        "candidate_id_order_hash": _id_order_hash(candidates["id"].astype(str).tolist()),
        "source_mode": row_universe["source_mode"],
    }


def label_manifest(labels: pd.DataFrame, *, algebra: Mapping[str, Any], rates: Mapping[str, Any]) -> dict[str, Any]:
    coordinate_summary = {
        "slot_coordinate": "offset_raw",
        "sequence_length_bp": 60,
        "tfbs_entries_per_active_row": 3,
        "fixed_elements_per_active_row": 2,
        "sigma_core_roles": ["sigma35", "sigma10"],
    }
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.label_manifest",
        "oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
        "label_recipe_hash": TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
        "parser_config_hash": TFBS_LEARNABILITY_LABEL_RECIPE_HASH,
        "row_universe_manifest_hash": None,
        "label_table_path": None,
        "label_table_hash": None,
        "label_table_row_count": int(len(labels)),
        "label_table_schema": list(labels.columns),
        "active_label_families": list(TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES),
        "active_label_names": list(TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES),
        "passive_control_names": [
            "sigma35_variant",
            "sigma10_consensus_identity",
            "spacer_length",
            "sigma35_offset_raw",
            "sigma10_offset_raw",
        ],
        "observed_label_rate_summary": dict(rates),
        "algebraic_consistency_summary": dict(algebra),
        "coordinate_contract_summary": coordinate_summary,
        "known_deviations": [],
    }


def source_hash_manifest(candidates: pd.DataFrame, densegen_sidecar: pd.DataFrame | None) -> dict[str, Any]:
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.source_hash",
        "git_sha": None,
        "uv_lock_hash": None,
        "source_records_path_hash_row_schema": {
            "path": None,
            "hash": _dataframe_hash(candidates),
            "row_count": int(len(candidates)),
            "schema_hash": _schema_hash(candidates),
        },
        "densegen_sidecar_path_hash_row_schema": None
        if densegen_sidecar is None
        else {
            "path": None,
            "hash": _dataframe_hash(densegen_sidecar),
            "row_count": int(len(densegen_sidecar)),
            "schema_hash": _schema_hash(densegen_sidecar),
        },
        "x_contract": {"kind": "opal_candidate_table_column", "column": X_COLUMN},
        "x_column": X_COLUMN,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "sklearn_version": sklearn.__version__,
        "thread_settings": {},
    }


def write_tfbs_learnability_oracle_artifacts(
    build: TfbsLearnabilityOracleBuild,
    out_dir: Path,
) -> TfbsLearnabilityOracleBuild:
    """Write label table and manifests using compact replay-safe defaults."""

    labels_dir = out_dir / "labels"
    manifests_dir = out_dir / "manifests"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    label_table_path = labels_dir / "densegen_tfbs_learnability_positive_v1.parquet"
    build.labels.to_parquet(label_table_path, index=False, compression="zstd")
    label_manifest_payload = dict(build.label_manifest)
    label_manifest_payload.update(
        {
            "row_universe_manifest_hash": _json_payload_hash(build.row_universe_manifest),
            "label_table_path": str(label_table_path),
            "label_table_hash": _file_sha256(label_table_path),
            "label_table_row_count": int(len(build.labels)),
        }
    )
    _write_json(manifests_dir / "row_universe_manifest.json", build.row_universe_manifest)
    _write_json(manifests_dir / "source_hash_manifest.json", build.source_hash_manifest)
    _write_json(manifests_dir / "label_manifest.json", label_manifest_payload)
    return replace(build, label_manifest=label_manifest_payload)


def _dataframe_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame.astype(str), index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _schema_hash(frame: pd.DataFrame) -> str:
    schema = [(column, str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _id_order_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
