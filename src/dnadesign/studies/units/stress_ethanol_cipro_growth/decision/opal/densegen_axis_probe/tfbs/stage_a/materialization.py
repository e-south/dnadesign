"""Stage A materialization for the DenseGen TFBS learnability probe v1."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from ...core.constants import CANDIDATE_RECORDS, DEFAULT_SEED, DENSEGEN_SIDECAR
from ..active_targets import validate_tfbs_learnability_target_set
from ..manifests import write_tfbs_learnability_oracle_artifacts
from ..null_artifacts import write_tfbs_null_artifacts
from ..nulls import (
    TFBS_SLOT_EVENT_COLUMNS,
    TfbsNullBuild,
    build_tfbs_family_content_matched_null,
    build_tfbs_slot_geometry_count_matched_null,
    build_tfbs_slot_position_count_fixed_shuffled_null,
)
from ..oracle import build_tfbs_learnability_oracle, validate_observed_label_rates
from ..profiles import is_count_fixed_slot_position_profile_id, resolve_tfbs_target_profile
from ..retention import (
    DEFAULT_RETENTION_MAX_ESTIMATED_BYTES,
    DEFAULT_TFBS_STAGE_ROUNDS,
    DEFAULT_TFBS_STAGE_SELECTION_K,
    TfbsRetentionPolicy,
    estimate_tfbs_learnability_retention,
)
from .manifests import (
    attach_source_file_fingerprints,
    build_pairing_manifest,
    build_stage_manifest,
)


@dataclass(frozen=True)
class TfbsStageAConfig:
    """Inputs and gates for Stage A label/null/preflight materialization."""

    candidate_records_path: Path = CANDIDATE_RECORDS
    densegen_sidecar_path: Path | None = DENSEGEN_SIDECAR
    run_root: Path = Path()
    seed: int = DEFAULT_SEED
    rounds: int = DEFAULT_TFBS_STAGE_ROUNDS
    selection_k: int = DEFAULT_TFBS_STAGE_SELECTION_K
    max_estimated_bytes: int = DEFAULT_RETENTION_MAX_ESTIMATED_BYTES
    fail_if_estimate_exceeds: bool = True
    enforce_live_label_rate_sanity: bool = True
    replace_run_root: bool = False
    label_names: tuple[str, ...] = ()
    target_profile_id: str | None = None


@dataclass(frozen=True)
class TfbsStageAResult:
    """Materialized Stage A output paths and status."""

    status: str
    run_root: Path
    positive_label_table_path: Path
    row_universe_manifest_path: Path
    label_manifest_path: Path
    source_hash_manifest_path: Path
    retention_estimate_path: Path
    pairing_manifest_path: Path
    stage_a_manifest_path: Path
    null_artifact_count: int
    sentinel_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_root": str(self.run_root),
            "positive_label_table_path": str(self.positive_label_table_path),
            "row_universe_manifest_path": str(self.row_universe_manifest_path),
            "label_manifest_path": str(self.label_manifest_path),
            "source_hash_manifest_path": str(self.source_hash_manifest_path),
            "retention_estimate_path": str(self.retention_estimate_path),
            "pairing_manifest_path": str(self.pairing_manifest_path),
            "stage_a_manifest_path": str(self.stage_a_manifest_path),
            "null_artifact_count": int(self.null_artifact_count),
            "sentinel_labels": list(self.sentinel_labels),
        }


def materialize_tfbs_stage_a(config: TfbsStageAConfig) -> TfbsStageAResult:
    """Write Stage A positive labels, sentinel nulls, manifests, and retention estimate."""

    cfg = _normalize_config(config)
    candidates = _read_candidate_records(cfg.candidate_records_path, densegen_sidecar_path=cfg.densegen_sidecar_path)
    densegen_sidecar = None if cfg.densegen_sidecar_path is None else _read_densegen_sidecar(cfg.densegen_sidecar_path)
    oracle = build_tfbs_learnability_oracle(candidates, densegen_sidecar=densegen_sidecar)
    if cfg.enforce_live_label_rate_sanity:
        label_rate_sanity = validate_observed_label_rates(oracle.labels)
    else:
        label_rate_sanity = {"status": "SKIPPED", "reason": "enforce_live_label_rate_sanity=false"}

    retention_policy = TfbsRetentionPolicy(
        rounds=cfg.rounds,
        selection_k=cfg.selection_k,
        max_estimated_bytes=cfg.max_estimated_bytes,
        fail_if_estimate_exceeds=cfg.fail_if_estimate_exceeds,
    )
    retention_estimate = estimate_tfbs_learnability_retention(
        candidate_row_count=len(oracle.labels),
        policy=retention_policy,
    )
    target_profile = resolve_tfbs_target_profile(target_profile_id=cfg.target_profile_id, label_names=cfg.label_names)
    null_builds = _build_sentinel_nulls(
        oracle.labels,
        seed=cfg.seed,
        label_names=cfg.label_names,
        target_profile_id=target_profile.profile_id,
    )
    _prepare_run_root(cfg.run_root, replace=cfg.replace_run_root)

    oracle = attach_source_file_fingerprints(
        oracle,
        candidates=candidates,
        densegen_sidecar=densegen_sidecar,
        candidate_records_path=cfg.candidate_records_path,
        densegen_sidecar_path=cfg.densegen_sidecar_path,
    )
    written_oracle = write_tfbs_learnability_oracle_artifacts(oracle, cfg.run_root)
    written_nulls = [write_tfbs_null_artifacts(build, cfg.run_root) for build in null_builds]

    manifests_dir = cfg.run_root / "manifests"
    retention_estimate_path = manifests_dir / "retention_estimate.json"
    _write_json(retention_estimate_path, retention_estimate)
    pairing_manifest = build_pairing_manifest(
        positive_label_manifest=written_oracle.label_manifest,
        retention_estimate=retention_estimate,
        written_nulls=written_nulls,
        seed=cfg.seed,
        target_profile=target_profile.to_manifest(),
    )
    pairing_manifest_path = manifests_dir / "pairing_manifest.json"
    _write_json(pairing_manifest_path, pairing_manifest)
    stage_manifest = build_stage_manifest(
        run_root=cfg.run_root,
        seed=cfg.seed,
        rounds=cfg.rounds,
        selection_k=cfg.selection_k,
        label_rate_sanity=label_rate_sanity,
        positive_label_manifest=written_oracle.label_manifest,
        retention_estimate=retention_estimate,
        written_nulls=written_nulls,
        pairing_manifest_path=pairing_manifest_path,
        retention_estimate_path=retention_estimate_path,
        sentinel_labels=cfg.label_names,
        target_profile=target_profile.to_manifest(),
    )
    stage_manifest_path = manifests_dir / "tfbs_stage_a_manifest.json"
    _write_json(stage_manifest_path, stage_manifest)

    return TfbsStageAResult(
        status="PASS",
        run_root=cfg.run_root,
        positive_label_table_path=Path(written_oracle.label_manifest["label_table_path"]),
        row_universe_manifest_path=manifests_dir / "row_universe_manifest.json",
        label_manifest_path=manifests_dir / "label_manifest.json",
        source_hash_manifest_path=manifests_dir / "source_hash_manifest.json",
        retention_estimate_path=retention_estimate_path,
        pairing_manifest_path=pairing_manifest_path,
        stage_a_manifest_path=stage_manifest_path,
        null_artifact_count=len(written_nulls),
        sentinel_labels=cfg.label_names,
    )


def _normalize_config(config: TfbsStageAConfig) -> TfbsStageAConfig:
    run_root = Path(config.run_root)
    if str(run_root) in {"", "."}:
        raise ValueError("Stage A run_root must be explicit")
    candidate_records_path = Path(config.candidate_records_path)
    densegen_sidecar_path = None if config.densegen_sidecar_path is None else Path(config.densegen_sidecar_path)
    if not candidate_records_path.exists():
        raise FileNotFoundError(f"candidate records not found: {candidate_records_path}")
    if densegen_sidecar_path is not None and not densegen_sidecar_path.exists():
        raise FileNotFoundError(f"DenseGen sidecar not found: {densegen_sidecar_path}")
    if config.seed < 0:
        raise ValueError("Stage A seed must be non-negative")
    if config.rounds <= 0:
        raise ValueError("Stage A rounds must be positive")
    if config.selection_k <= 0:
        raise ValueError("Stage A selection_k must be positive")
    if config.max_estimated_bytes <= 0:
        raise ValueError("Stage A max_estimated_bytes must be positive")
    target_profile = resolve_tfbs_target_profile(
        target_profile_id=config.target_profile_id,
        label_names=tuple(config.label_names),
    )
    label_names = validate_tfbs_learnability_target_set(target_profile.label_names)
    return replace(
        config,
        candidate_records_path=candidate_records_path,
        densegen_sidecar_path=densegen_sidecar_path,
        run_root=run_root,
        label_names=label_names,
        target_profile_id=target_profile.profile_id,
    )


def _read_candidate_records(path: Path, *, densegen_sidecar_path: Path | None) -> pd.DataFrame:
    columns = ["id", "sequence"]
    if densegen_sidecar_path is None:
        columns.append("densegen__used_tfbs_detail")
    return pd.read_parquet(path, columns=_required_columns(path, columns, surface="candidate records"))


def _read_densegen_sidecar(path: Path) -> pd.DataFrame:
    columns = [
        "id",
        "densegen__used_tfbs_detail",
        "densegen__plan",
        "densegen__required_regulators",
        "densegen__sampling_library_hash",
    ]
    available = _required_columns(path, ("id", "densegen__used_tfbs_detail"), surface="DenseGen sidecar")
    optional = [column for column in columns if column in _parquet_schema_names(path)]
    return pd.read_parquet(path, columns=list(dict.fromkeys([*available, *optional])))


def _required_columns(path: Path, columns: list[str] | tuple[str, ...], *, surface: str) -> list[str]:
    names = _parquet_schema_names(path)
    missing = [column for column in columns if column not in names]
    if missing:
        raise ValueError(f"{surface} missing required column(s): {missing}")
    return list(columns)


def _parquet_schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _build_sentinel_nulls(
    labels: pd.DataFrame,
    *,
    seed: int,
    label_names: tuple[str, ...],
    target_profile_id: str,
) -> list[TfbsNullBuild]:
    builds: list[TfbsNullBuild] = []
    for label_name in label_names:
        if is_count_fixed_slot_position_profile_id(target_profile_id):
            builds.append(build_tfbs_slot_position_count_fixed_shuffled_null(labels, label_name=label_name, seed=seed))
        elif label_name in TFBS_SLOT_EVENT_COLUMNS:
            builds.append(build_tfbs_slot_geometry_count_matched_null(labels, label_name=label_name, seed=seed))
        else:
            builds.append(build_tfbs_family_content_matched_null(labels, label_name=label_name, seed=seed))
    return builds


def _prepare_run_root(run_root: Path, *, replace: bool) -> None:
    if run_root.exists():
        if replace:
            shutil.rmtree(run_root)
        elif any(run_root.iterdir()):
            raise RuntimeError(f"Stage A run_root already exists and is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
