"""Stage A manifest builders for DenseGen TFBS learnability artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..active_targets import tfbs_learnability_active_target_spec
from ..nulls import TfbsNullBuild
from ..schema import (
    TFBS_LEARNABILITY_ORACLE_VERSION,
    TFBS_LEARNABILITY_SCHEMA_VERSION,
)


def attach_source_file_fingerprints(
    oracle: Any,
    *,
    candidates: pd.DataFrame,
    densegen_sidecar: pd.DataFrame | None,
    candidate_records_path: Path,
    densegen_sidecar_path: Path | None,
) -> Any:
    """Replace dataframe-only source hashes with actual source-file hashes."""

    candidate_file = _file_fingerprint(candidate_records_path, candidates)
    sidecar_file = None
    if densegen_sidecar_path is not None and densegen_sidecar is not None:
        sidecar_file = _file_fingerprint(densegen_sidecar_path, densegen_sidecar)

    row_universe = dict(oracle.row_universe_manifest)
    row_universe.update(
        {
            "candidate_records_path": str(candidate_records_path),
            "candidate_records_hash": candidate_file["hash"],
            "candidate_records_row_count": candidate_file["row_count"],
            "candidate_records_schema_hash": candidate_file["schema_hash"],
            "densegen_sidecar_path": None if sidecar_file is None else str(densegen_sidecar_path),
            "densegen_sidecar_hash": None if sidecar_file is None else sidecar_file["hash"],
            "densegen_sidecar_row_count": None if sidecar_file is None else sidecar_file["row_count"],
            "densegen_sidecar_schema_hash": None if sidecar_file is None else sidecar_file["schema_hash"],
            "source_hash_kind": "sha256_file",
        }
    )
    source_hash = dict(oracle.source_hash_manifest)
    source_hash.update(
        {
            "source_records_path_hash_row_schema": {
                "path": str(candidate_records_path),
                **candidate_file,
            },
            "densegen_sidecar_path_hash_row_schema": None
            if sidecar_file is None
            else {
                "path": str(densegen_sidecar_path),
                **sidecar_file,
            },
            "hash_kind": "sha256_file",
        }
    )
    return replace(oracle, row_universe_manifest=row_universe, source_hash_manifest=source_hash)


def build_pairing_manifest(
    *,
    positive_label_manifest: Mapping[str, Any],
    retention_estimate: Mapping[str, Any],
    written_nulls: list[TfbsNullBuild],
    seed: int,
) -> dict[str, Any]:
    """Build manifest-backed positive/null pair relationships for Stage B sentinels."""

    retention_hash = str(retention_estimate["retention_policy_hash"])
    pairs = []
    for build in written_nulls:
        report = build.null_viability_report
        label_name = str(report["label_name"])
        target_spec = tfbs_learnability_active_target_spec(label_name).to_dict()
        pairs.append(
            {
                "label_name": label_name,
                "split_id": "random_id",
                "seed": int(seed),
                "positive_oracle_role": "positive",
                "null_oracle_role": "matched_null",
                "positive_oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
                "null_version": report["null_version"],
                "positive_label_table_path": positive_label_manifest["label_table_path"],
                "positive_label_table_hash": positive_label_manifest["label_table_hash"],
                "null_label_table_path": report["null_label_table_path"],
                "null_label_table_hash": report["null_label_table_hash"],
                "positive_campaign_config_hash": _planned_campaign_hash(
                    label_name=label_name,
                    oracle_role="positive",
                    seed=seed,
                    target_spec=target_spec,
                    retention_policy_hash=retention_hash,
                ),
                "null_campaign_config_hash": _planned_campaign_hash(
                    label_name=label_name,
                    oracle_role="matched_null",
                    seed=seed,
                    target_spec=target_spec,
                    retention_policy_hash=retention_hash,
                ),
                "campaign_config_hash_kind": "stage_a_planned_sentinel_contract_hash",
                "retention_policy_hash": retention_hash,
            }
        )
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.pairing_manifest",
        "positive_oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
        "retention_policy_hash": retention_hash,
        "pairing_scope": "stage_b_sentinel_initial",
        "pairs": sorted(pairs, key=lambda row: row["label_name"]),
    }


def build_stage_manifest(
    *,
    run_root: Path,
    seed: int,
    rounds: int,
    selection_k: int,
    label_rate_sanity: Mapping[str, Any],
    positive_label_manifest: Mapping[str, Any],
    retention_estimate: Mapping[str, Any],
    written_nulls: list[TfbsNullBuild],
    pairing_manifest_path: Path,
    retention_estimate_path: Path,
    sentinel_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Build the Stage A summary manifest after artifact files are written."""

    null_artifacts = []
    for build in written_nulls:
        report = build.null_viability_report
        label_table_path = Path(report["null_label_table_path"])
        report_path = (
            label_table_path.parent.parent / "manifests" / f"{label_table_path.stem}.null_viability_report.json"
        )
        null_artifacts.append(
            {
                "label_name": report["label_name"],
                "null_version": report["null_version"],
                "viability_status": report["viability_status"],
                "null_label_table_path": str(label_table_path),
                "null_label_table_hash": report["null_label_table_hash"],
                "null_viability_report_path": str(report_path),
                "null_viability_report_hash": file_sha256(report_path),
            }
        )
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.stage_a_manifest",
        "stage": "A",
        "status": "PASS",
        "run_root": str(run_root),
        "seed": int(seed),
        "rounds": int(rounds),
        "selection_k": int(selection_k),
        "positive_oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
        "positive_label_table_path": positive_label_manifest["label_table_path"],
        "positive_label_table_hash": positive_label_manifest["label_table_hash"],
        "sentinel_labels": list(sentinel_labels),
        "label_rate_sanity": dict(label_rate_sanity),
        "retention_estimate_path": str(retention_estimate_path),
        "retention_policy_hash": retention_estimate["retention_policy_hash"],
        "retention_status": retention_estimate["status"],
        "pairing_manifest_path": str(pairing_manifest_path),
        "pairing_manifest_hash": file_sha256(pairing_manifest_path),
        "null_artifacts": sorted(null_artifacts, key=lambda row: row["label_name"]),
    }


def file_sha256(path: Path) -> str:
    """Hash a materialized file for replay manifests."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "hash": file_sha256(path),
        "row_count": int(len(frame)),
        "schema_hash": _schema_hash(frame),
    }


def _planned_campaign_hash(
    *,
    label_name: str,
    oracle_role: str,
    seed: int,
    target_spec: Mapping[str, Any],
    retention_policy_hash: str,
) -> str:
    return _payload_hash(
        {
            "stage": "B_sentinel_expected_label_target",
            "label_name": label_name,
            "oracle_role": oracle_role,
            "split_id": "random_id",
            "seed": int(seed),
            "positive_oracle_version": TFBS_LEARNABILITY_ORACLE_VERSION,
            "target_spec": dict(target_spec),
            "retention_policy_hash": retention_policy_hash,
        }
    )


def _schema_hash(frame: pd.DataFrame) -> str:
    schema = [(column, str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
