"""Materialize Stage B OPAL configs for DenseGen TFBS learnability sentinels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from ...active_targets import (
    tfbs_learnability_active_target_spec,
)
from ...schema import (
    TFBS_LEARNABILITY_SCHEMA_VERSION,
)
from ...stage_a.manifests import file_sha256
from ..collection import stage_b_collection_manifest
from ..commands import opal_ingest_command, opal_validate_command
from ..io import (
    prepare_stage_b_out_dir,
    read_stage_b_candidate_identity,
    read_stage_b_json,
    read_stage_b_label_table,
    write_stage_b_candidate_scope,
    write_stage_b_initial_label_input,
    write_stage_b_json,
    write_stage_b_records_reference,
)
from ..layout import TfbsStageBLayout
from ..payloads import tfbs_stage_b_campaign_config_payload, write_tfbs_stage_b_plot_config
from ..seed import (
    select_tfbs_stage_b_initial_ids,
)
from ..semantics import (
    TFBS_STAGE_B_ORACLE_ROLES,
    TFBS_STAGE_B_RETENTION_MODE,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_STAGE,
    TfbsStageBRunIdentity,
    stage_b_selection_budget_mode,
)
from .contracts import TfbsStageBConfig, TfbsStageBResult
from .validation import normalize_config, skipped_validation, validate_campaign_configs, validate_stage_a_inputs


def materialize_tfbs_stage_b_sentinel_configs(config: TfbsStageBConfig) -> TfbsStageBResult:
    """Write and optionally validate Stage B sentinel OPAL configs from Stage A artifacts."""

    cfg = normalize_config(config)
    layout = TfbsStageBLayout(
        cfg.out_dir or (cfg.stage_a_run_root / "stage_b_sentinel_configs"),
        cfg.split_id,
        cfg.seed,
    )
    prepare_stage_b_out_dir(layout.out_dir, replace=cfg.replace_out_dir)

    stage_a = read_stage_b_json(cfg.stage_a_run_root / "manifests" / "tfbs_stage_a_manifest.json")
    pairing = read_stage_b_json(cfg.stage_a_run_root / "manifests" / "pairing_manifest.json")
    retention = read_stage_b_json(cfg.stage_a_run_root / "manifests" / "retention_estimate.json")
    row_universe = read_stage_b_json(cfg.stage_a_run_root / "manifests" / "row_universe_manifest.json")
    validate_stage_a_inputs(
        stage_a=stage_a,
        pairing=pairing,
        retention=retention,
        requested_labels=cfg.label_names,
        seed=cfg.seed,
        split_id=cfg.split_id,
    )

    candidate_records_path = Path(str(row_universe["candidate_records_path"]))
    positive_labels = read_stage_b_label_table(Path(str(stage_a["positive_label_table_path"])))
    candidate_identity = read_stage_b_candidate_identity(candidate_records_path)
    write_stage_b_records_reference(candidate_records_path, layout.records_path)
    write_stage_b_candidate_scope(layout.candidate_scope_path, positive_labels["id"].astype(str).tolist())

    campaigns: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pair_rows = _pair_rows_by_label(pairing, cfg.label_names)
    for label_name in cfg.label_names:
        pair_row = pair_rows[label_name]
        positive = _write_campaign_artifacts(
            cfg=cfg,
            layout=layout,
            stage_a=stage_a,
            pairing_manifest_path=cfg.stage_a_run_root / "manifests" / "pairing_manifest.json",
            retention=retention,
            pair_row=pair_row,
            label_name=label_name,
            oracle_role="positive",
            label_table_path=Path(str(pair_row["positive_label_table_path"])),
            label_table_hash=str(pair_row["positive_label_table_hash"]),
            candidate_identity=candidate_identity,
            stage_a_planned_config_hash=str(pair_row["positive_campaign_config_hash"]),
        )
        null = _write_campaign_artifacts(
            cfg=cfg,
            layout=layout,
            stage_a=stage_a,
            pairing_manifest_path=cfg.stage_a_run_root / "manifests" / "pairing_manifest.json",
            retention=retention,
            pair_row=pair_row,
            label_name=label_name,
            oracle_role="matched_null",
            label_table_path=Path(str(pair_row["null_label_table_path"])),
            label_table_hash=str(pair_row["null_label_table_hash"]),
            candidate_identity=candidate_identity,
            stage_a_planned_config_hash=str(pair_row["null_campaign_config_hash"]),
        )
        campaigns.extend([positive, null])
        pairs.append(_build_pair_manifest_row(label_name=label_name, positive=positive, null=null, pair_row=pair_row))

    validation = (
        validate_campaign_configs(campaigns, cfg=cfg) if cfg.validate_configs else skipped_validation(campaigns)
    )
    collection_manifest = stage_b_collection_manifest(split_id=cfg.split_id, seed=cfg.seed)
    write_stage_b_json(layout.collection_manifest_path, collection_manifest)
    manifest = _build_config_manifest(
        cfg=cfg,
        layout=layout,
        stage_a=stage_a,
        retention=retention,
        campaigns=campaigns,
        pairs=pairs,
        validation=validation,
    )
    write_stage_b_json(layout.config_manifest_path, manifest)
    if validation["status"] != "PASS" and cfg.validate_configs:
        raise RuntimeError(f"Stage B sentinel OPAL config validation failed; see {layout.config_manifest_path}")
    return TfbsStageBResult(
        status="PASS",
        out_dir=layout.out_dir,
        config_manifest_path=layout.config_manifest_path,
        collection_manifest_path=layout.collection_manifest_path,
        campaign_count=len(campaigns),
        validation_status=str(validation["status"]),
    )


def _write_campaign_artifacts(
    *,
    cfg: TfbsStageBConfig,
    layout: TfbsStageBLayout,
    stage_a: Mapping[str, Any],
    pairing_manifest_path: Path,
    retention: Mapping[str, Any],
    pair_row: Mapping[str, Any],
    label_name: str,
    oracle_role: str,
    label_table_path: Path,
    label_table_hash: str,
    candidate_identity: pd.DataFrame,
    stage_a_planned_config_hash: str,
) -> dict[str, Any]:
    target = tfbs_learnability_active_target_spec(label_name)
    run_identity = TfbsStageBRunIdentity(
        label_name=label_name,
        oracle_role=oracle_role,
        split_id=cfg.split_id,
        seed=cfg.seed,
    )
    run_key = run_identity.run_key
    workdir = layout.campaign_workdir(run_key)
    config_path = layout.campaign_config_path(run_key)
    label_input_path = layout.initial_label_input_path(run_key)
    plot_config_path = layout.campaign_plot_config_path(run_key)
    label_table = read_stage_b_label_table(label_table_path)
    initial_ids = select_tfbs_stage_b_initial_ids(
        label_table,
        label_name=label_name,
        initial_label_count=cfg.initial_label_count,
        seed=cfg.seed,
        policy=cfg.initial_seed_policy,
        seed_context=run_key,
    )
    write_stage_b_initial_label_input(
        label_input_path,
        label_table,
        label_name=label_name,
        initial_ids=initial_ids,
        candidate_identity=candidate_identity,
    )
    write_tfbs_stage_b_plot_config(plot_config_path, label_name=label_name, target_display=target.target_description)
    config_payload = tfbs_stage_b_campaign_config_payload(
        cfg=cfg,
        layout=layout,
        workdir=workdir,
        run_key=run_key,
        stage_a=stage_a,
        pairing_manifest_path=pairing_manifest_path,
        retention=retention,
        pair_row=pair_row,
        label_name=label_name,
        oracle_role=oracle_role,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    return {
        "campaign_key": run_key,
        "campaign_slug": config_payload["campaign"]["slug"],
        "label_name": label_name,
        "label_family_id": target.label_family_id,
        "target_kind": target.target_kind,
        "oracle_role": oracle_role,
        "split_id": cfg.split_id,
        "seed": int(cfg.seed),
        "rounds": int(cfg.rounds),
        "selection_k": int(cfg.selection_k),
        "initial_seed_policy": cfg.initial_seed_policy,
        "config_path": str(config_path),
        "campaign_config_hash": file_sha256(config_path),
        "stage_a_planned_campaign_config_hash": stage_a_planned_config_hash,
        "initial_label_input_path": str(label_input_path),
        "initial_label_input_hash": file_sha256(label_input_path),
        "label_table_path": str(label_table_path),
        "label_table_hash": label_table_hash,
        "candidate_scope_path": str(layout.candidate_scope_path),
        "candidate_scope_hash": file_sha256(layout.candidate_scope_path),
        "records_path": str(layout.records_path),
        "records_hash": file_sha256(layout.records_path),
        "label_sidecar_path": str(layout.dataset_dir / layout.sidecar_relative_path(run_key)),
        "label_sidecar_relative_path": layout.sidecar_relative_path(run_key),
        "validate_command": opal_validate_command(config_path),
        "ingest_round0_command": opal_ingest_command(config_path, label_input_path, round_index=0),
    }


def _build_pair_manifest_row(
    *,
    label_name: str,
    positive: Mapping[str, Any],
    null: Mapping[str, Any],
    pair_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "label_name": label_name,
        "split_id": positive["split_id"],
        "seed": int(positive["seed"]),
        "positive_oracle_role": TFBS_STAGE_B_ORACLE_ROLES[0],
        "null_oracle_role": TFBS_STAGE_B_ORACLE_ROLES[1],
        "positive_oracle_version": pair_row["positive_oracle_version"],
        "null_version": pair_row["null_version"],
        "retention_policy_hash": pair_row["retention_policy_hash"],
        "initial_seed_policy": positive["initial_seed_policy"],
        "positive_campaign_key": positive["campaign_key"],
        "null_campaign_key": null["campaign_key"],
        "positive_campaign_config_path": positive["config_path"],
        "positive_campaign_config_hash": positive["campaign_config_hash"],
        "null_campaign_config_path": null["config_path"],
        "null_campaign_config_hash": null["campaign_config_hash"],
        "positive_initial_label_input_path": positive["initial_label_input_path"],
        "positive_initial_label_input_hash": positive["initial_label_input_hash"],
        "null_initial_label_input_path": null["initial_label_input_path"],
        "null_initial_label_input_hash": null["initial_label_input_hash"],
        "positive_label_table_path": positive["label_table_path"],
        "positive_label_table_hash": positive["label_table_hash"],
        "null_label_table_path": null["label_table_path"],
        "null_label_table_hash": null["label_table_hash"],
    }


def _build_config_manifest(
    *,
    cfg: TfbsStageBConfig,
    layout: TfbsStageBLayout,
    stage_a: Mapping[str, Any],
    retention: Mapping[str, Any],
    campaigns: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": f"{TFBS_LEARNABILITY_SCHEMA_VERSION}.stage_b_sentinel_config_manifest",
        "stage": TFBS_STAGE_B_STAGE,
        "status": "PASS" if str(validation.get("status")) in {"PASS", "SKIPPED"} else "FAIL",
        "scope": TFBS_STAGE_B_SCOPE,
        "stage_a_manifest_path": str(cfg.stage_a_run_root / "manifests" / "tfbs_stage_a_manifest.json"),
        "pairing_manifest_path": str(cfg.stage_a_run_root / "manifests" / "pairing_manifest.json"),
        "retention_estimate_path": str(cfg.stage_a_run_root / "manifests" / "retention_estimate.json"),
        "collection_manifest_path": str(layout.collection_manifest_path),
        "collection_manifest_hash": file_sha256(layout.collection_manifest_path),
        "positive_oracle_version": stage_a["positive_oracle_version"],
        "retention_policy_hash": retention["retention_policy_hash"],
        "retention_mode": TFBS_STAGE_B_RETENTION_MODE,
        "split_id": cfg.split_id,
        "seed": int(cfg.seed),
        "rounds": int(cfg.rounds),
        "selection_k": int(cfg.selection_k),
        "selection_tie_handling": cfg.selection_tie_handling,
        "selection_budget_mode": stage_b_selection_budget_mode(tie_handling=cfg.selection_tie_handling),
        "initial_label_count": int(cfg.initial_label_count),
        "initial_seed_policy": cfg.initial_seed_policy,
        "sentinel_labels": list(cfg.label_names),
        "campaign_count": int(len(campaigns)),
        "scratch_dataset": layout.dataset,
        "scratch_usr_dir": str(layout.scratch_usr_dir),
        "records_path": str(layout.records_path),
        "records_hash": file_sha256(layout.records_path),
        "candidate_scope_path": str(layout.candidate_scope_path),
        "candidate_scope_hash": file_sha256(layout.candidate_scope_path),
        "validation": dict(validation),
        "pairs": sorted((dict(row) for row in pairs), key=lambda row: row["label_name"]),
        "campaigns": sorted((dict(row) for row in campaigns), key=lambda row: row["campaign_key"]),
    }


def _pair_rows_by_label(pairing: Mapping[str, Any], label_names: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    rows = pairing.get("pairs")
    if not isinstance(rows, list):
        raise ValueError("pairing manifest pairs must be a list")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("pairing manifest pair entries must be mappings")
        label = str(row.get("label_name"))
        if label in label_names:
            out[label] = row
    missing = sorted(set(label_names) - set(out))
    if missing:
        raise ValueError(f"pairing manifest missing label(s): {missing}")
    return out
