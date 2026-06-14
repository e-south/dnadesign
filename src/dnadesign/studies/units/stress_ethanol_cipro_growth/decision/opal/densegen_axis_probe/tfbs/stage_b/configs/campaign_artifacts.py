"""Per-campaign artifact writers for Stage B TFBS config generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from ...active_targets import tfbs_learnability_active_target_spec
from ...stage_a.manifests import file_sha256
from ..commands import opal_ingest_command, opal_validate_command
from ..io import read_stage_b_label_table, write_stage_b_initial_label_input
from ..layout import TfbsStageBLayout
from ..payloads import tfbs_stage_b_campaign_config_payload, write_tfbs_stage_b_plot_config
from ..semantics import TfbsStageBRunIdentity
from .contracts import TfbsStageBConfig


def write_campaign_artifacts(
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
    candidate_scope_path: Path,
    candidate_scope_metadata: Mapping[str, Any],
    initial_ids: Sequence[str],
    initial_seed_context: str,
    initial_seed_source_role: str,
    candidate_identity: pd.DataFrame,
    stage_a_planned_config_hash: str,
    baserender_metadata_records_path: Path | None,
) -> dict[str, Any]:
    """Write one campaign config, plot config, and round-zero label input."""

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
        candidate_scope_path=candidate_scope_path,
        candidate_scope_metadata=candidate_scope_metadata,
        initial_seed_context=initial_seed_context,
        initial_seed_source_role=initial_seed_source_role,
        baserender_metadata_records_path=baserender_metadata_records_path,
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
        "null_version": pair_row.get("null_version"),
        "null_control_role": pair_row.get("null_control_role"),
        "negative_control_claim_status": pair_row.get("negative_control_claim_status"),
        "split_id": cfg.split_id,
        "seed": int(cfg.seed),
        "rounds": int(cfg.rounds),
        "selection_k": int(cfg.selection_k),
        "initial_seed_policy": cfg.initial_seed_policy,
        "initial_seed_context": initial_seed_context,
        "initial_seed_source_role": initial_seed_source_role,
        "initial_label_ids_hash": initial_ids_hash(initial_ids),
        "config_path": str(config_path),
        "campaign_config_hash": file_sha256(config_path),
        "stage_a_planned_campaign_config_hash": stage_a_planned_config_hash,
        "initial_label_input_path": str(label_input_path),
        "initial_label_input_hash": file_sha256(label_input_path),
        "label_table_path": str(label_table_path),
        "label_table_hash": label_table_hash,
        "candidate_scope_path": str(candidate_scope_path),
        "candidate_scope_hash": file_sha256(candidate_scope_path),
        **dict(candidate_scope_metadata),
        "records_path": str(layout.records_path),
        "records_hash": file_sha256(layout.records_path),
        "label_sidecar_path": str(layout.dataset_dir / layout.sidecar_relative_path(run_key)),
        "label_sidecar_relative_path": layout.sidecar_relative_path(run_key),
        "validate_command": opal_validate_command(config_path),
        "ingest_round0_command": opal_ingest_command(config_path, label_input_path, round_index=0),
    }


def initial_ids_hash(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
