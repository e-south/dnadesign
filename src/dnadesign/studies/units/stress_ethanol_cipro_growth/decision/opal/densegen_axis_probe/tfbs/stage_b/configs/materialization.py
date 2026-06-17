"""Materialize Stage B OPAL configs for DenseGen TFBS learnability sentinels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...profiles import resolve_tfbs_target_profile
from ...schema import TFBS_LEARNABILITY_SCHEMA_VERSION
from ...stage_a.manifests import file_sha256, tfbs_stage_a_null_permutation_seed_context
from ..collection import stage_b_collection_manifest
from ..io import (
    prepare_stage_b_out_dir,
    read_stage_b_candidate_identity,
    read_stage_b_json,
    read_stage_b_label_table,
    write_stage_b_candidate_scope,
    write_stage_b_json,
    write_stage_b_records_reference,
)
from ..layout import TfbsStageBLayout
from ..seed import tfbs_stage_b_shared_initial_seed_context
from ..semantics import (
    TFBS_STAGE_B_ORACLE_ROLES,
    TFBS_STAGE_B_RETENTION_MODE,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_STAGE,
    stage_b_selection_budget_mode,
)
from .campaign_artifacts import write_campaign_artifacts
from .contracts import TfbsStageBConfig, TfbsStageBResult
from .scopes import (
    control_pair_label,
    control_role_display_label,
    materialize_label_scope_artifacts,
    select_shared_initial_ids,
    uses_count_fixed_scope,
)
from .validation import (
    normalize_config,
    pair_rows_by_label,
    skipped_validation,
    validate_campaign_configs,
    validate_stage_a_inputs,
)


def materialize_tfbs_stage_b_sentinel_configs(config: TfbsStageBConfig) -> TfbsStageBResult:
    """Write and optionally validate Stage B sentinel OPAL configs from Stage A artifacts."""

    cfg = normalize_config(config)
    target_profile = resolve_tfbs_target_profile(target_profile_id=cfg.target_profile_id, label_names=cfg.label_names)
    layout = TfbsStageBLayout(
        cfg.out_dir or (cfg.stage_a_run_root / "stage_b_sentinel_configs"),
        cfg.split_id,
        cfg.seed,
    )
    prepare_stage_b_out_dir(
        layout.out_dir,
        replace=cfg.replace_out_dir,
        refresh_existing_execution_state=cfg.refresh_existing_execution_state,
    )

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
        target_profile_id=target_profile.profile_id,
    )

    candidate_records_path = Path(str(row_universe["candidate_records_path"]))
    densegen_sidecar_path = (
        Path(str(row_universe["densegen_sidecar_path"])) if row_universe.get("densegen_sidecar_path") else None
    )
    positive_labels = read_stage_b_label_table(Path(str(stage_a["positive_label_table_path"])))
    candidate_identity = read_stage_b_candidate_identity(candidate_records_path)
    write_stage_b_records_reference(candidate_records_path, layout.records_path)
    write_stage_b_candidate_scope(layout.candidate_scope_path, positive_labels["id"].astype(str).tolist())

    campaigns: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pair_rows = pair_rows_by_label(pairing, cfg.label_names)
    for label_name in cfg.label_names:
        pair_row = pair_rows[label_name]
        positive_label_table = read_stage_b_label_table(Path(str(pair_row["positive_label_table_path"])))
        null_label_table = read_stage_b_label_table(Path(str(pair_row["null_label_table_path"])))
        scope_artifacts = materialize_label_scope_artifacts(
            layout=layout,
            label_name=label_name,
            target_profile_id=target_profile.profile_id,
            positive_label_table=positive_label_table,
            null_label_table=null_label_table,
            pair_row=pair_row,
        )
        initial_seed_context = tfbs_stage_b_shared_initial_seed_context(
            label_name=label_name,
            split_id=cfg.split_id,
            seed=cfg.seed,
        )
        shared_initial_ids = select_shared_initial_ids(
            cfg=cfg,
            label_name=label_name,
            positive_label_table_path=Path(str(scope_artifacts["positive_label_table_path"])),
            null_label_table_path=Path(str(scope_artifacts["null_label_table_path"])),
            target_profile_id=target_profile.profile_id,
            initial_seed_context=initial_seed_context,
        )
        positive = write_campaign_artifacts(
            cfg=cfg,
            layout=layout,
            stage_a=stage_a,
            pairing_manifest_path=cfg.stage_a_run_root / "manifests" / "pairing_manifest.json",
            retention=retention,
            pair_row=pair_row,
            label_name=label_name,
            oracle_role="positive",
            label_table_path=Path(str(scope_artifacts["positive_label_table_path"])),
            label_table_hash=str(scope_artifacts["positive_label_table_hash"]),
            candidate_scope_path=Path(str(scope_artifacts["candidate_scope_path"])),
            candidate_scope_metadata=scope_artifacts["candidate_scope_metadata"],
            initial_ids=shared_initial_ids,
            initial_seed_context=initial_seed_context,
            initial_seed_source_role="positive",
            candidate_identity=candidate_identity,
            stage_a_planned_config_hash=str(pair_row["positive_campaign_config_hash"]),
            baserender_metadata_records_path=densegen_sidecar_path,
        )
        null = write_campaign_artifacts(
            cfg=cfg,
            layout=layout,
            stage_a=stage_a,
            pairing_manifest_path=cfg.stage_a_run_root / "manifests" / "pairing_manifest.json",
            retention=retention,
            pair_row=pair_row,
            label_name=label_name,
            oracle_role="matched_null",
            label_table_path=Path(str(scope_artifacts["null_label_table_path"])),
            label_table_hash=str(scope_artifacts["null_label_table_hash"]),
            candidate_scope_path=Path(str(scope_artifacts["candidate_scope_path"])),
            candidate_scope_metadata=scope_artifacts["candidate_scope_metadata"],
            initial_ids=shared_initial_ids,
            initial_seed_context=initial_seed_context,
            initial_seed_source_role="positive",
            candidate_identity=candidate_identity,
            stage_a_planned_config_hash=str(pair_row["null_campaign_config_hash"]),
            baserender_metadata_records_path=densegen_sidecar_path,
        )
        campaigns.extend([positive, null])
        pairs.append(_build_pair_manifest_row(label_name=label_name, positive=positive, null=null, pair_row=pair_row))

    validation = (
        validate_campaign_configs(campaigns, cfg=cfg) if cfg.validate_configs else skipped_validation(campaigns)
    )
    collection_manifest = stage_b_collection_manifest(
        split_id=cfg.split_id,
        seed=cfg.seed,
        control_pair_label=control_pair_label(target_profile_id=target_profile.profile_id),
        control_role_label=control_role_display_label(target_profile_id=target_profile.profile_id),
    )
    write_stage_b_json(layout.collection_manifest_path, collection_manifest)
    manifest = _build_config_manifest(
        cfg=cfg,
        layout=layout,
        stage_a=stage_a,
        retention=retention,
        target_profile=target_profile.to_manifest(),
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
        "null_control_role": pair_row.get("null_control_role"),
        "negative_control_claim_status": pair_row.get("negative_control_claim_status"),
        "null_permutation_seed": int(pair_row.get("null_permutation_seed", pair_row["seed"])),
        "null_permutation_seed_context": pair_row.get("null_permutation_seed_context")
        or tfbs_stage_a_null_permutation_seed_context(seed=int(pair_row["seed"])),
        "retention_policy_hash": pair_row["retention_policy_hash"],
        "initial_seed_policy": positive["initial_seed_policy"],
        "initial_seed_context": positive["initial_seed_context"],
        "initial_seed_source_role": positive["initial_seed_source_role"],
        "initial_seed_pairing": "shared_positive_null_starting_ids",
        "initial_label_ids_hash": positive["initial_label_ids_hash"],
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
        "candidate_scope_path": positive["candidate_scope_path"],
        "candidate_scope_hash": positive["candidate_scope_hash"],
        **_candidate_scope_metadata(positive),
    }


def _candidate_scope_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_scope_policy_id",
        "target_family_count_column",
        "required_count_value",
        "claim_boundary",
        "row_count",
        "positive_label_marginal",
        "candidate_scope_manifest_path",
        "candidate_scope_manifest_hash",
    )
    return {key: row[key] for key in keys if key in row}


def _candidate_scope_manifest_rows(campaigns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for campaign in campaigns:
        label_name = str(campaign["label_name"])
        scope_hash = str(campaign["candidate_scope_hash"])
        rows[(label_name, scope_hash)] = {
            "label_name": label_name,
            "candidate_scope_path": campaign["candidate_scope_path"],
            "candidate_scope_hash": scope_hash,
            **_candidate_scope_metadata(campaign),
        }
    return sorted(rows.values(), key=lambda row: row["label_name"])


def _build_config_manifest(
    *,
    cfg: TfbsStageBConfig,
    layout: TfbsStageBLayout,
    stage_a: Mapping[str, Any],
    retention: Mapping[str, Any],
    target_profile: Mapping[str, Any],
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
        "null_permutation_seed": int(stage_a.get("null_permutation_seed", stage_a["seed"])),
        "null_permutation_seed_context": stage_a.get("null_permutation_seed_context")
        or tfbs_stage_a_null_permutation_seed_context(seed=int(stage_a["seed"])),
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
        "target_profile": dict(target_profile),
        "sentinel_labels": list(cfg.label_names),
        "campaign_count": int(len(campaigns)),
        "scratch_dataset": layout.dataset,
        "scratch_usr_dir": str(layout.scratch_usr_dir),
        "records_path": str(layout.records_path),
        "records_hash": file_sha256(layout.records_path),
        "candidate_scope_path": str(layout.candidate_scope_path),
        "candidate_scope_hash": file_sha256(layout.candidate_scope_path),
        "candidate_scope_mode": "label_specific_count_fixed"
        if uses_count_fixed_scope(str(target_profile.get("profile_id") or ""))
        else "shared_all_active_candidates",
        "candidate_scopes": _candidate_scope_manifest_rows(campaigns),
        "validation": dict(validation),
        "pairs": sorted((dict(row) for row in pairs), key=lambda row: row["label_name"]),
        "campaigns": sorted((dict(row) for row in campaigns), key=lambda row: row["campaign_key"]),
    }
