"""Stage B TFBS OPAL YAML payload builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from ...constants import STUDY_ID, X_COLUMN
from ..active_targets import tfbs_learnability_active_target_spec
from .layout import TfbsStageBLayout
from .semantics import (
    TFBS_STAGE_B_LABEL_SOURCE_KIND,
    TFBS_STAGE_B_MODEL_ARTIFACT_RETENTION,
    TFBS_STAGE_B_PLOT_TIDY_RETENTION,
    TFBS_STAGE_B_PREDICTION_LEDGER_RETENTION,
    TFBS_STAGE_B_PROBE_FAMILY,
    TFBS_STAGE_B_RETENTION_MODE,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_STAGE,
    TFBS_STAGE_B_TABULAR_FORMAT,
    TfbsStageBRunIdentity,
    stage_b_selection_budget_mode,
    validate_stage_b_oracle_role,
)


class TfbsStageBConfigSurface(Protocol):
    """Config attributes required to build one OPAL campaign payload."""

    stage_a_run_root: Path
    split_id: str
    seed: int
    rounds: int
    selection_k: int
    initial_label_count: int
    initial_seed_policy: str
    selection_tie_handling: str
    score_batch_size: int
    max_x_matrix_gib: float


def tfbs_stage_b_campaign_config_payload(
    *,
    cfg: TfbsStageBConfigSurface,
    layout: TfbsStageBLayout,
    workdir: Path,
    run_key: str,
    stage_a: Mapping[str, Any],
    pairing_manifest_path: Path,
    retention: Mapping[str, Any],
    pair_row: Mapping[str, Any],
    label_name: str,
    oracle_role: str,
) -> dict[str, Any]:
    """Build the OPAL campaign config payload for one positive or null run."""

    target = tfbs_learnability_active_target_spec(label_name)
    validate_stage_b_oracle_role(oracle_role)
    slug = TfbsStageBRunIdentity(
        label_name=label_name,
        oracle_role=oracle_role,
        split_id=cfg.split_id,
        seed=cfg.seed,
    ).campaign_slug
    final_round = int(cfg.rounds) - 1
    return {
        "campaign": {
            "name": f"DenseGen TFBS learnability {label_name} {oracle_role} seed {int(cfg.seed)}",
            "slug": slug,
            "description": (
                "Stage B sentinel OPAL campaign for a synthetic DenseGen TFBS construction label. "
                "The label is an oracle target for learnability review, not a wet-lab phenotype."
            ),
            "workdir": str(workdir.resolve()),
            "metadata": {
                "study_id": STUDY_ID,
                "probe_family": TFBS_STAGE_B_PROBE_FAMILY,
                "probe_stage": TFBS_STAGE_B_STAGE,
                "probe_scope": TFBS_STAGE_B_SCOPE,
                "label_name": label_name,
                "label_family_id": target.label_family_id,
                "target": label_name,
                "target_label": target.target_description,
                "label_oracle_kind": "positive" if oracle_role == "positive" else "null",
                "label_split_id": cfg.split_id,
                "target_kind": target.target_kind,
                "target_description": target.target_description,
                "oracle_role": oracle_role,
                "positive_oracle_version": stage_a["positive_oracle_version"],
                "null_version": pair_row["null_version"],
                "split_id": cfg.split_id,
                "seed": int(cfg.seed),
                "rounds": int(cfg.rounds),
                "selection_k": int(cfg.selection_k),
                "initial_seed_policy": cfg.initial_seed_policy,
                "selection_tie_handling": cfg.selection_tie_handling,
                "selection_budget_mode": stage_b_selection_budget_mode(tie_handling=cfg.selection_tie_handling),
                "initial_label_count": int(cfg.initial_label_count),
                "retention_mode": TFBS_STAGE_B_RETENTION_MODE,
                "retention_policy_hash": retention["retention_policy_hash"],
                "stage_a_manifest_path": str(cfg.stage_a_run_root / "manifests" / "tfbs_stage_a_manifest.json"),
                "stage_a_pairing_manifest_path": str(pairing_manifest_path),
                "interpretation_boundary": target.interpretation_boundary,
            },
        },
        "plot_config": "plots.yaml",
        "ownership": {
            "owner_scope": "study_fixture",
            "study_id": STUDY_ID,
            "dataset_id": layout.dataset,
            "portable": False,
        },
        "data": {
            "location": {
                "kind": "usr",
                "path": str(layout.scratch_usr_dir.resolve()),
                "dataset": layout.dataset,
            },
            "x_column_name": X_COLUMN,
            "y_column_name": f"opal__{slug}__y",
            "y_expected_length": int(target.y_expected_length),
            "candidate_scope": {
                "kind": "id_list",
                "path": str(layout.candidate_scope_path.resolve()),
                "id_column": "id",
            },
        },
        "labels": {
            "source": {
                "kind": TFBS_STAGE_B_LABEL_SOURCE_KIND,
                "dataset": layout.dataset,
                "path": layout.sidecar_relative_path(run_key),
            },
            "y_space": target.y_space,
            "id_column": "id",
            "round_column": "observed_round",
            "batch_column": "batch_id",
            "dedup_policy": "latest_by_round",
        },
        "writeback": {"prediction_records": "ledger_only"},
        "artifact_retention": {
            "mode": TFBS_STAGE_B_RETENTION_MODE,
            "prediction_ledger": TFBS_STAGE_B_PREDICTION_LEDGER_RETENTION,
            "plot_tidy_data": TFBS_STAGE_B_PLOT_TIDY_RETENTION,
            "model_artifacts": TFBS_STAGE_B_MODEL_ARTIFACT_RETENTION,
            "tabular_format": TFBS_STAGE_B_TABULAR_FORMAT,
            "max_estimated_bytes": int(retention["max_estimated_bytes"]),
            "fail_if_estimate_exceeds": bool(retention["fail_if_estimate_exceeds"]),
            "final_round": final_round,
        },
        "ingest": {"duplicate_policy": "error"},
        "training": {
            "policy": {
                "cumulative_training": True,
                "label_cross_round_deduplication_policy": "latest_only",
                "allow_resuggesting_candidates_until_labeled": True,
            }
        },
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {
            "name": target.transforms_y["name"],
            "params": dict(target.transforms_y.get("params") or {}),
        },
        "model": {
            "name": "random_forest",
            "params": {
                "n_estimators": 100,
                "criterion": "friedman_mse",
                "bootstrap": True,
                "oob_score": True,
                "random_state": int(cfg.seed),
                "n_jobs": -1,
                "emit_feature_importance": True,
            },
        },
        "objectives": [{"name": item["name"], "params": dict(item.get("params") or {})} for item in target.objectives],
        "selection": {
            "name": "top_n",
            "params": {
                "top_k": int(cfg.selection_k),
                "score_ref": target.score_ref,
                "objective_mode": target.objective_mode,
                "tie_handling": cfg.selection_tie_handling,
            },
        },
        "scoring": {"score_batch_size": int(cfg.score_batch_size)},
        "safety": {
            "fail_on_mixed_biotype_or_alphabet": True,
            "require_biotype_and_alphabet_on_init": True,
            "conflict_policy_on_duplicate_ids": "error",
            "write_back_requires_columns_present": True,
            "accept_x_mismatch": False,
            "max_x_matrix_gib": float(cfg.max_x_matrix_gib),
        },
    }


def write_tfbs_stage_b_plot_config(path: Path, *, label_name: str, target_display: str) -> None:
    """Write the OPAL plot config for one Stage B sentinel campaign."""

    payload = {
        "plots": [
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "round_selector": "all",
                "tags": ["rounds", "tfbs_learnability", "sentinel"],
                "params": {
                    "metric": "pred__score_selected",
                    "cohort": "selected",
                    "summaries": ["mean", "count"],
                    "band": "iqr",
                    "title": f"Selected predicted {label_name} over rounds",
                    "surface_label": f"Selected predicted expected label: {target_display}",
                    "caption": (
                        "Selected-candidate mean predicted expected scalar label by OPAL round. "
                        "This is a synthetic construction-label objective, not a measured biological phenotype."
                    ),
                },
            }
        ],
        "plot_defaults": {"output": {"save_data": True}, "data": [], "params": {}},
        "plot_presets": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
