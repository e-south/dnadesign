"""Fail-fast validation for DenseGen TFBS Stage B sentinel config generation."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...active_targets import validate_tfbs_learnability_target_set
from ..commands import opal_validate_command
from ..io import write_stage_b_json
from ..layout import TfbsStageBLayout
from ..seed import validate_tfbs_stage_b_initial_seed_policy
from ..semantics import TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING, validate_stage_b_split_id, validate_stage_b_tie_handling
from .contracts import TfbsStageBConfig


def normalize_config(config: TfbsStageBConfig) -> TfbsStageBConfig:
    """Return a path-normalized, contract-validated Stage B config."""

    stage_a_run_root = Path(config.stage_a_run_root)
    if str(stage_a_run_root) in {"", "."}:
        raise ValueError("Stage B stage_a_run_root must be explicit")
    if not stage_a_run_root.exists():
        raise FileNotFoundError(f"Stage A run root not found: {stage_a_run_root}")
    label_names = validate_tfbs_learnability_target_set(tuple(config.label_names))
    validate_stage_b_split_id(config.split_id)
    if config.seed < 0:
        raise ValueError("Stage B seed must be non-negative")
    if config.rounds <= 0:
        raise ValueError("Stage B rounds must be positive")
    if config.selection_k <= 0:
        raise ValueError("Stage B selection_k must be positive")
    if config.initial_label_count <= 0:
        raise ValueError("Stage B initial_label_count must be positive")
    selection_tie_handling = validate_stage_b_tie_handling(config.selection_tie_handling)
    initial_seed_policy = validate_tfbs_stage_b_initial_seed_policy(config.initial_seed_policy)
    if selection_tie_handling == TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING and int(config.initial_label_count) != int(
        config.selection_k
    ):
        raise ValueError(
            "Stage B exact-budget acquisition requires initial_label_count == selection_k "
            f"(got initial_label_count={config.initial_label_count}, selection_k={config.selection_k})"
        )
    if config.score_batch_size <= 0:
        raise ValueError("Stage B score_batch_size must be positive")
    if config.max_x_matrix_gib <= 0:
        raise ValueError("Stage B max_x_matrix_gib must be positive")
    return replace(
        config,
        stage_a_run_root=stage_a_run_root,
        out_dir=None if config.out_dir is None else Path(config.out_dir),
        repo_root=None if config.repo_root is None else Path(config.repo_root),
        label_names=label_names,
        initial_seed_policy=initial_seed_policy,
        selection_tie_handling=selection_tie_handling,
    )


def validate_stage_a_inputs(
    *,
    stage_a: Mapping[str, Any],
    pairing: Mapping[str, Any],
    retention: Mapping[str, Any],
    requested_labels: Sequence[str],
    seed: int,
    split_id: str,
) -> None:
    """Validate Stage A manifests before materializing Stage B campaign configs."""

    if stage_a.get("status") != "PASS":
        raise ValueError("Stage A manifest status must be PASS before Stage B config generation")
    if retention.get("status") != "PASS":
        raise ValueError("Stage A retention status must be PASS before Stage B config generation")
    if stage_a.get("retention_policy_hash") != retention.get("retention_policy_hash"):
        raise ValueError("Stage A manifest retention hash does not match retention_estimate.json")
    pair_rows = pairing.get("pairs")
    if not isinstance(pair_rows, list) or not pair_rows:
        raise ValueError("Stage A pairing_manifest.json must contain non-empty pairs")
    labels_in_pairs = {str(row.get("label_name")) for row in pair_rows if isinstance(row, Mapping)}
    missing = sorted(set(requested_labels) - labels_in_pairs)
    if missing:
        raise ValueError(f"Stage A pairing manifest missing requested label(s): {missing}")
    for row in pair_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Stage A pairing manifest pairs must be mappings")
        if str(row.get("label_name")) not in set(requested_labels):
            continue
        if str(row.get("split_id")) != split_id:
            raise ValueError(f"Stage A pairing split mismatch for {row.get('label_name')}: {row.get('split_id')}")
        if int(row.get("seed")) != int(seed):
            raise ValueError(f"Stage A pairing seed mismatch for {row.get('label_name')}: {row.get('seed')}")
        if str(row.get("retention_policy_hash")) != str(retention.get("retention_policy_hash")):
            raise ValueError(f"Stage A pairing retention hash mismatch for {row.get('label_name')}")


def validate_campaign_configs(campaigns: Sequence[Mapping[str, Any]], *, cfg: TfbsStageBConfig) -> dict[str, Any]:
    """Run OPAL validation for generated campaign configs and write report artifacts."""

    reports = []
    ok = True
    repo_root = Path.cwd() if cfg.repo_root is None else cfg.repo_root
    validation_dir = TfbsStageBLayout(
        cfg.out_dir or (cfg.stage_a_run_root / "stage_b_sentinel_configs"), cfg.split_id, cfg.seed
    )
    for campaign in campaigns:
        config_path = Path(str(campaign["config_path"]))
        command = [*opal_validate_command(config_path), "--json"]
        proc = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
        report_path = validation_dir.validation_reports_dir / f"{campaign['campaign_key']}.opal_validate.json"
        payload = {
            "campaign_key": campaign["campaign_key"],
            "config_path": str(config_path),
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        write_stage_b_json(report_path, payload)
        ok = ok and proc.returncode == 0
        reports.append(
            {
                "campaign_key": campaign["campaign_key"],
                "config_path": str(config_path),
                "report_path": str(report_path),
                "returncode": int(proc.returncode),
                "status": "PASS" if proc.returncode == 0 else "FAIL",
            }
        )
    return {
        "status": "PASS" if ok else "FAIL",
        "mode": "opal_validate",
        "campaign_count": int(len(campaigns)),
        "reports": reports,
    }


def skipped_validation(campaigns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the explicit validation payload when OPAL config checks are skipped."""

    return {
        "status": "SKIPPED",
        "mode": "not_run",
        "campaign_count": int(len(campaigns)),
        "reports": [],
    }
