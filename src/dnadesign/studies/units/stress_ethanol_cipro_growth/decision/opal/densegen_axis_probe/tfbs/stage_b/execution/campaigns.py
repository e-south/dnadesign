"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/execution/campaigns.py

Per-campaign OPAL execution orchestration for Stage B.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ...stage_a.manifests import file_sha256
from ..semantics import TFBS_STAGE_B_RETENTION_MODE, stage_b_selection_budget_mode
from .commands import (
    opal_ingest_command,
    opal_init_command,
    opal_run_command,
    opal_status_command,
    opal_validate_command,
    run_command,
)
from .label_inputs import observed_label_ids_for_round, write_label_input_for_ids
from .manifest import read_json
from .selection import (
    assert_selection_budget,
    campaign_selection_k,
    campaign_tie_handling,
    selected_ids_from_round,
    selection_exists,
)


def run_campaign(
    campaign: Mapping[str, Any],
    *,
    repo_root: Path,
    rounds: int,
    resume: bool,
) -> dict[str, Any]:
    config_path = Path(str(campaign["config_path"]))
    label_table_path = Path(str(campaign["label_table_path"]))
    records_path = Path(str(campaign["records_path"]))
    sidecar_path = Path(str(campaign["label_sidecar_path"]))
    label_name = str(campaign["label_name"])
    selection_k = campaign_selection_k(campaign)
    selection_tie_handling = campaign_tie_handling(campaign)
    workdir = campaign_workdir(config_path)
    if not resume:
        fail_if_campaign_has_execution_state(workdir=workdir, campaign=campaign)
    assert_config_retention(config_path)

    run_command(opal_validate_command(config_path), cwd=repo_root)
    if not (workdir / "state.json").exists():
        run_command(opal_init_command(config_path), cwd=repo_root)
    if not observed_label_ids_for_round(sidecar_path=sidecar_path, round_index=0):
        run_command(
            opal_ingest_command(config_path, Path(str(campaign["initial_label_input_path"])), round_index=0),
            cwd=repo_root,
        )

    label_input_paths = [Path(str(campaign["initial_label_input_path"]))]
    for round_index in range(rounds):
        if round_index > 0:
            label_input_path = workdir / "inputs" / f"r{round_index}" / f"labels-b{round_index}.parquet"
            if not observed_label_ids_for_round(sidecar_path=sidecar_path, round_index=round_index):
                selected_ids = selected_ids_from_round(
                    workdir=workdir,
                    round_index=round_index - 1,
                    selection_k=selection_k,
                    tie_handling=selection_tie_handling,
                )
                write_label_input_for_ids(
                    path=label_input_path,
                    label_table_path=label_table_path,
                    records_path=records_path,
                    label_name=label_name,
                    ids=selected_ids,
                )
                run_command(opal_ingest_command(config_path, label_input_path, round_index=round_index), cwd=repo_root)
            label_input_paths.append(label_input_path)
        if not selection_exists(workdir=workdir, round_index=round_index):
            run_command(opal_run_command(config_path, round_index=round_index, resume=resume), cwd=repo_root)
            assert_retention_manifest(workdir)
        assert_selection_budget(
            workdir=workdir,
            round_index=round_index,
            selection_k=selection_k,
            tie_handling=selection_tie_handling,
        )

    run_command(opal_status_command(config_path), cwd=repo_root)
    retention_path = workdir / "outputs" / "retention_manifest.json"
    return {
        "campaign_key": campaign["campaign_key"],
        "label_name": label_name,
        "oracle_role": campaign["oracle_role"],
        "selection_k": int(selection_k),
        "selection_tie_handling": selection_tie_handling,
        "selection_budget_mode": stage_b_selection_budget_mode(tie_handling=selection_tie_handling),
        "config_path": str(config_path),
        "workdir": str(workdir),
        "rounds": int(rounds),
        "label_input_paths": [str(path) for path in label_input_paths],
        "retention_manifest_path": str(retention_path),
        "retention_manifest_hash": file_sha256(retention_path),
        "status": "PASS",
    }


def assert_config_retention(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    retention = payload.get("artifact_retention")
    if not isinstance(retention, Mapping):
        raise ValueError(f"Stage B config missing artifact_retention block: {config_path}")
    if retention.get("mode") != TFBS_STAGE_B_RETENTION_MODE:
        raise ValueError(f"Stage B config must use production_review retention: {config_path}")


def assert_retention_manifest(workdir: Path) -> None:
    path = workdir / "outputs" / "retention_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"OPAL retention manifest missing after run: {path}")
    payload = read_json(path)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"OPAL retention manifest did not PASS: {path}")


def fail_if_campaign_has_execution_state(*, workdir: Path, campaign: Mapping[str, Any]) -> None:
    execution_paths = [
        workdir / "state.json",
        workdir / "outputs",
        Path(str(campaign["label_sidecar_path"])),
    ]
    existing = [path for path in execution_paths if path.exists()]
    if existing:
        preview = ", ".join(str(path) for path in existing[:3])
        raise RuntimeError(
            "Stage B execution refuses to reuse existing campaign state without resume_existing=True "
            f"(sample={preview})"
        )


def campaign_workdir(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(str(payload["campaign"]["workdir"]))
