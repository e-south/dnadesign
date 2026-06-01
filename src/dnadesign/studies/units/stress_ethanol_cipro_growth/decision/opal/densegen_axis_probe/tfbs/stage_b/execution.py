"""Stage B execution loop for DenseGen TFBS learnability sentinel campaigns."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from ..stage_a.manifests import file_sha256
from .semantics import (
    TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING,
    TFBS_STAGE_B_RETENTION_MODE,
    TFBS_STAGE_B_SCOPE,
    TFBS_STAGE_B_STAGE,
    stage_b_selection_budget_mode,
    validate_stage_b_tie_handling,
)

EXECUTION_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_execution.v1"


@dataclass(frozen=True)
class TfbsStageBExecutionConfig:
    """Runtime inputs for executing generated Stage B sentinel campaigns."""

    config_manifest_path: Path
    repo_root: Path
    rounds: int | None = None
    campaign_keys: tuple[str, ...] = ()
    resume_existing: bool = False
    machine_readable: bool = True


@dataclass(frozen=True)
class TfbsStageBExecutionResult:
    """Result summary for a Stage B sentinel execution run."""

    status: str
    execution_manifest_path: Path
    campaign_count: int
    round_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_manifest_path": str(self.execution_manifest_path),
            "campaign_count": int(self.campaign_count),
            "round_count": int(self.round_count),
        }


def run_tfbs_stage_b_sentinel_campaigns(config: TfbsStageBExecutionConfig) -> TfbsStageBExecutionResult:
    """Run Stage B OPAL campaigns from a validated config manifest."""

    cfg = _normalize_execution_config(config)
    manifest = _read_json(cfg.config_manifest_path)
    _validate_config_manifest(manifest)
    campaigns = _selected_campaign_rows(manifest, cfg.campaign_keys)
    if not campaigns:
        raise ValueError("Stage B execution selected zero campaigns")
    round_count = int(cfg.rounds if cfg.rounds is not None else manifest["rounds"])
    if round_count <= 0:
        raise ValueError("Stage B execution rounds must be positive")

    results = []
    for campaign in campaigns:
        results.append(_run_campaign(campaign, repo_root=cfg.repo_root, rounds=round_count, resume=cfg.resume_existing))
    execution_manifest = _build_execution_manifest(
        source_manifest_path=cfg.config_manifest_path,
        source_manifest=manifest,
        campaign_results=results,
        rounds=round_count,
    )
    out_path = cfg.config_manifest_path.parent / "stage_b_sentinel_execution_manifest.json"
    _write_json(out_path, execution_manifest)
    return TfbsStageBExecutionResult(
        status="PASS",
        execution_manifest_path=out_path,
        campaign_count=len(results),
        round_count=round_count,
    )


def _normalize_execution_config(config: TfbsStageBExecutionConfig) -> TfbsStageBExecutionConfig:
    manifest_path = Path(config.config_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage B config manifest not found: {manifest_path}")
    repo_root = Path(config.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
    if config.rounds is not None and int(config.rounds) <= 0:
        raise ValueError("Stage B execution rounds must be positive")
    return TfbsStageBExecutionConfig(
        config_manifest_path=manifest_path,
        repo_root=repo_root,
        rounds=None if config.rounds is None else int(config.rounds),
        campaign_keys=tuple(dict.fromkeys(map(str, config.campaign_keys))),
        resume_existing=bool(config.resume_existing),
        machine_readable=bool(config.machine_readable),
    )


def _validate_config_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "PASS":
        raise ValueError("Stage B execution requires config manifest status PASS")
    if manifest.get("stage") != TFBS_STAGE_B_STAGE:
        raise ValueError(f"Stage B execution requires stage {TFBS_STAGE_B_STAGE!r}")
    if manifest.get("scope") != TFBS_STAGE_B_SCOPE:
        raise ValueError(f"Stage B execution requires scope {TFBS_STAGE_B_SCOPE!r}")
    if manifest.get("validation", {}).get("status") != "PASS":
        raise ValueError("Stage B execution requires OPAL config validation PASS")
    if manifest.get("retention_mode") != TFBS_STAGE_B_RETENTION_MODE:
        raise ValueError("Stage B execution requires production_review retention mode")
    if int(manifest.get("campaign_count") or 0) <= 0:
        raise ValueError("Stage B execution requires at least one campaign")


def _selected_campaign_rows(
    manifest: Mapping[str, Any],
    campaign_keys: Sequence[str],
) -> list[Mapping[str, Any]]:
    rows = manifest.get("campaigns")
    if not isinstance(rows, list):
        raise ValueError("Stage B config manifest campaigns must be a list")
    requested = set(map(str, campaign_keys))
    selected = [row for row in rows if isinstance(row, Mapping) and (not requested or row["campaign_key"] in requested)]
    found = {str(row["campaign_key"]) for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Stage B config manifest missing requested campaign key(s): {missing}")
    return sorted(selected, key=lambda row: str(row["campaign_key"]))


def _run_campaign(
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
    selection_k = _campaign_selection_k(campaign)
    selection_tie_handling = _campaign_tie_handling(campaign)
    workdir = _campaign_workdir(config_path)
    if not resume:
        _fail_if_campaign_has_execution_state(workdir=workdir, campaign=campaign)
    _assert_config_retention(config_path)

    _run_command(_opal_validate_command(config_path), cwd=repo_root)
    if not (workdir / "state.json").exists():
        _run_command(_opal_init_command(config_path), cwd=repo_root)
    if not _observed_label_ids_for_round(sidecar_path=sidecar_path, round_index=0):
        _run_command(
            _opal_ingest_command(config_path, Path(str(campaign["initial_label_input_path"])), round_index=0),
            cwd=repo_root,
        )

    label_input_paths = [Path(str(campaign["initial_label_input_path"]))]
    for round_index in range(rounds):
        if round_index > 0:
            label_input_path = workdir / "inputs" / f"r{round_index}" / f"labels-b{round_index}.parquet"
            if not _observed_label_ids_for_round(sidecar_path=sidecar_path, round_index=round_index):
                selected_ids = _selected_ids_from_round(
                    workdir=workdir,
                    round_index=round_index - 1,
                    selection_k=selection_k,
                    tie_handling=selection_tie_handling,
                )
                _write_label_input_for_ids(
                    path=label_input_path,
                    label_table_path=label_table_path,
                    records_path=records_path,
                    label_name=label_name,
                    ids=selected_ids,
                )
                _run_command(
                    _opal_ingest_command(config_path, label_input_path, round_index=round_index), cwd=repo_root
                )
            label_input_paths.append(label_input_path)
        if not _selection_exists(workdir=workdir, round_index=round_index):
            _run_command(_opal_run_command(config_path, round_index=round_index, resume=resume), cwd=repo_root)
            _assert_retention_manifest(workdir)
        _assert_selection_budget(
            workdir=workdir,
            round_index=round_index,
            selection_k=selection_k,
            tie_handling=selection_tie_handling,
        )

    _run_command(_opal_status_command(config_path), cwd=repo_root)
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


def _write_label_input_for_ids(
    *,
    path: Path,
    label_table_path: Path,
    records_path: Path,
    label_name: str,
    ids: Sequence[str],
) -> None:
    if not ids:
        raise ValueError("Stage B follow-up label input requires at least one selected id")
    label_table = pd.read_parquet(label_table_path)
    required = {"id", label_name}
    missing = sorted(required - set(label_table.columns))
    if missing:
        raise ValueError(f"Stage B label table missing column(s): {missing}")
    frame = label_table.copy()
    if "sequence" not in frame.columns:
        identity = pd.read_parquet(records_path, columns=["id", "sequence"])
        frame = frame.merge(identity, on="id", how="left", validate="one_to_one")
    if "sequence" not in frame.columns:
        raise ValueError("Stage B follow-up label input requires sequence")
    wanted = [str(value) for value in ids]
    if len(set(wanted)) != len(wanted):
        raise ValueError("Stage B follow-up label input selected ids must be unique")
    selected = frame.loc[frame["id"].astype(str).isin(set(wanted)), ["id", "sequence", label_name]].copy()
    found = set(selected["id"].astype(str).tolist())
    missing_ids = sorted(set(wanted) - found)
    if missing_ids:
        raise ValueError(f"Stage B label table missing selected id(s): {missing_ids[:10]}")
    order = {candidate_id: index for index, candidate_id in enumerate(wanted)}
    selected["__order__"] = selected["id"].astype(str).map(order)
    selected = selected.sort_values("__order__").drop(columns=["__order__"])
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(path, index=False, compression="zstd")


def _selected_ids_from_round(
    *,
    workdir: Path,
    round_index: int,
    selection_k: int | None = None,
    tie_handling: str | None = None,
) -> tuple[str, ...]:
    path = workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage B selection artifact missing: {path}")
    frame = pd.read_csv(path, usecols=["id"])
    ids = tuple(str(value).strip() for value in frame["id"].tolist())
    if not ids or any(not value for value in ids):
        raise ValueError(f"Stage B selection artifact has blank or empty ids: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage B selection artifact contains duplicate ids: {path}")
    if selection_k is not None and tie_handling is not None:
        _assert_selected_count(
            selected_count=len(ids),
            path=path,
            selection_k=selection_k,
            tie_handling=tie_handling,
        )
    return ids


def _selection_exists(*, workdir: Path, round_index: int) -> bool:
    return (workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv").exists()


def _assert_selection_budget(*, workdir: Path, round_index: int, selection_k: int, tie_handling: str) -> None:
    path = workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage B selection artifact missing: {path}")
    frame = pd.read_csv(path, usecols=["id"])
    _assert_selected_count(
        selected_count=len(frame),
        path=path,
        selection_k=selection_k,
        tie_handling=tie_handling,
    )


def _assert_selected_count(*, selected_count: int, path: Path, selection_k: int, tie_handling: str) -> None:
    tie = validate_stage_b_tie_handling(tie_handling)
    if int(selection_k) <= 0:
        raise ValueError("Stage B selection_k must be positive")
    if tie != TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING:
        return
    if int(selected_count) != int(selection_k):
        raise RuntimeError(
            "Stage B exact-budget selection contract failed: "
            f"expected {int(selection_k)} selected row(s), observed {int(selected_count)} in {path}"
        )


def _campaign_selection_k(campaign: Mapping[str, Any]) -> int:
    value = int(campaign.get("selection_k", 0))
    if value <= 0:
        raise ValueError(f"Stage B campaign missing positive selection_k: {campaign.get('campaign_key')}")
    return value


def _campaign_tie_handling(campaign: Mapping[str, Any]) -> str:
    value = campaign.get("selection_tie_handling", TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING)
    return validate_stage_b_tie_handling(str(value))


def _observed_label_ids_for_round(*, sidecar_path: Path, round_index: int) -> set[str]:
    if not sidecar_path.exists():
        return set()
    frame = pd.read_parquet(sidecar_path, columns=["id", "observed_round"])
    round_frame = frame.loc[pd.to_numeric(frame["observed_round"], errors="coerce") == int(round_index)]
    return {str(value).strip() for value in round_frame["id"].dropna().tolist()}


def _assert_config_retention(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    retention = payload.get("artifact_retention")
    if not isinstance(retention, Mapping):
        raise ValueError(f"Stage B config missing artifact_retention block: {config_path}")
    if retention.get("mode") != TFBS_STAGE_B_RETENTION_MODE:
        raise ValueError(f"Stage B config must use production_review retention: {config_path}")


def _assert_retention_manifest(workdir: Path) -> None:
    path = workdir / "outputs" / "retention_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"OPAL retention manifest missing after run: {path}")
    payload = _read_json(path)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"OPAL retention manifest did not PASS: {path}")


def _fail_if_campaign_has_execution_state(*, workdir: Path, campaign: Mapping[str, Any]) -> None:
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


def _campaign_workdir(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(str(payload["campaign"]["workdir"]))


def _opal_validate_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "validate", "-c", str(config_path), "--json"]


def _opal_init_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "init", "-c", str(config_path), "--json"]


def _opal_ingest_command(config_path: Path, labels_path: Path, *, round_index: int) -> list[str]:
    return [
        "uv",
        "run",
        "opal",
        "ingest-y",
        "-c",
        str(config_path),
        "--round",
        str(int(round_index)),
        "--in",
        str(labels_path),
        "--unknown-sequences",
        "error",
        "--apply",
        "--json",
    ]


def _opal_run_command(config_path: Path, *, round_index: int, resume: bool) -> list[str]:
    command = ["uv", "run", "opal", "run", "-c", str(config_path), "--round", str(int(round_index)), "--json"]
    if resume:
        command.append("--resume")
    return command


def _opal_status_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "status", "-c", str(config_path), "--with-ledger", "--json"]


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def _build_execution_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    campaign_results: Sequence[Mapping[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "stage": TFBS_STAGE_B_STAGE,
        "scope": TFBS_STAGE_B_SCOPE,
        "source_config_manifest_path": str(source_manifest_path),
        "source_config_manifest_hash": file_sha256(source_manifest_path),
        "retention_mode": source_manifest["retention_mode"],
        "rounds": int(rounds),
        "campaign_count": len(campaign_results),
        "campaigns": sorted((dict(row) for row in campaign_results), key=lambda row: row["campaign_key"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
