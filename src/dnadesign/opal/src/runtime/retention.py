"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/retention.py

Runtime artifact-retention enforcement for OPAL campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.types import ArtifactRetentionBlock, RootConfig
from ..core.utils import ExitCodes, OpalError, file_sha256, now_iso, write_json
from ..storage.parquet_io import read_parquet_df
from ..storage.prediction_selection import selected_by_any_view
from ..storage.workspace import CampaignWorkspace

RETENTION_MANIFEST_SCHEMA_VERSION = "opal.artifact_retention_manifest.v1"


def apply_runtime_artifact_retention(cfg: RootConfig, ws: CampaignWorkspace) -> dict[str, Any]:
    """Apply the configured artifact-retention policy and write a manifest.

    `audit_full` preserves complete run artifacts and records the applied
    policy. Production modes fail closed on unknown policy values and only
    delete artifacts whose ownership can be proven from the campaign workspace
    layout.
    """

    policy = cfg.artifact_retention
    _validate_policy(policy)
    manifest_path = ws.outputs_dir / "retention_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": RETENTION_MANIFEST_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "campaign": {"slug": cfg.campaign.slug, "workdir": str(ws.workdir)},
        "policy": asdict(policy),
        "actions": [],
        "status": "PASS",
    }
    if policy.mode == "audit_full":
        write_json(manifest_path, manifest)
        return manifest

    if policy.prediction_ledger == "latest_full_plus_selected_history":
        manifest["actions"].append(_compact_prediction_ledger(policy=policy, ws=ws))
    elif policy.prediction_ledger == "selected_history_only":
        manifest["actions"].append(_compact_prediction_ledger(policy=policy, ws=ws))

    if policy.model_artifacts == "latest":
        manifest["actions"].append(_prune_model_artifacts(policy=policy, ws=ws))

    write_json(manifest_path, manifest)
    return manifest


def _validate_policy(policy: ArtifactRetentionBlock) -> None:
    if policy.mode not in {"audit_full", "production_review", "ephemeral_selection"}:
        raise OpalError(f"unsupported artifact_retention.mode: {policy.mode!r}", ExitCodes.CONTRACT_VIOLATION)
    if policy.prediction_ledger not in {
        "all_rounds_full",
        "latest_full_plus_selected_history",
        "selected_history_only",
    }:
        raise OpalError(
            f"unsupported artifact_retention.prediction_ledger: {policy.prediction_ledger!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if policy.plot_tidy_data not in {"full", "compact", "none"}:
        raise OpalError(
            f"unsupported artifact_retention.plot_tidy_data: {policy.plot_tidy_data!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if policy.model_artifacts not in {"all", "latest"}:
        raise OpalError(
            f"unsupported artifact_retention.model_artifacts: {policy.model_artifacts!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if policy.tabular_format not in {"parquet", "parquet_zstd"}:
        raise OpalError(
            f"unsupported artifact_retention.tabular_format: {policy.tabular_format!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if int(policy.max_estimated_bytes) <= 0:
        raise OpalError("artifact_retention.max_estimated_bytes must be positive", ExitCodes.CONTRACT_VIOLATION)
    if policy.final_round is not None and int(policy.final_round) < 0:
        raise OpalError("artifact_retention.final_round must be non-negative", ExitCodes.CONTRACT_VIOLATION)


def _compact_prediction_ledger(*, policy: ArtifactRetentionBlock, ws: CampaignWorkspace) -> dict[str, Any]:
    predictions_dir = ws.ledger_predictions_dir
    runs_path = ws.ledger_runs_path
    if not predictions_dir.exists():
        raise OpalError(f"cannot apply retention without predictions ledger: {predictions_dir}")
    if not runs_path.exists():
        raise OpalError(f"cannot apply retention without run ledger: {runs_path}")
    runs = read_parquet_df(runs_path, columns=["run_id", "as_of_round"])
    predictions = read_parquet_df(predictions_dir)
    required = {"run_id", "as_of_round", "pred__selection_views"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise OpalError(
            f"predictions ledger missing retention column(s): {missing}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if runs.empty:
        raise OpalError("cannot apply retention with empty run ledger", ExitCodes.CONTRACT_VIOLATION)
    latest_round = int(runs["as_of_round"].max())
    full_rounds = _retained_full_prediction_rounds(policy=policy, latest_round=latest_round, runs=runs)
    if policy.prediction_ledger == "selected_history_only":
        full_rounds = set()
    keep_selected = predictions["pred__selection_views"].map(selected_by_any_view)
    keep_full = predictions["as_of_round"].astype(int).isin(full_rounds)
    retained = predictions.loc[keep_selected | keep_full].copy()
    if retained.empty and not predictions.empty:
        raise OpalError(
            "retention would remove every prediction row; refusing to compact",
            ExitCodes.CONTRACT_VIOLATION,
        )
    rows_before = int(len(predictions))
    rows_after = int(len(retained))
    _rewrite_parquet_dataset(predictions_dir, retained, compression=_compression(policy))
    return {
        "kind": "prediction_ledger_compaction",
        "status": "PASS",
        "path": str(predictions_dir),
        "policy": policy.prediction_ledger,
        "retained_full_rounds": sorted(full_rounds),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "sha256": _dataset_sha256(predictions_dir),
    }


def _retained_full_prediction_rounds(
    *,
    policy: ArtifactRetentionBlock,
    latest_round: int,
    runs: pd.DataFrame,
) -> set[int]:
    rounds = {int(latest_round)}
    if policy.final_round is not None:
        final_round = int(policy.final_round)
        observed_rounds = {int(value) for value in runs["as_of_round"].dropna().astype(int).tolist()}
        if final_round in observed_rounds:
            rounds.add(final_round)
    return rounds


def _prune_model_artifacts(*, policy: ArtifactRetentionBlock, ws: CampaignWorkspace) -> dict[str, Any]:
    if not ws.ledger_runs_path.exists():
        raise OpalError(f"cannot prune model artifacts without run ledger: {ws.ledger_runs_path}")
    runs = read_parquet_df(ws.ledger_runs_path, columns=["as_of_round"])
    if runs.empty:
        return {"kind": "model_artifact_prune", "status": "PASS", "deleted": [], "retained_rounds": []}
    latest_round = int(runs["as_of_round"].max())
    retained_rounds = _retained_full_prediction_rounds(policy=policy, latest_round=latest_round, runs=runs)
    deleted: list[dict[str, Any]] = []
    for model_dir in sorted(ws.rounds_dir.glob("round_*/model")):
        round_index = _round_index_from_model_dir(model_dir)
        if round_index in retained_rounds:
            continue
        _assert_owned_model_dir(model_dir, ws)
        size_bytes = _directory_size(model_dir)
        shutil.rmtree(model_dir)
        deleted.append({"path": str(model_dir), "round": round_index, "size_bytes": size_bytes})
    return {
        "kind": "model_artifact_prune",
        "status": "PASS",
        "retained_rounds": sorted(retained_rounds),
        "deleted": deleted,
        "bytes_deleted": sum(int(row["size_bytes"]) for row in deleted),
    }


def _round_index_from_model_dir(model_dir: Path) -> int:
    parent = model_dir.parent.name
    if not parent.startswith("round_"):
        raise OpalError(f"unexpected OPAL model artifact layout: {model_dir}", ExitCodes.CONTRACT_VIOLATION)
    try:
        return int(parent.removeprefix("round_"))
    except ValueError as exc:
        raise OpalError(f"unexpected OPAL round artifact name: {parent}", ExitCodes.CONTRACT_VIOLATION) from exc


def _assert_owned_model_dir(model_dir: Path, ws: CampaignWorkspace) -> None:
    resolved = model_dir.resolve()
    allowed = ws.rounds_dir.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise OpalError(
            f"refusing to prune model artifact outside campaign rounds directory: {model_dir}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if model_dir.name != "model":
        raise OpalError(f"refusing to prune non-model artifact directory: {model_dir}", ExitCodes.CONTRACT_VIOLATION)


def _rewrite_parquet_dataset(path: Path, frame: pd.DataFrame, *, compression: str | None) -> None:
    tmp_dir = path.with_name(f"{path.name}.tmp-retention")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    out = tmp_dir / "part-retained.parquet"
    frame.to_parquet(out, index=False, engine="pyarrow", compression=compression)
    if path.exists():
        shutil.rmtree(path)
    tmp_dir.replace(path)


def _compression(policy: ArtifactRetentionBlock) -> str | None:
    return "zstd" if policy.tabular_format == "parquet_zstd" else None


def _directory_size(path: Path) -> int:
    size = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            size += int(child.stat().st_size)
    return size


def _dataset_sha256(path: Path) -> str:
    files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
    if not files:
        raise OpalError(f"retained prediction ledger has no parquet parts: {path}", ExitCodes.CONTRACT_VIOLATION)
    return "sha256:" + "|".join(file_sha256(file) for file in files)
