"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/ops/status_providers.py

Provider-owned OPAL status builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.ops.status.artifacts import load_yaml_mapping
from dnadesign.ops.status.paths import required_path

from ..storage.state import CampaignState, RoundEntry


def provide_opal_campaign_state_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return _opal_campaign_state_status(
        opal_config=inputs.get("opal_config"),
        opal_workdir=inputs.get("opal_workdir"),
    )


def _opal_campaign_state_status(
    *,
    opal_config: Path | None,
    opal_workdir: Path | None,
) -> tuple[str, str, dict[str, object]]:
    if opal_workdir is None:
        resolved_config = required_path(
            opal_config,
            flag_name="--opal-config or --opal-workdir",
            status_kind="opal-campaign-state",
        )
        if not resolved_config.exists():
            inferred_workdir = _resolve_opal_campaign_root(resolved_config)
            return (
                "missing",
                "OPAL config not found",
                {
                    "opal_workdir": str(inferred_workdir),
                    "opal_config": str(resolved_config),
                    "state_path": str(inferred_workdir / "state.json"),
                    "ledger_runs_path": str(inferred_workdir / "outputs" / "ledger" / "runs.parquet"),
                },
            )
    workdir, config_path = _resolve_opal_workdir(opal_config=opal_config, opal_workdir=opal_workdir)
    state_path = workdir / "state.json"
    ledger_runs_path = workdir / "outputs" / "ledger" / "runs.parquet"
    candidate_records = _resolve_opal_candidate_records_path(config_path)
    if candidate_records is not None and not candidate_records["records_path"].exists():
        return (
            "missing",
            "OPAL candidate records.parquet not found",
            {
                "opal_workdir": str(workdir),
                "opal_config": str(config_path) if config_path is not None else None,
                "state_path": str(state_path),
                "ledger_runs_path": str(ledger_runs_path),
                "records_path": str(candidate_records["records_path"]),
                "data_location_kind": candidate_records["kind"],
                "dataset": candidate_records.get("dataset"),
            },
        )
    candidate_evidence = _opal_candidate_records_evidence(candidate_records)
    if not state_path.exists():
        return (
            "missing",
            "OPAL state.json not found; candidate records.parquet exists"
            if candidate_evidence
            else "OPAL state.json not found",
            {
                "opal_workdir": str(workdir),
                "opal_config": str(config_path) if config_path is not None else None,
                "state_path": str(state_path),
                "ledger_runs_path": str(ledger_runs_path),
                **candidate_evidence,
            },
        )

    try:
        campaign_state = CampaignState.load(state_path)
    except Exception as exc:
        return (
            "attention",
            "OPAL state.json is not loadable",
            {
                "opal_workdir": str(workdir),
                "opal_config": str(config_path) if config_path is not None else None,
                "state_path": str(state_path),
                "ledger_runs_path": str(ledger_runs_path),
                "ledger_runs_present": ledger_runs_path.exists(),
                **candidate_evidence,
                "state_load_error": str(exc),
            },
        )

    rounds = sorted(campaign_state.rounds, key=lambda round_entry: int(round_entry.round_index))
    latest_round = rounds[-1] if rounds else None
    num_rounds = len(rounds)
    campaign_slug = str(campaign_state.campaign_slug or "")
    if num_rounds == 0:
        summary = "OPAL campaign initialized with no completed rounds yet"
        state = "attention"
    else:
        summary = f"OPAL campaign has {num_rounds} recorded rounds; latest round {latest_round.round_index}"
        state = "ok"
    return (
        state,
        summary,
        {
            "opal_workdir": str(workdir),
            "opal_config": str(config_path) if config_path is not None else None,
            "state_path": str(state_path),
            "ledger_runs_path": str(ledger_runs_path),
            "ledger_runs_present": ledger_runs_path.exists(),
            **candidate_evidence,
            "campaign_slug": campaign_slug,
            "campaign_name": campaign_state.campaign_name,
            "x_column_name": campaign_state.x_column_name,
            "y_column_name": campaign_state.y_column_name,
            "num_rounds": num_rounds,
            "latest_round": _round_status_evidence(latest_round),
        },
    )


def _round_status_evidence(round_entry: RoundEntry | None) -> dict[str, object] | None:
    if round_entry is None:
        return None
    return {
        "round_index": round_entry.round_index,
        "run_id": round_entry.run_id,
        "round_dir": round_entry.round_dir,
        "selection_views": round_entry.selection_views,
        "selection_batch": round_entry.selection_batch,
    }


def _opal_candidate_records_evidence(candidate_records: dict[str, object] | None) -> dict[str, object]:
    if candidate_records is None:
        return {}
    records_path = candidate_records["records_path"]
    if not isinstance(records_path, Path) or not records_path.exists():
        return {}
    try:
        row_count: int | None = int(pq.ParquetFile(records_path).metadata.num_rows)
    except Exception:
        row_count = None
    return {
        "records_path": str(records_path),
        "records_present": True,
        "records_row_count": row_count,
        "data_location_kind": candidate_records["kind"],
        "dataset": candidate_records.get("dataset"),
    }


def _resolve_opal_workdir(*, opal_config: Path | None, opal_workdir: Path | None) -> tuple[Path, Path | None]:
    if opal_workdir is not None:
        resolved_config = opal_config.expanduser().resolve() if opal_config else None
        return opal_workdir.expanduser().resolve(), resolved_config
    resolved_config = required_path(
        opal_config,
        flag_name="--opal-config or --opal-workdir",
        status_kind="opal-campaign-state",
    )
    if not resolved_config.exists():
        raise ValueError(f"OPAL config not found: {resolved_config}")
    payload = load_yaml_mapping(resolved_config, label="OPAL config")
    campaign_payload = payload.get("campaign")
    if not isinstance(campaign_payload, dict):
        raise ValueError(f"OPAL config missing 'campaign' mapping: {resolved_config}")
    workdir = str(campaign_payload.get("workdir") or "").strip()
    if not workdir:
        raise ValueError(f"OPAL config missing campaign.workdir: {resolved_config}")
    return _resolve_opal_config_workdir(config_path=resolved_config, workdir=workdir), resolved_config


def _resolve_opal_config_workdir(*, config_path: Path, workdir: str) -> Path:
    workdir_path = Path(workdir).expanduser()
    if workdir_path.is_absolute():
        return workdir_path.resolve()
    campaign_root = _resolve_opal_campaign_root(config_path)
    return (campaign_root / workdir_path).resolve()


def _resolve_opal_candidate_records_path(config_path: Path | None) -> dict[str, object] | None:
    if config_path is None:
        return None
    payload = load_yaml_mapping(config_path, label="OPAL config")
    data_payload = payload.get("data")
    if not isinstance(data_payload, Mapping):
        return None
    location = data_payload.get("location")
    if not isinstance(location, Mapping):
        return None

    kind = str(location.get("kind") or "").strip()
    raw_path = str(location.get("path") or "").strip()
    if not kind or not raw_path:
        return None
    base_path = _resolve_opal_config_path_like(config_path=config_path, raw_path=raw_path)
    if kind == "usr":
        dataset = str(location.get("dataset") or "").strip()
        if not dataset:
            return None
        return {
            "kind": kind,
            "dataset": dataset,
            "records_path": base_path / dataset / "records.parquet",
        }
    if kind == "local":
        return {
            "kind": kind,
            "records_path": base_path,
        }
    return None


def _resolve_opal_config_path_like(*, config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_resolve_opal_campaign_root(config_path) / path).resolve()


def _resolve_opal_campaign_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


__all__ = ["provide_opal_campaign_state_status"]
