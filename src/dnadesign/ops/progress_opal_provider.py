"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress_opal_provider.py

Provider-owned OPAL status builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .progress_support import load_yaml_mapping, required_path


def provide_opal_campaign_state_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    del repo_root
    return opal_campaign_state_progress(
        opal_config=inputs.get("opal_config"),
        opal_workdir=inputs.get("opal_workdir"),
    )


def opal_campaign_state_progress(
    *,
    opal_config: Path | None,
    opal_workdir: Path | None,
) -> tuple[str, str, dict[str, object]]:
    if opal_workdir is None:
        resolved_config = required_path(
            opal_config,
            flag_name="--opal-config or --opal-workdir",
            progress_kind="opal-campaign-state",
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
    if not state_path.exists():
        return (
            "missing",
            "OPAL state.json not found",
            {
                "opal_workdir": str(workdir),
                "opal_config": str(config_path) if config_path is not None else None,
                "state_path": str(state_path),
                "ledger_runs_path": str(ledger_runs_path),
            },
        )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    rounds = sorted(
        list(payload.get("rounds") or []),
        key=lambda round_payload: int(round_payload.get("round_index", -1)),
    )
    latest_round = rounds[-1] if rounds else None
    num_rounds = len(rounds)
    campaign_slug = str(payload.get("campaign_slug") or payload.get("slug") or "")
    if num_rounds == 0:
        summary = "OPAL campaign initialized with no completed rounds yet"
        state = "attention"
    else:
        summary = f"OPAL campaign has {num_rounds} recorded rounds; latest round {latest_round.get('round_index')}"
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
            "campaign_slug": campaign_slug,
            "campaign_name": payload.get("campaign_name") or payload.get("name"),
            "x_column_name": payload.get("x_column_name"),
            "y_column_name": payload.get("y_column_name"),
            "num_rounds": num_rounds,
            "latest_round": {
                "round_index": latest_round.get("round_index"),
                "run_id": latest_round.get("run_id"),
                "round_dir": latest_round.get("round_dir"),
                "selection_top_k_requested": latest_round.get("selection_top_k_requested"),
                "selection_top_k_effective_after_ties": latest_round.get("selection_top_k_effective_after_ties"),
            }
            if latest_round is not None
            else None,
        },
    )


def _resolve_opal_workdir(*, opal_config: Path | None, opal_workdir: Path | None) -> tuple[Path, Path | None]:
    if opal_workdir is not None:
        resolved_config = opal_config.expanduser().resolve() if opal_config else None
        return opal_workdir.expanduser().resolve(), resolved_config
    resolved_config = required_path(
        opal_config,
        flag_name="--opal-config or --opal-workdir",
        progress_kind="opal-campaign-state",
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


def _resolve_opal_campaign_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


__all__ = [
    "opal_campaign_state_progress",
    "provide_opal_campaign_state_status",
]
