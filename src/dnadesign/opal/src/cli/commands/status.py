"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/status.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...config.types import LabelSourceUSRSidecar, LocationUSR
from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.status import build_status
from ..formatting import render_status_text
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
    store_from_cfg,
)


def _label_source_config_status(cfg) -> dict[str, object]:
    source = cfg.labels.source
    if isinstance(source, LabelSourceUSRSidecar):
        location = cfg.data.location
        if not isinstance(location, LocationUSR):
            return {"kind": "usr_sidecar", "valid": False, "error": "usr_sidecar requires data.location.kind=usr"}
        path = Path(location.path) / source.dataset / source.path
        return {
            "kind": "usr_sidecar",
            "path": str(path),
            "y_space": cfg.labels.y_space,
            "id_column": cfg.labels.id_column,
            "round_column": cfg.labels.round_column,
            "batch_column": cfg.labels.batch_column,
            "dedup_policy": cfg.labels.dedup_policy,
            "exists": path.exists(),
            "valid": None,
            "label_count": 0,
            "available_rounds": [],
            "counts_by_round": {},
        }
    hist_col = f"opal__{cfg.campaign.slug}__label_hist"
    return {
        "kind": "campaign_history",
        "column": hist_col,
        "exists": None,
        "valid": None,
        "label_count": 0,
        "available_rounds": [],
        "counts_by_round": {},
    }


def _preinit_status(cfg, ws, store) -> dict[str, object]:
    return {
        "campaign_slug": cfg.campaign.slug,
        "campaign_name": cfg.campaign.name,
        "workdir": str(ws.workdir),
        "x_column_name": cfg.data.x_column_name,
        "y_column_name": cfg.data.y_column_name,
        "num_rounds": 0,
        "latest_round": None,
        "state_exists": ws.state_path.exists(),
        "state_path": str(ws.state_path),
        "data": {
            "records_path": str(store.records_path),
            "records_exists": store.records_path.exists(),
            "location_kind": store.kind,
        },
        "label_source": _label_source_config_status(cfg),
        "writeback": {"prediction_records": cfg.writeback.prediction_records},
    }


@cli_command("status", help="Dashboard from state.json (latest round by default).")
def cmd_status(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", help="Round selector: int or 'latest'."),
    all: bool = typer.Option(False, "--all"),
    with_ledger: bool = typer.Option(False, "--with-ledger", help="Include ledger summaries in output."),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        if not json:
            print_config_context(cfg_path, cfg=cfg)
        if all and round is not None:
            raise OpalError("Provide only one of --all or --round.")
        ledger_reader = None
        from ...storage.workspace import CampaignWorkspace

        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        if with_ledger:
            from ...storage.ledger import LedgerReader

            ledger_reader = LedgerReader(ws)
        store = store_from_cfg(cfg)
        if not ws.state_path.exists():
            st = _preinit_status(cfg, ws, store)
        else:
            round_k = resolve_round_index_from_state(ws.state_path, round) if round is not None else None
            try:
                st = build_status(
                    ws.state_path,
                    round_k=round_k,
                    show_all=all,
                    ledger_reader=ledger_reader,
                    include_ledger=with_ledger,
                )
            except ValueError as exc:
                raise OpalError(f"Failed to load state.json at {ws.state_path}: {exc}", ExitCodes.BAD_ARGS) from exc
            if "error" in st:
                raise OpalError(str(st["error"]))
        if store.records_path.exists():
            from ...storage.label_sources import label_source_status

            df = store.load_label_status_frame()
            st["label_source"] = label_source_status(cfg, store, df, strict=False)
            st["data"] = {
                "records_path": str(store.records_path),
                "records_exists": True,
                "location_kind": store.kind,
                "row_count": store.row_count(),
            }
        elif "label_source" not in st:
            st["label_source"] = _label_source_config_status(cfg)
            st["data"] = {
                "records_path": str(store.records_path),
                "records_exists": False,
                "location_kind": store.kind,
            }
        st["writeback"] = {"prediction_records": cfg.writeback.prediction_records}
        if json or all:
            json_out(st)
        else:
            print_stdout(render_status_text(st))
    except OpalError as e:
        opal_error("status", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("status", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
