"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/history.py

Exposes verified OPAL campaign-history inspection and relocation commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...storage.history_relocation.inspection import plan_history_relocation
from ...storage.history_relocation.materialization import apply_history_relocation
from ...storage.history_relocation.state_projection import require_target_config_matches_run_history
from ..formatting import kv_block
from ..registry import cli_group
from ._common import (
    internal_error,
    json_error,
    json_out,
    load_cli_config,
    opal_error,
    resolve_config_path,
    store_from_cfg,
)

history_app = typer.Typer(no_args_is_help=True, help="Inspect and relocate one campaign history.")
cli_group("history", help="Inspect and relocate one campaign history.")(history_app)


@history_app.command("import", help="Import disjoint prior rounds into the configured campaign history.")
def history_import(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="Target campaign.yaml."),
    source_workdir: Path = typer.Option(..., "--source-workdir", file_okay=False, dir_okay=True),
    apply: bool = typer.Option(False, "--apply", help="Apply the verified relocation plan."),
    json: bool = typer.Option(False, "--json/--text", help="Output format."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        target_workdir = Path(cfg.campaign.workdir)
        plan = plan_history_relocation(
            source_workdir=source_workdir,
            target_workdir=target_workdir,
            expected_slug=cfg.campaign.slug,
        )
        require_target_config_matches_run_history(plan, cfg)
        payload: dict[str, object] = {
            "schema_version": "opal.history_import.v1",
            "campaign_slug": plan.campaign_slug,
            "imported_rounds": list(plan.imported_rounds),
            "existing_rounds": list(plan.existing_rounds),
            "canonical_rounds": list(plan.canonical_rounds),
            "run_invariant_sha256": plan.invariant_sha256,
            "applied": False,
        }
        if apply:
            store = store_from_cfg(cfg)
            receipt_path = apply_history_relocation(plan, cfg=cfg, records_path=store.records_path)
            payload["applied"] = True
            payload["receipt_path"] = str(receipt_path)
        if json:
            json_out(payload)
        else:
            print_stdout(kv_block("history import", payload))
    except OpalError as exc:
        if json:
            json_error("history import", exc)
        else:
            opal_error("history import", exc)
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        internal_error("history import", exc)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
