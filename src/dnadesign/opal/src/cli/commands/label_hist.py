"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/label_hist.py

Label history utilities (validate/repair).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...utils import ExitCodes, OpalError, print_stdout
from ..formatting import kv_block
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


@cli_command(
    "label-hist",
    help="Validate or repair the label_hist column (explicit, no silent fixes).",
)
def cmd_label_hist(
    action: str = typer.Argument(..., help="Action: validate | repair"),
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes (default: dry-run)."),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        action = str(action).strip().lower()
        if action in ("validate", "check"):
            store.validate_label_hist(df, require=True)
            out = {"ok": True, "action": "validate"}
            if json:
                json_out(out)
            else:
                print_stdout(kv_block("label-hist", out))
            return
        if action != "repair":
            raise OpalError("Unknown action. Use 'validate' or 'repair'.")

        cleaned, report = store.repair_label_hist(df)
        out = {
            "ok": True,
            "action": "repair",
            "rows_changed": report.get("rows_changed", 0),
            "entries_dropped": report.get("entries_dropped", 0),
            "applied": bool(apply),
        }
        if apply:
            store.save_atomic(cleaned)
        if json:
            json_out(out)
        else:
            print_stdout(kv_block("label-hist", out))
    except OpalError as e:
        opal_error("label-hist", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("label-hist", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
