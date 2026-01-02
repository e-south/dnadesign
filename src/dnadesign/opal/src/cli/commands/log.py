"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/log.py

Summarize round.log.jsonl for a given round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.summary import load_round_log, summarize_round_log
from ...storage.workspace import CampaignWorkspace
from ..formatting import render_round_log_summary_human
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error, print_config_context, resolve_config_path


@cli_command("log", help="Summarize round.log.jsonl for a given round.")
def cmd_log(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option("latest", "--round", "-r"),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        r = resolve_round_index_from_state(ws.state_path, round)
        path = ws.round_dir(r) / "round.log.jsonl"
        events = load_round_log(path)
        summary = summarize_round_log(events)
        summary["round_index"] = int(r)
        summary["path"] = str(path.resolve())
        if json:
            json_out(summary)
        else:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout(render_round_log_summary_human(summary))
    except OpalError as e:
        opal_error("log", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("log", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
