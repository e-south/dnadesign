"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/review.py

CLI command for writing campaign-scoped review artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_error, json_out, opal_error


@cli_command("review", help="Persist campaign review artifacts under outputs/review.")
def cmd_review(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    round: Optional[str] = typer.Option("latest", "--round", "-r", help="Round selector: int or latest."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to review."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Review output directory."),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Write portable review plots."),
    json: bool = typer.Option(False, "--json/--text", help="Output format."),
) -> None:
    try:
        from ...reporting.review import build_campaign_review

        result = build_campaign_review(
            config,
            round_selector=round,
            run_id=run_id,
            out_dir=out_dir,
            include_plots=plots,
        )
        if json:
            json_out(result.to_dict())
        else:
            print_stdout(
                "\n".join(
                    [
                        "OPAL campaign review written",
                        f"manifest: {result.manifest_path}",
                        f"review: {result.review_path}",
                        f"index: {result.index_path}",
                        f"plots: {len(result.plot_paths)}",
                    ]
                )
            )
    except OpalError as e:
        if json:
            json_error("review", e)
        else:
            opal_error("review", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("review", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
