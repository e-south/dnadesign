"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/notebook.py

Generate/run campaign-tied marimo notebooks.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Optional

import typer

from ...analysis.facade import CampaignAnalysis, ensure_labels_path, ensure_predictions_dir, ensure_runs_path
from ...analysis.notebook_template import render_campaign_notebook
from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_group
from ._common import internal_error, opal_error, print_config_context

notebook_app = typer.Typer(no_args_is_help=True, help="Notebook workflows (marimo).")
cli_group("notebook", help="Notebook workflows (marimo).")(notebook_app)


@notebook_app.command("generate", help="Generate a campaign-tied marimo notebook.")
def cmd_notebook_generate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="campaign.yaml or campaign directory", envvar="OPAL_CONFIG"
    ),
    round: Optional[str] = typer.Option(
        "latest",
        "--round",
        "-r",
        help="Default round selector (int or 'latest').",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output notebook path (default: <workdir>/notebooks/opal_<slug>_analysis.py).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if the notebook already exists."),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate ledger artifacts exist before generating the notebook.",
    ),
) -> None:
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        cfg = analysis.config
        ws = analysis.workspace

        round_sel_raw = (round or "latest").strip().lower()
        if round_sel_raw in ("", "latest"):
            round_sel = "latest"
        else:
            try:
                round_val = int(round_sel_raw)
            except Exception as e:
                raise OpalError("Invalid --round: must be an integer or 'latest'.") from e
            round_sel = str(round_val)

        if validate:
            ensure_runs_path(ws.ledger_runs_path)
            ensure_predictions_dir(ws.ledger_predictions_dir)
            ensure_labels_path(ws.ledger_labels_path)
            runs_df = analysis.read_runs()
            # Validate requested round exists (or at least that runs are available for "latest").
            resolve_round_index_from_runs(runs_df, round_sel)

        default_out = ws.workdir / "notebooks" / f"opal_{cfg.campaign.slug}_analysis.py"
        out_path = Path(out) if out is not None else default_out
        if out_path.exists() and not force:
            raise OpalError(
                f"Notebook already exists: {out_path}. Use --force to overwrite.",
                ExitCodes.BAD_ARGS,
            )

        content = render_campaign_notebook(analysis.config_path, round_selector=round_sel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)

        print_config_context(analysis.config_path, cfg=cfg)
        print_stdout(f"Notebook written: {out_path}")
    except OpalError as e:
        opal_error("notebook.generate", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("notebook.generate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@notebook_app.command("run", help="Launch a marimo notebook (if installed).")
def cmd_notebook_run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="campaign.yaml or campaign directory", envvar="OPAL_CONFIG"
    ),
    path: Optional[Path] = typer.Option(None, "--path", help="Notebook path (defaults to generated path)."),
) -> None:
    try:
        if importlib.util.find_spec("marimo") is None:
            raise OpalError(
                "marimo is not installed. Install with `uv sync --group notebooks` or `uv pip install marimo`.",
                ExitCodes.BAD_ARGS,
            )

        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        cfg = analysis.config
        ws = analysis.workspace
        default_path = ws.workdir / "notebooks" / f"opal_{cfg.campaign.slug}_analysis.py"
        nb_path = Path(path) if path is not None else default_path
        if not nb_path.exists():
            raise OpalError(
                f"Notebook not found: {nb_path}. Run `opal notebook generate -c <campaign.yaml>` first.",
                ExitCodes.BAD_ARGS,
            )

        print_stdout(f"Launching marimo: {nb_path}")
        subprocess.run(["marimo", "edit", str(nb_path)], check=True)
    except OpalError as e:
        opal_error("notebook.run", e)
        raise typer.Exit(code=e.exit_code)
    except FileNotFoundError:
        opal_error(
            "notebook.run",
            OpalError("marimo CLI not found on PATH. Install marimo or use `uv run marimo edit ...`."),
        )
        raise typer.Exit(code=ExitCodes.BAD_ARGS)
    except Exception as e:
        internal_error("notebook.run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
