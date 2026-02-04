"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/notebook.py

Provides notebook CLI commands for campaign analysis workflows. Generates and
launches marimo notebooks tied to campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ...analysis.facade import (
    CampaignAnalysis,
    ensure_labels_path,
    ensure_predictions_dir,
    ensure_runs_path,
)
from ...analysis.notebook_template import render_campaign_notebook
from ...core.pretty import console_out
from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_group
from ..tui import tui_enabled
from ._common import internal_error, opal_error, print_config_context, prompt_confirm

notebook_app = typer.Typer(no_args_is_help=False, help="Notebook workflows (marimo).")
cli_group("notebook", help="Notebook workflows (marimo).")(notebook_app)


def _list_notebooks(notebooks_dir: Path) -> list[Path]:
    if not notebooks_dir.exists():
        return []
    return sorted([p for p in notebooks_dir.glob("*.py") if p.is_file()])


def _notebook_rows(paths: list[Path]) -> list[str]:
    return [f"{idx}: {path.name}" for idx, path in enumerate(paths)]


def _format_notebook_choices(paths: list[Path]) -> str:
    return "\n".join(_notebook_rows(paths))


def _print_rich(obj: object) -> bool:
    console = console_out()
    if console is None:
        return False
    console.print(obj)
    return True


def _rich_kv_table(title: str, items: dict[str, object]):
    from rich import box
    from rich.table import Table

    table = Table(
        title=title,
        show_header=False,
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key, value in items.items():
        table.add_row(str(key), "" if value is None else str(value))
    return table


def _rich_list_table(title: str, rows: list[str]):
    from rich import box
    from rich.table import Table

    table = Table(
        title=title,
        show_header=False,
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("Item", overflow="fold")
    if not rows:
        table.add_row("(none)")
    else:
        for row in rows:
            table.add_row(str(row))
    return table


def _pick_notebook_interactive(paths: list[Path]) -> Path:
    if not sys.stdin.isatty():
        msg = "Multiple notebooks found but no TTY available. Re-run with --path to select one."
        raise OpalError(msg, ExitCodes.BAD_ARGS)
    rows = _notebook_rows(paths)
    if tui_enabled():
        table = _rich_list_table("Notebooks", rows)
        if _print_rich(table):
            pass
        else:
            print_stdout("Multiple notebooks found:\n" + "\n".join(rows))
    else:
        print_stdout("Multiple notebooks found:\n" + "\n".join(rows))
    resp = input("Select notebook index: ").strip()
    try:
        idx = int(resp)
    except Exception as e:
        raise OpalError("Invalid notebook index; expected an integer.", ExitCodes.BAD_ARGS) from e
    if idx < 0 or idx >= len(paths):
        raise OpalError("Notebook index out of range.", ExitCodes.BAD_ARGS)
    return paths[idx]


def _resolve_notebook_name(name: Optional[str], default_name: str) -> str:
    if not name:
        return default_name
    raw = str(name).strip()
    if not raw:
        return default_name
    if Path(raw).name != raw:
        raise OpalError("--name must be a file name, not a path.", ExitCodes.BAD_ARGS)
    suffix = Path(raw).suffix
    if suffix and suffix != ".py":
        raise OpalError("--name must end with .py (or omit the extension).", ExitCodes.BAD_ARGS)
    return raw if suffix else f"{raw}.py"


@notebook_app.callback(invoke_without_command=True)
def notebook_root(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
) -> None:
    if ctx.invoked_subcommand:
        return
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        ws = analysis.workspace
        notebooks_dir = ws.workdir / "notebooks"
        notebooks = _list_notebooks(notebooks_dir)
        if not notebooks:
            if tui_enabled():
                table = _rich_kv_table(
                    "Notebook",
                    {
                        "Status": "No notebooks found",
                        "Next step": f"uv run opal notebook generate -c {analysis.config_path} --round latest",
                        "Tip": "use --name to customize the notebook filename",
                    },
                )
                if _print_rich(table):
                    return
            print_stdout(
                "No notebooks found. Generate one with:\n"
                f"  uv run opal notebook generate -c {analysis.config_path} --round latest\n"
                "Tip: use --name to customize the notebook filename."
            )
            return
        if len(notebooks) == 1:
            if tui_enabled():
                table = _rich_kv_table(
                    "Notebook",
                    {
                        "Notebook": notebooks[0].name,
                        "Run": f"uv run opal notebook run -c {analysis.config_path}",
                    },
                )
                if _print_rich(table):
                    return
            print_stdout(
                "Notebook available:\n"
                f"  {notebooks[0].name}\n"
                "Run it with:\n"
                f"  uv run opal notebook run -c {analysis.config_path}"
            )
            return
        rows = _notebook_rows(notebooks)
        if tui_enabled():
            table = _rich_list_table("Notebooks", rows)
            if _print_rich(table):
                hint = _rich_kv_table(
                    "Next steps",
                    {
                        "Run": f"uv run opal notebook run -c {analysis.config_path}",
                        "Pick": "Or specify a file with --path",
                    },
                )
                _print_rich(hint)
                return
        print_stdout(
            "Multiple notebooks found:\n" + "\n".join(rows) + "\nRun with:\n"
            f"  uv run opal notebook run -c {analysis.config_path}\n"
            "Or specify a file with --path."
        )
    except OpalError as e:
        opal_error("notebook", e)
        raise typer.Exit(code=e.exit_code)


@notebook_app.command("generate", help="Generate a campaign-tied marimo notebook.")
def cmd_notebook_generate(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
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
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Notebook file name (defaults to opal_<slug>_analysis.py).",
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
        store = analysis.records_store()
        if not store.records_path.exists():
            raise OpalError(f"records.parquet not found: {store.records_path}", ExitCodes.BAD_ARGS)
        if out is not None and name is not None:
            raise OpalError("Use --out or --name, not both.", ExitCodes.BAD_ARGS)

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

        default_name = f"opal_{cfg.campaign.slug}_analysis.py"
        notebook_name = _resolve_notebook_name(name, default_name)
        default_out = ws.workdir / "notebooks" / notebook_name
        out_path = Path(out) if out is not None else default_out
        if out_path.exists() and not force:
            msg = (
                f"Notebook already exists: {out_path}. "
                "Use --force to overwrite or --name to choose a different filename."
            )
            try:
                confirmed = prompt_confirm(
                    f"{msg}\nOverwrite? (y/N): ",
                    non_interactive_hint=msg,
                )
            except OpalError:
                raise
            if not confirmed:
                print_stdout("Aborted.")
                return

        content = render_campaign_notebook(analysis.config_path, round_selector=round_sel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)

        if tui_enabled():
            table = _rich_kv_table(
                "Notebook Generated",
                {
                    "Config": analysis.config_path,
                    "Workdir": ws.workdir,
                    "Notebook": out_path,
                },
            )
            if _print_rich(table):
                hint = _rich_list_table(
                    "Next steps",
                    [f"uv run opal notebook run -c {analysis.config_path}"],
                )
                _print_rich(hint)
            else:
                print_config_context(analysis.config_path, cfg=cfg)
                print_stdout(f"Notebook written: {out_path}")
        else:
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
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    path: Optional[Path] = typer.Option(None, "--path", help="Notebook path (defaults to generated path)."),
) -> None:
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        ws = analysis.workspace
        notebooks_dir = ws.workdir / "notebooks"
        if path is None:
            notebooks = _list_notebooks(notebooks_dir)
            if not notebooks:
                raise OpalError(
                    (
                        f"No notebooks found in {notebooks_dir}. "
                        f"Run `uv run opal notebook generate -c {analysis.config_path}` first."
                    ),
                    ExitCodes.BAD_ARGS,
                )
            if len(notebooks) == 1:
                nb_path = notebooks[0]
            else:
                if sys.stdin.isatty():
                    nb_path = _pick_notebook_interactive(notebooks)
                else:
                    msg = (
                        "Multiple notebooks found:\n"
                        + _format_notebook_choices(notebooks)
                        + "\nUse --path to select one."
                    )
                    raise OpalError(msg, ExitCodes.BAD_ARGS)
        else:
            nb_path = Path(path)
            if not nb_path.is_absolute():
                nb_path = (Path.cwd() / nb_path).resolve()
            if not nb_path.exists():
                raise OpalError(
                    f"Notebook not found: {nb_path}. Run `uv run opal notebook generate -c <campaign.yaml>` first.",
                    ExitCodes.BAD_ARGS,
                )

        if importlib.util.find_spec("marimo") is None:
            raise OpalError(
                "marimo is not installed. Install with `uv sync --group notebooks` or `uv pip install marimo`.",
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
