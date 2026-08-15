"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/notebook_support.py

Support helpers for OPAL notebook CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from dnadesign.usr import require_explicit_usr_root

from ...analysis.campaign import CampaignAnalysis
from ...core.pretty import console_out
from ...core.utils import ExitCodes, OpalError, print_stdout
from ..tui import tui_enabled


def list_notebooks(notebooks_dir: Path) -> list[Path]:
    """Return generated marimo notebooks under a campaign notebook directory."""

    if not notebooks_dir.exists():
        return []
    return sorted([path for path in notebooks_dir.glob("*.py") if path.is_file()])


def notebook_rows(paths: list[Path]) -> list[str]:
    """Return numbered notebook rows for CLI display."""

    return [f"{idx}: {path.name}" for idx, path in enumerate(paths)]


def format_notebook_choices(paths: list[Path]) -> str:
    """Return numbered notebook choices as a newline-delimited string."""

    return "\n".join(notebook_rows(paths))


def print_rich(obj: object) -> bool:
    """Print a rich object when rich console output is enabled."""

    console = console_out()
    if console is None:
        return False
    console.print(obj)
    return True


def rich_kv_table(title: str, items: dict[str, object]):
    """Build a compact key-value rich table."""

    from rich import box
    from rich.table import Table

    table = Table(title=title, show_header=False, box=box.ROUNDED, border_style="cyan", title_style="bold cyan")
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key, value in items.items():
        table.add_row(str(key), "" if value is None else str(value))
    return table


def rich_list_table(title: str, rows: list[str]):
    """Build a compact one-column rich table."""

    from rich import box
    from rich.table import Table

    table = Table(title=title, show_header=False, box=box.ROUNDED, border_style="cyan", title_style="bold cyan")
    table.add_column("Item", overflow="fold")
    if not rows:
        table.add_row("(none)")
    else:
        for row in rows:
            table.add_row(str(row))
    return table


def pick_notebook_interactive(paths: list[Path]) -> Path:
    """Prompt for a notebook path when multiple generated notebooks exist."""

    if not sys.stdin.isatty():
        raise OpalError(
            "Multiple notebooks found but no TTY available. Re-run with --path to select one.",
            ExitCodes.BAD_ARGS,
        )
    rows = notebook_rows(paths)
    if tui_enabled():
        table = rich_list_table("Notebooks", rows)
        if print_rich(table):
            pass
        else:
            print_stdout("Multiple notebooks found:\n" + "\n".join(rows))
    else:
        print_stdout("Multiple notebooks found:\n" + "\n".join(rows))
    resp = input("Select notebook index: ").strip()
    try:
        idx = int(resp)
    except Exception as exc:
        raise OpalError("Invalid notebook index; expected an integer.", ExitCodes.BAD_ARGS) from exc
    if idx < 0 or idx >= len(paths):
        raise OpalError("Notebook index out of range.", ExitCodes.BAD_ARGS)
    return paths[idx]


def resolve_notebook_path(analysis: CampaignAnalysis, path: Path | None) -> Path:
    """Resolve a generated notebook path for run/edit commands."""

    notebooks_dir = analysis.workspace.workdir / "notebooks"
    if path is None:
        notebooks = list_notebooks(notebooks_dir)
        if not notebooks:
            raise OpalError(
                (
                    f"No notebooks found in {notebooks_dir}. "
                    f"Run `uv run opal notebook generate -c {analysis.config_path}` first."
                ),
                ExitCodes.BAD_ARGS,
            )
        if len(notebooks) == 1:
            return notebooks[0]
        if sys.stdin.isatty():
            return pick_notebook_interactive(notebooks)
        msg = "Multiple notebooks found:\n" + format_notebook_choices(notebooks) + "\nUse --path to select one."
        raise OpalError(msg, ExitCodes.BAD_ARGS)

    nb_path = Path(path)
    if not nb_path.is_absolute():
        nb_path = (Path.cwd() / nb_path).resolve()
    if not nb_path.exists():
        raise OpalError(
            f"Notebook not found: {nb_path}. Run `uv run opal notebook generate -c <campaign.yaml>` first.",
            ExitCodes.BAD_ARGS,
        )
    return nb_path


def marimo_command(
    *,
    mode: str,
    notebook_path: Path,
    host: str | None,
    port: int | None,
    headless: bool,
) -> list[str]:
    """Build the marimo run/edit command."""

    if mode not in {"run", "edit"}:
        raise ValueError(f"Unsupported marimo notebook mode: {mode}")
    command = ["marimo", mode, str(notebook_path)]
    if host is not None:
        command.extend(["--host", str(host)])
    if port is not None:
        command.extend(["--port", str(port)])
    if headless:
        command.append("--headless")
    return command


def launch_marimo_notebook(
    *,
    mode: str,
    notebook_path: Path,
    host: str | None,
    port: int | None,
    headless: bool,
    usr_root: str | Path | None = None,
) -> None:
    """Launch a marimo notebook with a local marimo installation."""

    if importlib.util.find_spec("marimo") is None:
        command_hint = "run" if mode == "run" else "edit"
        raise OpalError(
            f"marimo is not installed. Install with `uv sync --locked` or use `uv run marimo {command_hint} ...`.",
            ExitCodes.BAD_ARGS,
        )
    subprocess.run(
        marimo_command(
            mode=mode,
            notebook_path=notebook_path,
            host=host,
            port=port,
            headless=headless,
        ),
        check=True,
        env=marimo_subprocess_environment(usr_root),
    )


def marimo_subprocess_environment(
    usr_root: str | Path | None,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Bind an explicit USR coordinate into a marimo child process."""

    environment = dict(os.environ if base_environment is None else base_environment)
    if usr_root is not None:
        environment["OPAL_NOTEBOOK_USR_ROOT"] = str(require_explicit_usr_root(usr_root))
    return environment


def resolve_notebook_name(name: Optional[str], default_name: str) -> str:
    """Resolve the generated notebook filename."""

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


def parse_notebook_round_selector(round_value: str | None, *, allow_all: bool) -> str:
    """Parse a notebook round selector from CLI input."""

    raw = (round_value or "latest").strip().lower()
    if raw in ("", "latest"):
        return "latest"
    if allow_all and raw == "all":
        return "all"
    try:
        round_index = int(raw)
    except Exception as exc:
        accepted = "an integer, 'latest', or 'all'" if allow_all else "an integer or 'latest'"
        raise OpalError(f"Invalid --round: must be {accepted}.", ExitCodes.BAD_ARGS) from exc
    return str(round_index)


__all__ = [
    "format_notebook_choices",
    "launch_marimo_notebook",
    "marimo_subprocess_environment",
    "list_notebooks",
    "notebook_rows",
    "parse_notebook_round_selector",
    "print_rich",
    "resolve_notebook_name",
    "resolve_notebook_path",
    "rich_kv_table",
    "rich_list_table",
]
