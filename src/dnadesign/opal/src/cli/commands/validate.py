"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/validate.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

import typer

from ...config import load_config
from ...data_access import ESSENTIAL_COLS
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, resolve_config_path, store_from_cfg


@cli_command(
    "validate", help="End-to-end table checks (essentials present; X column present)."
)
def cmd_validate(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG")
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()

        # Always print absolute context so there's no ambiguity
        cfg_abs = str(Path(cfg_path).resolve())
        wd_abs = str(Path(cfg.campaign.workdir).resolve())
        rec_abs = str(store.records_path.resolve())
        print_stdout(f"Config:  {cfg_abs}")
        print_stdout(f"Workdir: {wd_abs}")
        print_stdout(f"Records: {rec_abs}")
        print_stdout(f"X in YAML: {cfg.data.x_column_name}")
        print_stdout(f"Y in YAML: {cfg.data.y_column_name}")
        print_stdout(f"Table shape: {df.shape[0]} rows × {df.shape[1]} cols")

        missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
        if missing:
            raise OpalError(f"Missing essential columns: {missing}")
        if cfg.data.x_column_name not in df.columns:
            # Helpful hints: case-only match and fuzzy suggestions
            target = cfg.data.x_column_name
            cols = list(map(str, df.columns))
            case_only = [c for c in cols if c.lower() == target.lower()]
            fuzzy = get_close_matches(target, cols, n=5, cutoff=0.6)
            hint_lines = []
            if case_only and target not in case_only:
                hint_lines.append(f"Case-only match found: {case_only[0]!r}")
            if fuzzy:
                hint_lines.append(
                    "Similar columns: " + ", ".join(repr(c) for c in fuzzy)
                )
            hint = (" " + " | ".join(hint_lines)) if hint_lines else ""
            raise OpalError(f"Missing X column: {target}.{hint}")
        print_stdout("OK: validation passed.")

        # Hint when the user is not inside the campaign workspace
        try:
            cwd = Path.cwd().resolve()
            wd = Path(cfg.campaign.workdir).resolve()
            if wd not in cwd.parents and cwd != wd:
                print_stdout(
                    f"OK: validation passed. (Note: your CWD '{cwd}' is outside campaign workdir '{wd}')"
                )
                return
        except Exception:
            pass

    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("validate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
