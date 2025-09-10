"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/validate.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()
        missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
        if missing:
            raise OpalError(f"Missing essential columns: {missing}")
        if cfg.data.representation_column_name not in df.columns:
            raise OpalError(
                f"Missing representation column: {cfg.data.representation_column_name}"
            )
        print_stdout("OK: validation passed.")
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("validate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
