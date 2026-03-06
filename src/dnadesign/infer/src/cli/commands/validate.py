"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/validate.py

Registration for infer validation CLI command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from ...config import RootConfig
from ...input_parsing import read_ids_arg
from ...ingest.sources import load_usr_input
from ..common import discovery_config, raise_cli_error
from ..console import console, render_config_summary


def register(app: typer.Typer) -> None:
    validate_app = typer.Typer(no_args_is_help=False, help="Validation utilities.")
    app.add_typer(validate_app, name="validate")

    @validate_app.command("config", help="Validate a config file (default discovery if omitted).")
    def validate_config(config: Optional[Path] = typer.Option(None, "--config")) -> None:
        try:
            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
            render_config_summary(root.model, root.jobs)
            console.print("[green]✔ Config validated.[/green]")
        except Exception as error:
            raise_cli_error(error)

    @validate_app.command("usr", help="Validate a USR dataset can be read (id + field).")
    def validate_usr(
        dataset: str = typer.Option(..., "--dataset"),
        field: str = typer.Option("sequence", "--field"),
        usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
        ids: Optional[str] = typer.Option(None, "--ids", help="Path or CSV of ids to subset"),
    ) -> None:
        try:
            seqs, _id_list, ds = load_usr_input(
                dataset_name=dataset,
                field=field,
                root=(usr_root.as_posix() if usr_root else None),
                ids=read_ids_arg(ids),
            )
            console.print(f"[green]✔ USR OK[/green]  dataset={dataset}  rows={len(seqs)}  field={field}")
            console.print(f"[accent]records:[/accent] {ds.records_path}")
        except Exception as error:
            raise_cli_error(error)
