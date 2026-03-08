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
from ...runtime.capacity_planner import probe_gpu_inventory, validate_model_hardware_contract
from ...usr_registry import derive_usr_registry_spec
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
            inventory = probe_gpu_inventory()
            if root.model.device.startswith("cuda") and inventory.count == 0:
                console.print(
                    "[yellow]Capacity check skipped: no local GPU inventory detected. "
                    "Run this check on a GPU node or use ops runbook planning for declared scheduler resources.[/yellow]"
                )
            else:
                validate_model_hardware_contract(model=root.model, inventory=inventory)
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

    @validate_app.command("usr-registry", help="Render the required USR namespace registration spec for infer write-back jobs.")
    def validate_usr_registry(
        config: Optional[Path] = typer.Option(None, "--config"),
        job: Optional[str] = typer.Option(None, "--job", help="Restrict to one job id."),
    ) -> None:
        try:
            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
            spec = derive_usr_registry_spec(root=root, job_id=job)
            typer.echo(f"namespace: {spec.namespace}")
            typer.echo(f"root: {spec.root}")
            typer.echo(f"columns: {spec.columns_spec}")
            typer.echo(f"register: {spec.register_command}")
        except Exception as error:
            raise_cli_error(error)
