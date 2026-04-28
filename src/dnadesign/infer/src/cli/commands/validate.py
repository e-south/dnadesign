"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/validate.py

Registration for infer validation CLI command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import typer
import yaml

from dnadesign.infer import plan_sequence_view_feature_completion_from_config

from ...config import RootConfig
from ...ingest.sources import load_usr_input
from ...input_parsing import read_ids_arg
from ...runtime.adapter_runtime import validate_adapter_runtime_contract
from ...runtime.capacity_planner import probe_gpu_inventory, validate_model_hardware_contract
from ...usr_registry import derive_usr_registry_spec
from ..common import discovery_config, raise_cli_error
from ..config_inputs import resolve_config_usr_root
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
                    "Run this check on a GPU node or use ops runbook planning "
                    "for declared scheduler resources.[/yellow]"
                )
            else:
                validate_model_hardware_contract(model=root.model, inventory=inventory)
            validate_adapter_runtime_contract(model=root.model)
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

    @validate_app.command(
        "usr-registry",
        help="Render the required USR namespace registration spec for infer write-back jobs.",
    )
    def validate_usr_registry(
        config: Optional[Path] = typer.Option(None, "--config"),
        job: Optional[str] = typer.Option(None, "--job", help="Restrict to one job id."),
    ) -> None:
        try:
            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
            for selected_job in root.jobs:
                if selected_job.ingest.source != "usr":
                    continue
                selected_job.ingest.root = resolve_config_usr_root(
                    usr_root=selected_job.ingest.root,
                    config_dir=cfg_path.parent,
                )
            spec = derive_usr_registry_spec(root=root, job_id=job)
            typer.echo(f"namespace: {spec.namespace}")
            typer.echo(f"root: {spec.root}")
            typer.echo(f"columns: {spec.columns_spec}")
            typer.echo(f"register: {spec.register_command}")
        except Exception as error:
            raise_cli_error(error)

    @validate_app.command(
        "sequence-view-completion",
        help="Classify sequence-view feature work as reusable, stale, missing, or blocked before GPU execution.",
    )
    def validate_sequence_view_completion(
        config: Optional[Path] = typer.Option(None, "--config"),
        job: Optional[str] = typer.Option(None, "--job", help="Restrict to one feature-bundle job id."),
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
        max_missing_vectors: Optional[int] = typer.Option(
            None,
            "--max-missing-vectors",
            help="Fail if missing feature-vector count exceeds this threshold.",
        ),
        max_stale_vectors: Optional[int] = typer.Option(
            None,
            "--max-stale-vectors",
            help="Fail if stale or unclassified feature-vector count exceeds this threshold.",
        ),
        max_missing_products: Optional[int] = typer.Option(
            None,
            "--max-missing-products",
            help="Fail if missing sequence-product selector count exceeds this threshold.",
        ),
    ) -> None:
        try:
            cfg_path = discovery_config(config)
            plans = list(plan_sequence_view_feature_completion_from_config(cfg_path, job=job))
            violations = _sequence_view_completion_threshold_violations(
                plans=plans,
                max_missing_vectors=max_missing_vectors,
                max_stale_vectors=max_stale_vectors,
                max_missing_products=max_missing_products,
            )
            if violations:
                raise ValueError("sequence-view completion thresholds failed: " + "; ".join(violations))

            if fmt == "json":
                typer.echo(json.dumps(plans, sort_keys=True, separators=(",", ":")))
                return
            if fmt != "text":
                raise ValueError("format must be one of: text, json.")
            for plan in plans:
                typer.echo(
                    "sequence-view-completion "
                    f"job={plan['bundle_id']} dataset={plan['dataset']} required_views={plan['required_views']} "
                    f"required_vectors={plan['required_vectors']} reusable={plan['reusable_vectors']} "
                    f"stale={plan['stale_vectors']} missing={plan['missing_vectors']}"
                )
        except Exception as error:
            raise_cli_error(error)


def _sequence_view_completion_threshold_violations(
    *,
    plans: Sequence[Mapping[str, object]],
    max_missing_vectors: int | None,
    max_stale_vectors: int | None,
    max_missing_products: int | None,
) -> list[str]:
    thresholds = (
        ("missing_vectors", max_missing_vectors),
        ("stale_vectors", max_stale_vectors),
        ("missing_products", max_missing_products),
    )
    violations: list[str] = []
    for field, threshold in thresholds:
        if threshold is None:
            continue
        if int(threshold) < 0:
            raise ValueError(f"{field} threshold must be >= 0")
        observed = sum(_plan_int(plan, field) for plan in plans)
        if observed > int(threshold):
            violations.append(f"{field}={observed} exceeds max_{field}={threshold}")
    return violations


def _plan_int(plan: Mapping[str, object], field: str) -> int:
    value = plan.get(field, 0)
    if isinstance(value, bool):
        raise ValueError(f"planner field {field} must be an integer, not boolean")
    return int(value or 0)
