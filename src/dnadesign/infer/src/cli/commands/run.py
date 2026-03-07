"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/run.py

Registration and implementation of the infer `run` CLI command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
import yaml

from ...api import run_job
from ...config import JobConfig, OutputSpec, RootConfig
from ...errors import ConfigError
from ...input_parsing import read_ids_arg
from ...presets import load_preset
from ...runtime.capacity_planner import probe_gpu_inventory, validate_model_hardware_contract
from ..builders import build_model_config, run_with_progress
from ..common import discovery_config, guard_pickle, raise_cli_error
from ..config_inputs import resolve_config_job_inputs
from ..console import (
    console,
    render_config_summary,
    render_outputs_spec_table,
    render_outputs_summary,
)


def _validate_run_mode_contract(
    *,
    config: Optional[Path],
    preset: Optional[str],
    job: List[str],
    usr: Optional[str],
    field: str,
    ids: Optional[str],
    usr_root: Optional[Path],
    write_back: bool,
) -> None:
    if preset:
        if config is not None:
            raise ConfigError("infer run accepts exactly one mode: use --config or --preset, not both.")
        if job:
            raise ConfigError("--job is only valid with --config.")
        return

    invalid_flags: list[str] = []
    if usr is not None:
        invalid_flags.append("--usr")
    if field != "sequence":
        invalid_flags.append("--field")
    if ids is not None:
        invalid_flags.append("--ids")
    if usr_root is not None:
        invalid_flags.append("--usr-root")
    if write_back:
        invalid_flags.append("--write-back")
    if invalid_flags:
        joined = ", ".join(invalid_flags)
        raise ConfigError(f"Preset-mode options are not allowed with --config: {joined}")


def register(app: typer.Typer) -> None:
    @app.command(help="Run jobs from a config OR a single job from a preset.")
    def run(
        config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
        preset: Optional[str] = typer.Option(None, "--preset", help="Run a single job from a named preset."),
        job: List[str] = typer.Option([], "--job", help="If --config is used: one or more job ids to run."),
        device: Optional[str] = typer.Option(None, "--device", help="Override model.device."),
        precision: Optional[str] = typer.Option(None, "--precision", help="Override model.precision."),
        batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override model.batch_size."),
        overwrite: Optional[bool] = typer.Option(None, "--overwrite/--no-overwrite", help="Override job.io.overwrite."),
        progress: bool = typer.Option(True, "--progress/--no-progress", help="Use progress bars."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print summary, then exit."),
        usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset (preset mode)."),
        field: str = typer.Option("sequence", "--field", help="USR field/column."),
        ids: Optional[str] = typer.Option(None, "--ids", help="CSV or file of ids (USR)."),
        usr_root: Optional[Path] = typer.Option(None, "--usr-root", help="USR datasets root."),
        write_back: bool = typer.Option(
            False,
            "--write-back/--no-write-back",
            help="Preset mode: attach outputs to USR.",
        ),
        i_know_this_is_pickle: bool = typer.Option(False, "--i-know-this-is-pickle", help="Needed only for pt_file."),
    ):
        try:
            _validate_run_mode_contract(
                config=config,
                preset=preset,
                job=job,
                usr=usr,
                field=field,
                ids=ids,
                usr_root=usr_root,
                write_back=write_back,
            )
            if preset:
                p = load_preset(preset)
                model = build_model_config(
                    model_id=p.get("model", {}).get("id"),
                    device=device,
                    precision=precision,
                    alphabet=None,
                    batch_size=batch_size,
                    preset_model=p.get("model", {}),
                )

                if p["kind"] == "extract":
                    outputs = [OutputSpec(**o) for o in p.get("outputs", [])]
                    job_cfg = JobConfig(
                        id=f"preset__{p['id'].split('/')[-1]}",
                        operation="extract",
                        ingest={
                            "source": "usr",
                            "dataset": usr,
                            "root": usr_root.as_posix() if usr_root else None,
                            "field": field,
                            "ids": read_ids_arg(ids),
                        },
                        outputs=outputs,
                        io={"write_back": write_back, "overwrite": bool(overwrite)},
                    )
                else:
                    job_cfg = JobConfig(
                        id=f"preset__{p['id'].split('/')[-1]}",
                        operation="generate",
                        ingest={
                            "source": "usr",
                            "dataset": usr,
                            "root": usr_root.as_posix() if usr_root else None,
                            "field": field,
                            "ids": read_ids_arg(ids),
                        },
                        params=p.get("params") or {},
                        io={"write_back": False, "overwrite": False},
                    )

                preset_inventory = probe_gpu_inventory()
                if not (dry_run and model.device.startswith("cuda") and preset_inventory.count == 0):
                    validate_model_hardware_contract(model=model, inventory=preset_inventory)

                if dry_run:
                    render_config_summary(model, [job_cfg])
                    if p["kind"] == "extract":
                        render_outputs_spec_table(p.get("outputs", []))
                    console.print("[green]✔ Preset validated (dry run).[/green]")
                    raise typer.Exit(code=0)

                result = run_with_progress(
                    progress=progress,
                    runner=lambda progress_factory: run_job(
                        inputs=None,
                        model=model,
                        job=job_cfg,
                        progress_factory=progress_factory,
                    ),
                )
                if p["kind"] == "extract":
                    render_outputs_summary(job_cfg, result)
                else:
                    console.print(f"[green]Generated {len(result.get('gen_seqs', []))} sequence(s).[/green]")
                raise typer.Exit(code=0)

            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
            jobs = root.jobs if not job else [j for j in root.jobs if j.id in set(job)]
            if not jobs:
                raise ConfigError("No jobs selected. Check --job or the config file.")

            model = root.model
            if device:
                model.device = device
            if precision:
                model.precision = precision  # type: ignore[assignment]
            if batch_size is not None:
                model.batch_size = batch_size
            if overwrite is not None:
                for selected_job in jobs:
                    selected_job.io.overwrite = overwrite

            config_inventory = probe_gpu_inventory()
            if not (dry_run and model.device.startswith("cuda") and config_inventory.count == 0):
                validate_model_hardware_contract(model=model, inventory=config_inventory)

            if dry_run:
                render_config_summary(model, jobs)
                console.print("[green]✔ Config validated (dry run).[/green]")
                raise typer.Exit(code=0)

            def _run_selected_jobs(progress_factory):
                for selected_job in jobs:
                    inputs = resolve_config_job_inputs(
                        job=selected_job,
                        config_dir=cfg_path.parent,
                        i_know_this_is_pickle=i_know_this_is_pickle,
                        guard_pickle=guard_pickle,
                    )
                    result = run_job(
                        inputs=inputs,
                        model=model,
                        job=selected_job,
                        progress_factory=progress_factory,
                    )
                    if selected_job.operation == "extract":
                        render_outputs_summary(selected_job, result)
                    else:
                        console.print(f"[green]Generated {len(result.get('gen_seqs', []))} sequence(s).[/green]")

            run_with_progress(progress=progress, runner=_run_selected_jobs)

        except typer.Exit:
            raise
        except Exception as error:
            raise_cli_error(error)
