"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/run.py

Run execution CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import typer

from ..cli_commands.context import CliContext
from ..core.pipeline import resolve_plan, run_pipeline
from ..core.run_paths import has_existing_run_outputs, run_outputs_root
from ..utils.logging_utils import install_native_stderr_filters, setup_logging
from ..utils.mpl_utils import ensure_mpl_cache_dir


def register_run_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    render_missing_input_hint: Callable[..., None],
    render_output_schema_hint: Callable[..., bool],
    ensure_fimo_available: Callable[..., None],
) -> None:
    console = context.console

    @app.command(help="Run generation for the job. Optionally auto-run plots declared in YAML.")
    def run(
        ctx: typer.Context,
        no_plot: bool = typer.Option(False, help="Do not auto-run plots even if configured."),
        fresh: bool = typer.Option(False, "--fresh", help="Clear outputs and start a new run."),
        resume: bool = typer.Option(False, "--resume", help="Resume from existing outputs."),
        rebuild_stage_a: bool = typer.Option(
            False,
            "--rebuild-stage-a",
            help="Rebuild Stage-A pools before running (required if pools are missing or stale).",
        ),
        log_file: Optional[Path] = typer.Option(
            None,
            help="Override logfile path (must be inside outputs/ under the run root).",
        ),
        show_tfbs: bool = typer.Option(False, "--show-tfbs", help="Show TFBS sequences in progress output."),
        show_solutions: bool = typer.Option(False, "--show-solutions", help="Show full solution sequences in output."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        root = loaded.root
        cfg = root.densegen
        run_root = context.run_root_for(loaded)
        if rebuild_stage_a:
            ensure_fimo_available(cfg, strict=True)

        if fresh and resume:
            console.print("[bold red]Choose either --fresh or --resume, not both.[/]")
            raise typer.Exit(code=1)
        outputs_root = run_outputs_root(run_root)
        existing_outputs = has_existing_run_outputs(run_root)
        if fresh:
            if outputs_root.exists():
                try:
                    shutil.rmtree(outputs_root)
                except Exception as exc:
                    console.print(f"[bold red]Failed to clear outputs:[/] {exc}")
                    raise typer.Exit(code=1)
                console.print(
                    ":broom: [bold yellow]Cleared outputs[/]: "
                    f"{context.display_path(outputs_root, run_root, absolute=False)}"
                )
            else:
                console.print("[yellow]No outputs directory found; starting fresh.[/]")
            resume_run = False
        elif resume:
            if not existing_outputs:
                console.print(
                    f"[bold red]--resume requested but no outputs were found under[/] "
                    f"{context.display_path(outputs_root, run_root, absolute=False)}. "
                    "Run without --resume or use --fresh to reset the workspace."
                )
                raise typer.Exit(code=1)
            resume_run = True
        else:
            if existing_outputs:
                console.print(
                    f"[bold red]Existing outputs found under[/] "
                    f"{context.display_path(outputs_root, run_root, absolute=False)}. "
                    "Use --resume to continue or --fresh to clear outputs."
                )
                raise typer.Exit(code=1)
            resume_run = False

        # Logging setup
        log_cfg = cfg.logging
        log_dir = context.resolve_outputs_path_or_exit(
            loaded.path,
            run_root,
            Path(log_cfg.log_dir),
            label="logging.log_dir",
        )
        default_logfile = log_dir / f"{cfg.run.id}.log"
        if log_file is not None:
            logfile = context.resolve_outputs_path_or_exit(loaded.path, run_root, log_file, label="logging.log_file")
        else:
            logfile = default_logfile
        setup_logging(
            level=log_cfg.level,
            logfile=str(logfile),
            suppress_solver_stderr=bool(log_cfg.suppress_solver_stderr),
        )

        # Plan & solver
        pl = resolve_plan(loaded)
        console.print("[bold]Quota plan[/]: " + ", ".join(f"{p.name}={p.quota}" for p in pl))
        try:
            run_pipeline(
                loaded,
                resume=resume_run,
                build_stage_a=rebuild_stage_a,
                show_tfbs=show_tfbs,
                show_solutions=show_solutions,
            )
        except FileNotFoundError as exc:
            render_missing_input_hint(cfg_path, loaded, exc)
            raise typer.Exit(code=1)
        except RuntimeError as exc:
            if render_output_schema_hint(exc):
                raise typer.Exit(code=1)
            message = str(exc)
            if "Stage-A pools missing or stale" in message:
                console.print(f"[bold red]{message}[/]")
                console.print("[bold]Next steps[/]:")
                rebuild_cmd = context.workspace_command(
                    "dense stage-a build-pool --fresh",
                    cfg_path=cfg_path,
                    run_root=run_root,
                )
                console.print(f"  - {rebuild_cmd}")
                console.print("  - or rerun with --rebuild-stage-a to bootstrap pools")
                console.print(
                    "  - Stage-B libraries are built during dense run; no need to run dense stage-b build-libraries"
                )
                raise typer.Exit(code=1)
            raise

        console.print(":tada: [bold green]Run complete[/].")
        console.print("[bold]Next steps[/]:")
        console.print(
            f"  - {context.workspace_command('dense inspect run --library', cfg_path=cfg_path, run_root=run_root)}"
        )
        console.print(f"  - {context.workspace_command('dense report', cfg_path=cfg_path, run_root=run_root)}")

        # Auto-plot if configured
        if not no_plot and root.plots:
            try:
                ensure_mpl_cache_dir(run_root / "outputs" / ".mpl-cache")
            except Exception as exc:
                console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
                console.print("[bold]Tip[/]: set MPLCONFIGDIR=outputs/.mpl-cache inside the workspace.")
                raise typer.Exit(code=1)
            install_native_stderr_filters(suppress_solver_messages=False)
            from ..viz.plotting import run_plots_from_config

            console.print("[bold]Generating plots...[/]")
            run_plots_from_config(root, loaded.path, source="run")
            console.print(":bar_chart: [bold green]Plots written.[/]")

    @app.command("campaign-reset", hidden=True, help="Remove run outputs to reset a workspace.")
    def campaign_reset(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        run_root = context.run_root_for(loaded)
        outputs_root = run_outputs_root(run_root)
        if not outputs_root.exists():
            console.print(
                f"[bold yellow]No outputs found under[/] {context.display_path(outputs_root, run_root, absolute=False)}"
            )
            return
        if not outputs_root.is_dir():
            console.print(
                "[bold red]Outputs path is not a directory:[/] "
                f"{context.display_path(outputs_root, run_root, absolute=False)}"
            )
            raise typer.Exit(code=1)
        shutil.rmtree(outputs_root)
        console.print(
            ":broom: [bold green]Removed outputs under[/] "
            f"{context.display_path(outputs_root, run_root, absolute=False)}"
        )
