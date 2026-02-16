"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/config.py

Config validation CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import typer

from ..cli_commands.context import CliContext


def register_validate_command(
    app: typer.Typer,
    *,
    context: CliContext,
    warn_pwm_sampling_configs: Callable[..., None],
    ensure_fimo_available: Callable[..., None],
) -> None:
    console = context.console

    @app.command("validate-config", help="Validate the config YAML (schema + sanity).")
    def validate_config(
        ctx: typer.Context,
        probe_solver: bool = typer.Option(False, help="Also probe the solver backend."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        warn_pwm_sampling_configs(loaded, cfg_path)
        context.warn_full_pool_strategy(loaded)
        explicit_cfg = bool(
            config or (ctx.obj and ctx.obj.get("config_path") is not None) or os.environ.get("DENSEGEN_CONFIG_PATH")
        )
        ensure_fimo_available(loaded.root.densegen, strict=explicit_cfg)
        if probe_solver:
            from ..adapters.optimizer import DenseArraysAdapter
            from ..core.pipeline import select_solver

            solver_cfg = loaded.root.densegen.solver
            try:
                select_solver(
                    solver_cfg.backend,
                    DenseArraysAdapter(),
                    strategy=str(solver_cfg.strategy),
                )
            except Exception as exc:
                console.print(f"[bold red]Solver probe failed:[/] {exc}")
                console.print("[bold]Next steps[/]:")
                console.print("  - install/configure the requested solver backend")
                console.print("  - or set densegen.solver.backend to an available backend (for example CBC)")
                raise typer.Exit(code=1)
        console.print(":white_check_mark: [bold green]Config is valid.[/]")
