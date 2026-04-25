"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli/plots.py

Plotting CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..utils.logging_utils import install_native_stderr_filters
from ..utils.mpl_utils import ensure_mpl_cache_dir
from .context import CliContext


def _has_resumable_run_state(run_root: Path) -> bool:
    candidates = (
        run_root / "outputs" / "meta" / "run_state.json",
        run_root / "outputs" / "meta" / "run_manifest.json",
        run_root / "outputs" / "tables" / "attempts.parquet",
        run_root / "outputs" / "tables" / "records.parquet",
    )
    return any(path.exists() for path in candidates)


def _is_plot_selection_error(message: str) -> bool:
    lowered = message.lower()
    return "unknown plot name requested" in lowered or "no plot names selected" in lowered


def _is_truncation_error(message: str) -> bool:
    return "output records rows were truncated" in message.lower()


def _is_missing_local_artifact_error(message: str) -> bool:
    lowered = message.lower()
    tokens = (
        "run_manifest.json",
        "pool manifest not found",
        "pool_manifest.json",
        "attempts.parquet not found",
        "composition.parquet not found",
    )
    return any(token in lowered for token in tokens)


def register_plot_commands(app: typer.Typer, *, context: CliContext) -> None:
    console = context.console
    make_table = context.make_table

    @app.command("ls-plots", help="List available plot names and descriptions.")
    def ls_plots(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        if config is not None:
            cfg_path, is_default = context.resolve_config_path(ctx, config)
            context.load_config_or_exit(
                cfg_path,
                missing_message=context.default_config_missing_message if is_default else None,
            )
        from ..viz.plot_registry import PLOT_SPECS

        table = make_table("plot", "description")
        for name, meta in PLOT_SPECS.items():
            table.add_row(name, meta["description"])
        console.print(table)

    @app.command(help="Generate plots from outputs according to YAML. Use --only to select plots.")
    def plot(
        ctx: typer.Context,
        only: Optional[str] = typer.Option(None, help="Comma-separated plot names (subset of available plots)."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        allow_truncated: bool = typer.Option(
            False,
            "--allow-truncated",
            help="Allow sampled plotting when output records exceed the row materialization limit.",
        ),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        from ..viz.plot_inventory import base_plot_id, plot_missing_hint, plot_required_artifacts

        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        try:
            ensure_mpl_cache_dir()
        except Exception as exc:
            console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
            console.print(
                "[bold]Tip[/]: DenseGen defaults to a repo-local cache at .cache/matplotlib/densegen; "
                "set MPLCONFIGDIR to override."
            )
            raise typer.Exit(code=1)
        install_native_stderr_filters(suppress_solver_messages=False)
        from ..viz.plotting import run_plots_from_config

        run_root = Path(context.run_root_for(loaded))
        try:
            run_plots_from_config(
                loaded.root,
                loaded.path,
                only=only,
                source="plot",
                absolute=absolute,
                allow_truncated=allow_truncated,
            )
        except Exception as exc:
            message = str(exc)
            console.print(f"[bold red]Plot generation failed:[/] {message}")
            console.print("[bold]Next step[/]:")
            if _is_plot_selection_error(message):
                console.print(context.workspace_command("dense ls-plots", cfg_path=cfg_path, run_root=run_root))
                raise typer.Exit(code=1) from exc
            if _is_truncation_error(message):
                allow_truncated_cmd = "dense plot --allow-truncated"
                if only:
                    allow_truncated_cmd += f" --only {only}"
                console.print(context.workspace_command(allow_truncated_cmd, cfg_path=cfg_path, run_root=run_root))
                raise typer.Exit(code=1) from exc
            if _is_missing_local_artifact_error(message):
                console.print(
                    "[bold]Tip[/]: full DenseGen analysis needs workspace-local artifacts under "
                    "`outputs/meta`, `outputs/pools`, and `outputs/tables`, not just shared output records."
                )
                selected_plot_ids = [base_plot_id(token) for token in str(only or "").split(",") if base_plot_id(token)]
                if len(selected_plot_ids) == 1:
                    selected_plot_id = selected_plot_ids[0]
                    required_artifacts = plot_required_artifacts(selected_plot_id)
                    if required_artifacts:
                        console.print(
                            "[bold]Required artifacts[/]: " + ", ".join(f"`{item}`" for item in required_artifacts)
                        )
                    missing_hint = plot_missing_hint(selected_plot_id)
                    if missing_hint:
                        console.print(f"[bold]Plot contract[/]: {missing_hint}")
            if _has_resumable_run_state(run_root):
                rerun = "dense run --resume --no-plot"
            else:
                rerun = "dense run --fresh --no-plot"
            console.print(context.workspace_command(rerun, cfg_path=cfg_path, run_root=run_root))
            if (
                len(loaded.root.densegen.output.targets) > 1
                and ("records not found" in message.lower() or "output not found" in message.lower())
                and "plots.source" not in message
            ):
                console.print("[bold]Tip[/]: verify `plots.source` points to a sink with records (`parquet` or `usr`).")
            raise typer.Exit(code=1) from exc
        console.print(":bar_chart: [bold green]Plots written.[/]")
