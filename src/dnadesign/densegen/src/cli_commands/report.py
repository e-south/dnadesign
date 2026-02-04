"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/report.py

Report command for the DenseGen CLI.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..core.reporting import write_report
from .context import CliContext


def register_report_command(app: typer.Typer, *, context: CliContext) -> None:
    @app.command(help="Generate audit-grade report summary for a run.")
    def report(
        ctx: typer.Context,
        run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        out: str = typer.Option(
            "outputs/report",
            "--out",
            help="Output directory (relative to run root; must be inside outputs/).",
        ),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        plots: str = typer.Option(
            "none",
            "--plots",
            help="Include plot links in the report: none or include (requires outputs/plots/plot_manifest.json).",
        ),
        strict: bool = typer.Option(
            False,
            "--strict",
            "--fail-on-missing",
            help="Fail if core report inputs are missing.",
        ),
        format: str = typer.Option(
            "all",
            "--format",
            "-f",
            help="Report format: json, md, html, or all (comma-separated allowed).",
        ),
    ):
        if run is not None and config is not None:
            context.console.print("[bold red]Choose either --run or --config, not both.[/]")
            raise typer.Exit(code=1)
        if run is not None:
            cfg_path = Path(run) / "config.yaml"
            if not cfg_path.exists():
                context.console.print(
                    f"[bold red]Config not found under run:[/] "
                    f"{context.display_path(cfg_path, Path(run), absolute=absolute)}"
                )
                raise typer.Exit(code=1)
            loaded = context.load_config_or_exit(cfg_path, absolute=absolute, display_root=Path(run))
        else:
            cfg_path, is_default = context.resolve_config_path(ctx, config)
            loaded = context.load_config_or_exit(
                cfg_path,
                missing_message=context.default_config_missing_message if is_default else None,
                absolute=absolute,
                display_root=Path.cwd(),
            )
        raw_formats = {f.strip().lower() for f in format.split(",") if f.strip()}
        if not raw_formats:
            raw_formats = {"all"}
        allowed_formats = {"json", "md", "html", "all"}
        unknown = sorted(raw_formats - allowed_formats)
        if unknown:
            context.console.print(f"[bold red]Unknown report format(s):[/] {', '.join(unknown)}")
            context.console.print("Allowed: json, md, html, all.")
            raise typer.Exit(code=1)
        plots_mode = str(plots or "none").strip().lower()
        if plots_mode not in {"none", "include"}:
            context.console.print("[bold red]--plots must be one of: none, include.[/]")
            raise typer.Exit(code=1)
        include_plots = plots_mode == "include"
        formats_used = {"json", "md", "html"} if "all" in raw_formats else raw_formats
        run_root = context.run_root_for(loaded)
        out_dir = context.resolve_outputs_path_or_exit(cfg_path, run_root, out, label="report.out")
        try:
            with context.suppress_pyarrow_sysctl_warnings():
                write_report(
                    loaded.root,
                    cfg_path,
                    out_dir=out_dir,
                    include_plots=include_plots,
                    strict=strict,
                    formats=raw_formats,
                )
        except (FileNotFoundError, ValueError) as exc:
            context.console.print(f"[bold red]Report failed:[/] {exc}")
            entries = context.list_dir_entries(run_root, limit=8)
            if entries:
                context.console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
            context.console.print("[bold]Next steps[/]:")
            if "plot_manifest" in str(exc):
                context.console.print(context.workspace_command("dense plot", cfg_path=cfg_path, run_root=run_root))
            else:
                context.console.print(context.workspace_command("dense run", cfg_path=cfg_path, run_root=run_root))
            raise typer.Exit(code=1)
        context.console.print(
            f":sparkles: [bold green]Report written[/]: {context.display_path(out_dir, run_root, absolute=absolute)}"
        )
        outputs = []
        if "json" in formats_used:
            outputs.append("report.json")
        if "md" in formats_used:
            outputs.append("report.md")
        if "html" in formats_used:
            outputs.append("report.html")
        context.console.print(f"[bold]Outputs[/]: {', '.join(outputs) if outputs else '-'}")
