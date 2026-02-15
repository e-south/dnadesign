"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/notebook.py

Workspace-scoped notebook commands for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional

import typer

from .context import CliContext

DEFAULT_NOTEBOOK_FILENAME = "densegen_run_overview.py"


def _ensure_marimo_installed() -> None:
    if importlib.util.find_spec("marimo") is not None:
        return
    raise RuntimeError("marimo is not installed. Install with `uv sync --locked`.")


def _default_notebook_path(run_root: Path) -> Path:
    return run_root / "outputs" / "notebooks" / DEFAULT_NOTEBOOK_FILENAME


def _render_notebook_template(*, run_root: Path, cfg_path: Path) -> str:
    run_root_text = json.dumps(str(run_root.resolve()))
    cfg_path_text = json.dumps(str(cfg_path.resolve()))
    generated_with = "unknown"
    try:
        generated_with = importlib_metadata.version("marimo")
    except importlib_metadata.PackageNotFoundError:
        pass
    generated_with_text = json.dumps(generated_with)
    return f"""import marimo

__generated_with = {generated_with_text}

app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, json, mo, pd


@app.cell
def _(Path):
    run_root = Path({run_root_text})
    config_path = Path({cfg_path_text})
    local_records_candidates = [
        run_root / "outputs" / "tables" / "records.parquet",
        run_root / "outputs" / "tables" / "dense_arrays.parquet",
    ]
    run_manifest_path = run_root / "outputs" / "meta" / "run_manifest.json"
    effective_config_path = run_root / "outputs" / "meta" / "effective_config.json"
    run_metrics_path = run_root / "outputs" / "tables" / "run_metrics.parquet"
    plot_manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    return (
        config_path,
        effective_config_path,
        local_records_candidates,
        plot_manifest_path,
        run_manifest_path,
        run_metrics_path,
        run_root,
    )


@app.cell
def _(mo):
    mo.md(\"\"\"
    # DenseGen Run Notebook

    Workspace-scoped run dashboard with summary metrics, records preview, and plot gallery.
    Use **Refresh** to re-read artifacts after new DenseGen output is written.
    \"\"\")
    return


@app.cell
def _(config_path, mo, run_root):
    mo.md(
        f\"\"\"\n**Workspace root:** `{{run_root}}`  \n**Config:** `{{config_path}}`  \n\"\"\"
    )
    return


@app.cell
def _(local_records_candidates, mo, run_root):
    sources = {{}}
    for _candidate in local_records_candidates:
        if _candidate.exists():
            _rel = _candidate.relative_to(run_root)
            sources[f"local: {{_rel}}"] = _candidate
    usr_root = run_root / "outputs" / "usr_datasets"
    if usr_root.exists():
        for _path in sorted(usr_root.glob("**/records.parquet")):
            _rel = _path.relative_to(run_root)
            sources[f"usr: {{_rel}}"] = _path
    if not sources:
        mo.stop(
            True,
            mo.md(
                "No `records.parquet` artifact was found for this workspace. "
                "Run `dense run` first."
            ),
        )
    return sources


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh", kind="neutral")
    sample_n = mo.ui.slider(50, 5000, value=500, step=50, label="Preview rows")
    return refresh, sample_n


@app.cell
def _(mo, sources):
    options = list(sources.keys())
    source_picker = mo.ui.dropdown(options=options, value=options[0], label="Records source")
    source_picker
    return source_picker


@app.cell
def _(mo, refresh, sample_n, source_picker):
    mo.hstack([source_picker, sample_n, refresh], justify="start")
    return


@app.cell
def _(json, pd, refresh, run_manifest_path):
    _ = refresh.value
    run_manifest = {{}}
    run_manifest_error = None
    if run_manifest_path.exists():
        try:
            run_manifest = json.loads(run_manifest_path.read_text())
        except Exception as exc:
            run_manifest_error = f"Failed to parse run_manifest.json: {{exc}}"
    run_items = pd.DataFrame(run_manifest.get("items", [])) if isinstance(run_manifest, dict) else pd.DataFrame()
    return run_items, run_manifest, run_manifest_error


@app.cell
def _(mo, refresh, run_items, run_manifest, run_manifest_error):
    _ = refresh.value
    _summary_lines = []
    if run_manifest_error:
        _summary_lines.append(f"- Warning: {{run_manifest_error}}")
    if isinstance(run_manifest, dict) and run_manifest:
        _summary_lines.append(f"- Run id: `{{run_manifest.get('run_id', '-')}}`")
        _summary_lines.append(f"- Schema version: `{{run_manifest.get('schema_version', '-')}}`")
        solver_backend = run_manifest.get("solver_backend", "-")
        solver_strategy = run_manifest.get("solver_strategy", "-")
        _summary_lines.append(f"- Solver: `{{solver_backend}} / {{solver_strategy}}`")
    if not run_items.empty:
        _summary_lines.append(f"- Plan rows in manifest: `{{len(run_items)}}`")
        if "generated" in run_items.columns:
            _summary_lines.append(f"- Total generated: `{{int(run_items['generated'].fillna(0).sum())}}`")
    mo.md("## Run summary\\n" + "\\n".join(_summary_lines))
    return


@app.cell
def _(pd, refresh, sample_n, source_picker, sources):
    _ = refresh.value
    records_path = sources[source_picker.value]
    df = pd.read_parquet(records_path)
    preview_n = int(sample_n.value) if int(sample_n.value) > 0 else 500
    df_preview = df.head(preview_n)
    return df, df_preview, records_path


@app.cell
def _(df, mo, records_path):
    cols = set(df.columns)
    _records_lines = [
        f"- Records path: `{{records_path}}`",
        f"- Rows: `{{len(df):,}}`",
        f"- Columns: `{{len(df.columns)}}`",
        f"- Unique ids: `{{df['id'].nunique() if 'id' in cols else 'n/a'}}`",
        f"- Unique plans: `{{df['densegen__plan'].nunique() if 'densegen__plan' in cols else 'n/a'}}`",
    ]
    mo.md("## Records summary\\n" + "\\n".join(_records_lines))
    return


@app.cell
def _(df_preview, mo):
    mo.ui.dataframe(df_preview)
    return


@app.cell
def _(json, plot_manifest_path, run_root):
    plot_paths = []
    if plot_manifest_path.exists():
        payload = json.loads(plot_manifest_path.read_text())
        for _entry in payload.get("plots", []):
            _rel = str(_entry.get("path") or "").strip()
            if _rel:
                _candidate = (plot_manifest_path.parent / _rel).resolve()
                if _candidate.exists():
                    plot_paths.append(_candidate)
    return plot_paths


@app.cell
def _(mo, plot_paths):
    current_index, set_current_index = mo.state(0)

    def _prev(_):
        if not plot_paths:
            return
        set_current_index(max(0, current_index() - 1))

    def _next(_):
        if not plot_paths:
            return
        set_current_index(min(len(plot_paths) - 1, current_index() + 1))

    prev_button = mo.ui.button(label="Previous", on_click=_prev, disabled=(len(plot_paths) == 0))
    next_button = mo.ui.button(label="Next", on_click=_next, disabled=(len(plot_paths) == 0))
    return current_index, next_button, prev_button


@app.cell
def _(current_index, mo, next_button, plot_paths, prev_button):
    if not plot_paths:
        mo.md("## Plot gallery\\nNo `outputs/plots/plot_manifest.json` plots found yet. Run `dense plot`.")
    mo.stop(not plot_paths)
    _idx = int(current_index())
    _active = plot_paths[_idx]
    mo.vstack(
        [
            mo.md(f"## Plot gallery\\nPlot {{_idx + 1}} / {{len(plot_paths)}}: `{{_active.name}}`"),
            mo.hstack([prev_button, next_button], justify="start"),
            mo.image(_active),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
"""


def register_notebook_commands(app: typer.Typer, *, context: CliContext) -> None:
    @app.command("generate", help="Generate a workspace-scoped marimo notebook for the current run.")
    def notebook_generate(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        out: Optional[Path] = typer.Option(
            None,
            "--out",
            help="Notebook output path (default: <run_root>/outputs/notebooks/densegen_run_overview.py).",
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite notebook if it already exists."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    ) -> None:
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
            absolute=absolute,
            display_root=Path.cwd(),
        )
        run_root = Path(context.run_root_for(loaded))
        notebook_path = Path(out).expanduser().resolve() if out is not None else _default_notebook_path(run_root)
        if notebook_path.exists() and not force:
            context.console.print(f"[bold red]Notebook already exists:[/] {notebook_path}")
            context.console.print("[bold]Next step[/]: rerun with --force to overwrite.")
            raise typer.Exit(code=1)
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(_render_notebook_template(run_root=run_root, cfg_path=loaded.path))
        notebook_label = context.display_path(notebook_path, run_root, absolute=absolute)
        context.console.print(f":sparkles: [bold green]Notebook generated[/]: {notebook_label}")
        context.console.print("[bold]Next steps[/]:")
        context.console.print(context.workspace_command("dense notebook run", cfg_path=cfg_path, run_root=run_root))

    @app.command("run", help="Launch a DenseGen marimo notebook.")
    def notebook_run(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        path: Optional[Path] = typer.Option(
            None,
            "--path",
            help="Notebook path (default: <run_root>/outputs/notebooks/densegen_run_overview.py).",
        ),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    ) -> None:
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
            absolute=absolute,
            display_root=Path.cwd(),
        )
        run_root = Path(context.run_root_for(loaded))
        notebook_path = Path(path).expanduser().resolve() if path is not None else _default_notebook_path(run_root)
        if not notebook_path.exists():
            context.console.print(
                f"[bold red]No notebook found:[/] {context.display_path(notebook_path, run_root, absolute=absolute)}"
            )
            context.console.print("[bold]Next step[/]:")
            context.console.print(
                context.workspace_command("dense notebook generate", cfg_path=cfg_path, run_root=run_root)
            )
            raise typer.Exit(code=1)
        try:
            _ensure_marimo_installed()
        except RuntimeError as exc:
            context.console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1)
        context.console.print(
            f"[bold]Launching marimo[/]: {context.display_path(notebook_path, run_root, absolute=absolute)}"
        )
        try:
            subprocess.run(["marimo", "edit", str(notebook_path)], check=True)
        except FileNotFoundError:
            context.console.print("[bold red]marimo CLI not found on PATH.[/]")
            context.console.print("Try: uv run marimo edit " + str(notebook_path))
            raise typer.Exit(code=1)
        except subprocess.CalledProcessError as exc:
            context.console.print(f"[bold red]marimo exited with code {exc.returncode}[/]")
            raise typer.Exit(code=1)
