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
import os
import shlex
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Literal, Optional

import typer

from .context import CliContext

DEFAULT_NOTEBOOK_FILENAME = "densegen_run_overview.py"


def _ensure_marimo_installed() -> None:
    if importlib.util.find_spec("marimo") is not None:
        return
    raise RuntimeError("marimo is not installed. Install with `uv sync --locked`.")


def _default_notebook_path(run_root: Path) -> Path:
    return run_root / "outputs" / "notebooks" / DEFAULT_NOTEBOOK_FILENAME


def _resolve_notebook_records_path(*, loaded, run_root: Path, context: CliContext) -> Path:
    output_cfg = loaded.root.densegen.output
    targets = list(output_cfg.targets)
    if not targets:
        raise ValueError("output.targets must contain at least one sink")
    if len(targets) > 1:
        plots_cfg = loaded.root.plots
        if plots_cfg is None or plots_cfg.source is None:
            raise ValueError("plots.source must be set when output.targets has multiple sinks")
        source = str(plots_cfg.source)
        if source not in targets:
            raise ValueError("plots.source must be one of output.targets")
    else:
        source = str(targets[0])

    if source == "parquet":
        parquet_cfg = output_cfg.parquet
        if parquet_cfg is None:
            raise ValueError("output.parquet is required when notebook source resolves to parquet")
        return Path(
            context.resolve_outputs_path_or_exit(
                loaded.path,
                run_root,
                parquet_cfg.path,
                label="output.parquet.path",
            )
        )

    if source == "usr":
        usr_cfg = output_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when notebook source resolves to usr")
        dataset = str(usr_cfg.dataset).strip()
        if not dataset:
            raise ValueError("output.usr.dataset must be a non-empty string")
        usr_root = Path(
            context.resolve_outputs_path_or_exit(
                loaded.path,
                run_root,
                usr_cfg.root,
                label="output.usr.root",
            )
        )
        return usr_root / dataset / "records.parquet"

    raise ValueError(f"Unsupported notebook source: {source!r}")


def _render_notebook_template(*, run_root: Path, cfg_path: Path, records_path: Path) -> str:
    run_root_text = json.dumps(str(run_root.resolve()))
    cfg_path_text = json.dumps(str(cfg_path.resolve()))
    records_path_text = json.dumps(str(records_path.resolve()))
    generated_with = "unknown"
    try:
        generated_with = importlib_metadata.version("marimo")
    except importlib_metadata.PackageNotFoundError:
        pass
    generated_with_text = json.dumps(generated_with)
    template = """import marimo

__generated_with = __GENERATED_WITH__

app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from pyarrow.parquet import ParquetFile

    from dnadesign.baserender import load_records_from_parquet
    from dnadesign.baserender import render_record_figure
    from dnadesign.densegen.notebook_render_contract import densegen_notebook_render_contract

    return (
        Path,
        ParquetFile,
        json,
        load_records_from_parquet,
        mo,
        pd,
        render_record_figure,
        densegen_notebook_render_contract,
    )


@app.cell
def _(Path, densegen_notebook_render_contract):
    run_root = Path(__RUN_ROOT__)
    config_path = Path(__CFG_PATH__)
    records_path = Path(__RECORDS_PATH__)
    contract = densegen_notebook_render_contract()
    record_window_limit = int(contract.record_window_limit)
    run_manifest_path = run_root / "outputs" / "meta" / "run_manifest.json"
    plot_manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    return (
        config_path,
        contract,
        plot_manifest_path,
        record_window_limit,
        records_path,
        run_manifest_path,
        run_root,
    )


@app.cell
def _(mo):
    mo.md(\"\"\"
    # DenseGen Run Notebook

    Workspace-scoped run dashboard with summary metrics, record previews, and artifact gallery.
    Use **Refresh** to re-read artifacts after new DenseGen output is written.
    \"\"\")
    return


@app.cell
def _(config_path, mo, run_root):
    mo.md(f\"\"\"\n**Workspace root:** `{run_root}`  \n**Config:** `{config_path}`  \n\"\"\")
    return


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="Refresh", kind="neutral")
    refresh
    return refresh


@app.cell
def _(json, pd, refresh, run_manifest_path):
    _ = refresh.value
    run_manifest = {}
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text())
    run_items = pd.DataFrame(run_manifest.get("items", [])) if isinstance(run_manifest, dict) else pd.DataFrame()
    return run_items, run_manifest


@app.cell
def _(mo, run_items, run_manifest):
    summary_lines = []
    if isinstance(run_manifest, dict) and run_manifest:
        summary_lines.append(f"- Run id: `{run_manifest.get('run_id', '-')}`")
        summary_lines.append(f"- Schema version: `{run_manifest.get('schema_version', '-')}`")
        summary_lines.append(
            f"- Solver: `{run_manifest.get('solver_backend', '-')} / {run_manifest.get('solver_strategy', '-')}`"
        )
    if not run_items.empty:
        summary_lines.append(f"- Plan rows in manifest: `{len(run_items)}`")
        if "generated" in run_items.columns:
            summary_lines.append(f"- Total generated: `{int(run_items['generated'].fillna(0).sum())}`")
    mo.md("## Run summary\\n" + "\\n".join(summary_lines))
    return


@app.cell
def _(mo, records_path):
    mo.stop(
        not records_path.exists(),
        mo.md(
            f"No `{records_path.name}` artifact was found for this workspace (`{records_path}`). "
            "Run `uv run dense run` first."
        ),
    )
    return


@app.cell
def _(ParquetFile, contract, mo, pd, record_window_limit, records_path, refresh):
    _ = refresh.value
    record_id_column = str(contract.adapter_columns["id"])
    sequence_column = str(contract.adapter_columns["sequence"])
    annotations_column = str(contract.adapter_columns["annotations"])
    parquet_file = ParquetFile(records_path)
    schema_names = set(parquet_file.schema_arrow.names)
    required = {
        contract.adapter_columns["id"],
        contract.adapter_columns["sequence"],
        contract.adapter_columns["annotations"],
    }
    missing = sorted(required - schema_names)
    mo.stop(bool(missing), mo.md(f"`{records_path.name}` missing required columns: {missing}"))
    row_count = int(parquet_file.metadata.num_rows or 0)
    mo.stop(row_count <= 0, mo.md(f"`{records_path.name}` is empty."))
    window_n = min(row_count, max(1, int(record_window_limit)))
    preview_columns = [record_id_column, sequence_column, annotations_column]
    if "densegen__plan" in schema_names:
        preview_columns.append("densegen__plan")
    remaining = int(window_n)
    batches = []
    for batch in parquet_file.iter_batches(
        columns=preview_columns,
        batch_size=min(1024, int(window_n)),
    ):
        frame = batch.to_pandas()
        if remaining < len(frame):
            frame = frame.iloc[:remaining]
        batches.append(frame)
        remaining -= len(frame)
        if remaining <= 0:
            break
    df_window = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=preview_columns)
    mo.stop(df_window.empty, mo.md("No rows available in preview window."))
    return df_window, record_id_column, row_count, schema_names, window_n


@app.cell
def _(df_window, mo, record_id_column, records_path, row_count, schema_names, window_n):
    cols = set(schema_names)
    duplicate_id_count = int(df_window[record_id_column].astype(str).duplicated().sum())
    mo.stop(
        duplicate_id_count > 0,
        mo.md(
            "Duplicate ids detected in records preview window: "
            f"`{duplicate_id_count}`. Ensure `{records_path.name}` has unique `{record_id_column}` values."
        ),
    )
    lines = [
        f"- Records path: `{records_path}`",
        f"- Rows: `{row_count:,}`",
        f"- Columns: `{len(cols)}`",
        f"- Unique ids in preview: `{df_window[record_id_column].nunique() if record_id_column in cols else 'n/a'}`",
        f"- Unique plans in preview: `{df_window['densegen__plan'].nunique() if 'densegen__plan' in cols else 'n/a'}`",
        f"- Preview window: `{window_n}` rows",
    ]
    mo.md("## Records summary\\n" + "\\n".join(lines))
    return


@app.cell
def _(df_window, mo):
    mo.ui.dataframe(df_window)
    return


@app.cell
def _(mo, window_n):
    current_record_index, set_current_record_index = mo.state(0)

    def _prev(_):
        if window_n <= 0:
            return
        set_current_record_index(max(0, current_record_index() - 1))

    def _next(_):
        if window_n <= 0:
            return
        set_current_record_index(min(window_n - 1, current_record_index() + 1))

    prev_record_button = mo.ui.button(label="Previous record", on_click=_prev, disabled=(window_n <= 1))
    next_record_button = mo.ui.button(label="Next record", on_click=_next, disabled=(window_n <= 1))
    return current_record_index, next_record_button, prev_record_button


@app.cell
def _(current_record_index, df_window, mo, next_record_button, prev_record_button, record_id_column, window_n):
    mo.stop(window_n <= 0, mo.md("No records available for preview."))
    active_row_index = min(max(int(current_record_index()), 0), window_n - 1)
    active_row = df_window.iloc[active_row_index]
    active_record_id = str(active_row[record_id_column])
    mo.vstack(
        [
            mo.hstack([prev_record_button, next_record_button], justify="start"),
            mo.md(
                f"### BaseRender record preview\\n"
                f"Record `{active_row_index + 1}` / `{window_n}`  \\n"
                f"`id`: `{active_record_id}`"
            ),
        ]
    )
    return active_record_id


@app.cell
def _(contract, df_window, load_records_from_parquet, record_id_column, records_path, refresh):
    _ = refresh.value
    record_ids = [str(record_id) for record_id in df_window[record_id_column].tolist()]
    records = load_records_from_parquet(
        dataset_path=records_path,
        record_ids=record_ids,
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
    )
    records_by_id = {record.id: record for record in records}
    return records_by_id


@app.cell
def _(active_record_id, mo, records_by_id):
    mo.stop(active_record_id not in records_by_id, mo.md(f"Record `{active_record_id}` missing from preview cache."))
    active_record = records_by_id[active_record_id]
    return active_record


@app.cell
def _(active_record, contract, render_record_figure):
    baserender_figure = render_record_figure(active_record, style_preset=contract.style_preset)
    return baserender_figure


@app.cell
def _(baserender_figure):
    baserender_figure
    return


@app.cell
def _(json, plot_manifest_path):
    plot_paths = []
    if plot_manifest_path.exists():
        payload = json.loads(plot_manifest_path.read_text())
        for entry in payload.get("plots", []):
            rel_path = str(entry.get("path") or "").strip()
            if rel_path:
                candidate = (plot_manifest_path.parent / rel_path).resolve()
                if candidate.exists():
                    plot_paths.append(candidate)
    return plot_paths


@app.cell
def _(mo, plot_paths):
    current_plot_index, set_current_plot_index = mo.state(0)

    def _prev(_):
        if not plot_paths:
            return
        set_current_plot_index(max(0, current_plot_index() - 1))

    def _next(_):
        if not plot_paths:
            return
        set_current_plot_index(min(len(plot_paths) - 1, current_plot_index() + 1))

    prev_plot_button = mo.ui.button(label="Previous plot", on_click=_prev, disabled=(len(plot_paths) <= 1))
    next_plot_button = mo.ui.button(label="Next plot", on_click=_next, disabled=(len(plot_paths) <= 1))
    return current_plot_index, next_plot_button, prev_plot_button


@app.cell
def _(current_plot_index, mo, next_plot_button, plot_paths, prev_plot_button):
    mo.stop(
        not plot_paths,
        mo.md("## Plot gallery\\nNo `outputs/plots/plot_manifest.json` plots found yet. Run `uv run dense plot`."),
    )
    idx = min(max(int(current_plot_index()), 0), len(plot_paths) - 1)
    active_plot = plot_paths[idx]
    mo.vstack(
        [
            mo.md(f"## Plot gallery\\nPlot {idx + 1} / {len(plot_paths)}: `{active_plot.name}`"),
            mo.hstack([prev_plot_button, next_plot_button], justify="start"),
            mo.image(active_plot),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
"""
    return (
        template.replace("__GENERATED_WITH__", generated_with_text)
        .replace("__RUN_ROOT__", run_root_text)
        .replace("__CFG_PATH__", cfg_path_text)
        .replace("__RECORDS_PATH__", records_path_text)
    )


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
        try:
            records_path = _resolve_notebook_records_path(loaded=loaded, run_root=run_root, context=context)
        except Exception as exc:
            context.console.print(f"[bold red]Failed to resolve notebook records source:[/] {exc}")
            raise typer.Exit(code=1) from exc
        notebook_path = Path(out).expanduser().resolve() if out is not None else _default_notebook_path(run_root)
        if notebook_path.exists() and not force:
            context.console.print(f"[bold red]Notebook already exists:[/] {notebook_path}")
            context.console.print("[bold]Next step[/]: rerun with --force to overwrite.")
            raise typer.Exit(code=1)
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(
            _render_notebook_template(run_root=run_root, cfg_path=loaded.path, records_path=records_path)
        )
        notebook_label = context.display_path(notebook_path, run_root, absolute=absolute)
        context.console.print(f":sparkles: [bold green]Notebook generated[/]: {notebook_label}")
        context.console.print("[bold]Next steps[/]:")
        default_notebook = _default_notebook_path(run_root).resolve()
        if notebook_path.resolve() == default_notebook:
            run_command = "dense notebook run"
        else:
            notebook_run_path = context.display_path(notebook_path, run_root, absolute=absolute)
            run_command = f"dense notebook run --path {shlex.quote(notebook_run_path)}"
        context.console.print(context.workspace_command(run_command, cfg_path=cfg_path, run_root=run_root))

    @app.command("run", help="Launch a DenseGen marimo notebook.")
    def notebook_run(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        path: Optional[Path] = typer.Option(
            None,
            "--path",
            help="Notebook path (default: <run_root>/outputs/notebooks/densegen_run_overview.py).",
        ),
        mode: Literal["run", "edit"] = typer.Option(
            "run",
            "--mode",
            help="Launch mode: run (read-only app) or edit (interactive editor).",
        ),
        headless: bool = typer.Option(
            False,
            "--headless",
            help="Run without opening a browser window (marimo run mode only).",
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
        if headless and mode != "run":
            context.console.print("[bold red]--headless is only supported with --mode run.[/]")
            raise typer.Exit(code=1)
        context.console.print(
            f"[bold]Launching marimo ({mode})[/]: {context.display_path(notebook_path, run_root, absolute=absolute)}"
        )
        command = ["marimo", mode, str(notebook_path)]
        if headless:
            command.append("--headless")
        env = dict(os.environ)
        env.setdefault("MARIMO_SKIP_UPDATE_CHECK", "1")
        try:
            subprocess.run(command, check=True, env=env)
        except FileNotFoundError:
            context.console.print("[bold red]marimo CLI not found on PATH.[/]")
            context.console.print(f"Try: uv run marimo {mode} " + str(notebook_path))
            raise typer.Exit(code=1)
        except subprocess.CalledProcessError as exc:
            context.console.print(f"[bold red]marimo exited with code {exc.returncode}[/]")
            if mode == "edit":
                notebook_run_path = context.display_path(notebook_path, run_root, absolute=absolute)
                rerun_command = f"dense notebook run --mode run --path {shlex.quote(notebook_run_path)}"
                context.console.print("[bold]Next step[/]:")
                context.console.print(context.workspace_command(rerun_command, cfg_path=cfg_path, run_root=run_root))
            raise typer.Exit(code=1)
