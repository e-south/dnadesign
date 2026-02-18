"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_template.py

Marimo notebook template sections and renderer for DenseGen CLI notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path


@dataclass(frozen=True)
class NotebookTemplateContext:
    run_root: Path
    cfg_path: Path
    records_path: Path
    output_source: str
    usr_root: Path | None
    usr_dataset: str | None


def _template_header() -> str:
    return """import marimo

__generated_with = __GENERATED_WITH__

app = marimo.App(width=\"medium\")

"""


def _template_cells() -> str:
    return """
@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import yaml
    from pyarrow.parquet import ParquetFile

    from dnadesign.baserender import load_records_from_parquet
    from dnadesign.baserender import render_record_figure
    from dnadesign.densegen.src.integrations.baserender.notebook_contract import densegen_notebook_render_contract
    from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS

    def require(condition: bool, message: str) -> None:
        if bool(condition):
            raise RuntimeError(message)

    return (
        ParquetFile,
        Path,
        densegen_notebook_render_contract,
        json,
        PLOT_SPECS,
        load_records_from_parquet,
        mo,
        pd,
        require,
        render_record_figure,
        yaml,
    )


@app.cell
def _(Path, densegen_notebook_render_contract):
    run_root = Path(__RUN_ROOT__)
    workspace_name = run_root.name
    config_path = Path(__CFG_PATH__)
    records_path = Path(__RECORDS_PATH__)
    output_source = __OUTPUT_SOURCE__
    usr_root_text = __USR_ROOT__
    usr_root = Path(usr_root_text) if usr_root_text else None
    usr_dataset = __USR_DATASET__
    contract = densegen_notebook_render_contract()
    record_window_limit = int(contract.record_window_limit)
    run_manifest_path = run_root / "outputs" / "meta" / "run_manifest.json"
    plot_manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    return (
        config_path,
        contract,
        output_source,
        plot_manifest_path,
        record_window_limit,
        records_path,
        run_manifest_path,
        run_root,
        usr_dataset,
        usr_root,
        workspace_name,
    )


@app.cell
def _(mo, workspace_name):
    mo.md(
        f\"\"\"
# {workspace_name}

This notebook is a workspace-scoped run dashboard for DenseGen outputs.
\"\"\"
    )
    return


@app.cell
def _(config_path, json, pd, require, run_manifest_path, workspace_name, yaml):
    config_load_error = None
    run_manifest_load_error = None
    run_manifest = {}
    config_payload = {}

    if run_manifest_path.exists():
        try:
            run_manifest = json.loads(run_manifest_path.read_text())
        except Exception as exc:
            run_manifest_load_error = f"Failed to parse `run_manifest.json`: {exc}"
    require(run_manifest_load_error is not None, run_manifest_load_error or "Run manifest is invalid.")

    try:
        config_payload = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        config_load_error = str(exc)

    densegen_cfg = config_payload.get("densegen", {}) if isinstance(config_payload, dict) else {}
    run_items = pd.DataFrame(run_manifest.get("items", [])) if isinstance(run_manifest, dict) else pd.DataFrame()

    output_cfg = densegen_cfg.get("output", {}) if isinstance(densegen_cfg.get("output", {}), dict) else {}
    output_targets = output_cfg.get("targets", [])
    output_targets_text = ", ".join(str(item) for item in output_targets) if isinstance(output_targets, list) else "-"

    context_rows = [
        {"Field": "Workspace", "Value": workspace_name},
        {"Field": "Run root", "Value": str(config_path.parent)},
        {"Field": "Config path", "Value": str(config_path)},
        {"Field": "Schema version", "Value": str(densegen_cfg.get("schema_version", "-"))},
        {"Field": "Output targets", "Value": output_targets_text},
    ]
    if config_load_error:
        context_rows.append({"Field": "Config parse warning", "Value": config_load_error})
    context_table = pd.DataFrame(context_rows)
    return context_table, densegen_cfg, run_items, run_manifest


@app.cell
def _(records_path, require):
    require(
        not records_path.exists(),
        f"No `{records_path.name}` artifact was found for this workspace (`{records_path}`). "
        "Run `uv run dense run` first.",
    )
    return


@app.cell
def _(
    ParquetFile,
    contract,
    output_source,
    pd,
    record_window_limit,
    records_path,
    require,
    run_root,
    usr_dataset,
    usr_root,
):
    record_id_column = str(contract.adapter_columns["id"])
    preview_records_path = records_path
    parquet_open_error = None
    parquet_file = None
    try:
        parquet_file = ParquetFile(preview_records_path)
    except Exception as exc:
        parquet_open_error = f"Failed to open `{preview_records_path.name}`: {exc}"
    require(parquet_open_error is not None, parquet_open_error or "Unable to read records artifact.")
    schema_names = set(parquet_file.schema_arrow.names)
    required = {
        contract.adapter_columns["id"],
        contract.adapter_columns["sequence"],
        contract.adapter_columns["annotations"],
    }
    missing = sorted(required - schema_names)
    if missing and str(output_source or "").strip() == "usr":
        require(
            usr_root is None or not str(usr_dataset or "").strip(),
            "Notebook source is USR but generation context does not include a dataset path.",
        )
        usr_export_path = run_root / "outputs" / "notebooks" / "records_with_overlays.parquet"
        try:
            from dnadesign.usr import Dataset

            ds = Dataset(usr_root, str(usr_dataset))
            require(not ds.records_path.exists(), f"USR records not found: {ds.records_path}")
            ds.export("parquet", usr_export_path, include_deleted=False)
            preview_records_path = usr_export_path
            parquet_file = ParquetFile(preview_records_path)
            schema_names = set(parquet_file.schema_arrow.names)
            missing = sorted(required - schema_names)
        except Exception as exc:
            raise RuntimeError(f"Failed to build merged USR records for notebook preview: {exc}") from exc
    require(
        bool(missing),
        f"`{preview_records_path.name}` missing required columns: {missing}. "
        "DenseGen BaseRender preview requires id, sequence, and densegen placement detail.",
    )
    row_count = int(parquet_file.metadata.num_rows or 0)
    require(row_count <= 0, f"`{preview_records_path.name}` is empty.")

    window_n = min(row_count, max(1, int(record_window_limit)))
    preview_columns = list(parquet_file.schema_arrow.names)

    remaining = int(window_n)
    batches = []
    for batch in parquet_file.iter_batches(columns=preview_columns, batch_size=min(1024, int(window_n))):
        frame = batch.to_pandas()
        if remaining < len(frame):
            frame = frame.iloc[:remaining]
        batches.append(frame)
        remaining -= len(frame)
        if remaining <= 0:
            break

    df_window = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=preview_columns)
    require(df_window.empty, "No rows available in preview window.")
    return df_window, preview_records_path, record_id_column, row_count, schema_names, window_n


@app.cell
def _(pd, run_items, run_manifest):
    run_rows = []
    if isinstance(run_manifest, dict) and run_manifest:
        run_rows.extend(
            [
                {"Field": "Run id (manifest)", "Value": str(run_manifest.get("run_id", "-"))},
                {"Field": "Generated total", "Value": str(run_manifest.get("total_generated", "-"))},
                {"Field": "Solver backend", "Value": str(run_manifest.get("solver_backend", "-"))},
                {"Field": "Solver strategy", "Value": str(run_manifest.get("solver_strategy", "-"))},
            ]
        )
    if not run_items.empty:
        run_rows.append({"Field": "Plan rows in manifest", "Value": str(len(run_items))})
        if "generated" in run_items.columns:
            generated_total = int(run_items["generated"].fillna(0).sum())
            run_rows.append({"Field": "Generated across plans", "Value": str(generated_total)})

    if not run_rows:
        run_rows = [{"Field": "Run summary", "Value": "Run manifest not available yet."}]

    run_summary_table = pd.DataFrame(run_rows)
    return run_summary_table


@app.cell
def _(df_window, pd, preview_records_path, record_id_column, require, row_count, schema_names, window_n):
    cols = set(schema_names)
    duplicate_id_count = int(df_window[record_id_column].astype(str).duplicated().sum())
    require(
        duplicate_id_count > 0,
        "Duplicate record ids detected in the notebook preview window "
        f"({duplicate_id_count}). Resolve id collisions in `{preview_records_path.name}` and rerun.",
    )
    records_rows = [
        {"Field": "Records path", "Value": str(preview_records_path)},
        {"Field": "Rows", "Value": f"{row_count:,}"},
        {"Field": "Columns", "Value": str(len(cols))},
        {"Field": "Preview rows", "Value": str(window_n)},
        {"Field": "Duplicate ids in preview", "Value": str(duplicate_id_count)},
        {
            "Field": "Unique plans in preview",
            "Value": str(df_window["densegen__plan"].nunique()) if "densegen__plan" in cols else "n/a",
        },
    ]
    records_summary_table = pd.DataFrame(records_rows)
    return records_summary_table


@app.cell
def _(context_table, mo, records_summary_table, run_summary_table):
    mo.vstack(
        [
            mo.md("### Workspace context"),
            mo.ui.table(context_table),
            mo.hstack(
                [
                    mo.vstack([mo.md("### Run summary"), mo.ui.table(run_summary_table)]),
                    mo.vstack([mo.md("### Records summary"), mo.ui.table(records_summary_table)]),
                ],
                justify="start",
                align="start",
                wrap=True,
                widths=[1, 1],
            ),
        ],
        align="stretch",
    )
    return


@app.cell
def _(mo):
    get_active_record_index, set_active_record_index = mo.state(0)
    return get_active_record_index, set_active_record_index


@app.cell
def _(df_window, mo):
    has_plan_column = "densegen__plan" in set(df_window.columns)
    _plan_options = ["all"]
    if has_plan_column:
        _plan_values = sorted(df_window["densegen__plan"].dropna().astype(str).unique().tolist())
        _plan_options.extend(_plan_values)

    record_plan_filter = mo.ui.dropdown(options=_plan_options, value=_plan_options[0], label="Record plan")
    return has_plan_column, record_plan_filter


@app.cell
def _(
    df_window,
    get_active_record_index,
    has_plan_column,
    record_id_column,
    record_plan_filter,
    require,
    set_active_record_index,
):
    selected_record_plan = str(record_plan_filter.value or "all")
    if selected_record_plan == "all" or not has_plan_column:
        df_window_filtered = df_window.reset_index(drop=True)
    else:
        _mask = df_window["densegen__plan"].astype(str) == selected_record_plan
        df_window_filtered = df_window[_mask].reset_index(drop=True)
    require(
        df_window_filtered.empty,
        f"No records found for plan `{selected_record_plan}` in preview window.",
    )

    _record_options = []
    for _row_index, _record_id in enumerate(df_window_filtered[record_id_column].astype(str).tolist()):
        _option = f"{_row_index + 1}. {_record_id}"
        _record_options.append(_option)

    record_count = len(_record_options)
    require(record_count <= 0, "No records are available in the selected preview window.")

    _raw_active_index = int(get_active_record_index() or 0)
    active_index = max(0, min(record_count - 1, _raw_active_index))
    if active_index != _raw_active_index:
        set_active_record_index(active_index)

    prev_record_button = mo.ui.button(
        label="Prev",
        kind="neutral",
        on_click=lambda _: set_active_record_index(lambda index: (int(index) - 1) % record_count),
    )
    next_record_button = mo.ui.button(
        label="Next",
        kind="neutral",
        on_click=lambda _: set_active_record_index(lambda index: (int(index) + 1) % record_count),
    )
    return (
        df_window_filtered,
        next_record_button,
        prev_record_button,
        record_count,
        selected_record_plan,
        record_plan_filter,
    )


@app.cell
def _(df_window_filtered, mo, run_root):
    export_format = mo.ui.dropdown(options=["parquet", "csv"], value="parquet", label="Export format")
    default_export_path = run_root / "outputs" / "notebooks" / "records_preview.parquet"
    export_path = mo.ui.text(value=str(default_export_path), label="Export path", full_width=True)
    export_button = mo.ui.run_button(label="Export", kind="neutral")
    export_controls = mo.hstack(
        [
            mo.vstack([export_format, export_path], align="stretch"),
            mo.hstack([export_button], justify="end", align="end"),
        ],
        justify="space-between",
        align="end",
        widths=[8, 1],
    )
    mo.vstack(
        [
            mo.md("### Records preview"),
            mo.ui.table(df_window_filtered),
            export_controls,
        ]
    )
    return export_button, export_format, export_path


@app.cell
def _(
    df_window_filtered,
    get_active_record_index,
    mo,
    next_record_button,
    prev_record_button,
    record_count,
    record_id_column,
    selected_record_plan,
    record_plan_filter,
    set_active_record_index,
):
    _raw_active_index = int(get_active_record_index() or 0)
    active_row_index = max(0, min(record_count - 1, _raw_active_index))
    if active_row_index != _raw_active_index:
        set_active_record_index(active_row_index)

    active_row = df_window_filtered.iloc[active_row_index]
    active_record_id = str(active_row[record_id_column])
    filtered_n = len(df_window_filtered)
    _record_status = mo.md(
        "<div style='text-align:center'>"
        + f"`{active_row_index + 1} / {filtered_n}` | `id: {active_record_id}`"
        + "</div>"
    )
    nav_row = mo.hstack(
        [prev_record_button, next_record_button],
        justify="space-between",
        align="center",
        widths=[1, 1],
    )
    mo.vstack(
        [
            mo.md("### BaseRender preview"),
            mo.hstack([record_plan_filter], justify="start", align="center"),
            _record_status,
            nav_row,
        ]
    )
    return active_record_id, selected_record_plan


@app.cell
def _(contract, df_window_filtered, load_records_from_parquet, preview_records_path, record_id_column, require):
    record_ids = [str(record_id) for record_id in df_window_filtered[record_id_column].tolist()]
    records = load_records_from_parquet(
        dataset_path=preview_records_path,
        record_ids=record_ids,
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
    )
    records_by_id = {record.id: record for record in records}
    require(
        len(records_by_id) != len(records),
        "Preview cache contains duplicate record ids. Resolve id collisions and rerun.",
    )
    missing_ids = [record_id for record_id in record_ids if record_id not in records_by_id]
    require(
        bool(missing_ids),
        "Preview cache is missing records from the selected window: "
        + ", ".join(f"`{record_id}`" for record_id in missing_ids[:8])
        + (" ..." if len(missing_ids) > 8 else ""),
    )
    return records_by_id


@app.cell
def _(active_record_id, records_by_id, require):
    require(active_record_id not in records_by_id, f"Record `{active_record_id}` missing from preview cache.")
    active_record = records_by_id[active_record_id]
    return active_record


@app.cell
def _(active_record, contract, render_record_figure):
    baserender_figure = render_record_figure(active_record, style_preset=contract.style_preset)
    baserender_figure.patch.set_facecolor("white")
    for axis in baserender_figure.axes:
        axis.set_facecolor("white")
    return baserender_figure


@app.cell
def _(baserender_figure):
    baserender_figure
    return


@app.cell
def _(json, plot_manifest_path):
    plot_entries = []
    plot_manifest_load_error = None
    plot_root = plot_manifest_path.parent
    image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
    supported_suffixes = image_suffixes | {".pdf"}
    seen_paths = set()
    if plot_manifest_path.exists():
        try:
            _payload = json.loads(plot_manifest_path.read_text())
        except Exception as exc:
            plot_manifest_load_error = f"Failed to parse `plot_manifest.json`: {exc}"
        else:
            for _entry in _payload.get("plots", []):
                _rel_path = str(_entry.get("path") or "").strip()
                if not _rel_path:
                    continue
                _candidate = (plot_root / _rel_path).resolve()
                if not _candidate.exists():
                    continue
                _suffix = str(_candidate.suffix).lower()
                if _suffix not in supported_suffixes:
                    continue
                _key = str(_candidate)
                if _key in seen_paths:
                    continue
                seen_paths.add(_key)
                plot_entries.append(
                    {
                        "path": _candidate,
                        "plot_id": str(_entry.get("plot_id") or _entry.get("name") or _candidate.stem),
                        "plan_name": str(_entry.get("plan_name") or "unscoped"),
                        "plot_name": str(_entry.get("name") or _candidate.stem),
                        "variant": str(_entry.get("variant") or ""),
                        "description": str(_entry.get("description") or ""),
                    }
                )

    for _candidate in sorted(plot_root.rglob("*")):
        if not _candidate.is_file():
            continue
        _resolved = _candidate.resolve()
        _suffix = str(_resolved.suffix).lower()
        if _suffix not in supported_suffixes:
            continue
        _key = str(_resolved)
        if _key in seen_paths:
            continue
        seen_paths.add(_key)
        _relative_parts = _resolved.relative_to(plot_root.resolve()).parts
        _plan_name = "unscoped"
        if len(_relative_parts) >= 2 and _relative_parts[0] == "stage_b":
            _plan_name = str(_relative_parts[1])
        elif len(_relative_parts) >= 1 and _relative_parts[0] == "stage_a":
            _plan_name = "stage_a"
        plot_entries.append(
            {
                "path": _resolved,
                "plot_id": "",
                "plan_name": _plan_name,
                "plot_name": str(_resolved.stem),
                "variant": "",
                "description": "",
            }
        )

    def _suffix_priority(entry: dict[str, object]) -> tuple[int, str]:
        _suffix = str(getattr(entry["path"], "suffix", "")).lower()
        if _suffix in image_suffixes:
            return (0, _suffix)
        if _suffix == ".pdf":
            return (1, _suffix)
        return (2, _suffix)

    preferred_entries: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for _entry in plot_entries:
        _key = (
            str(_entry.get("plot_id") or ""),
            str(_entry.get("plan_name") or ""),
            str(_entry.get("plot_name") or ""),
            str(_entry.get("variant") or ""),
        )
        _current = preferred_entries.get(_key)
        if _current is None or _suffix_priority(_entry) < _suffix_priority(_current):
            preferred_entries[_key] = _entry

    plot_entries = sorted(
        preferred_entries.values(),
        key=lambda entry: (str(entry["plan_name"]), str(entry["plot_name"]), str(entry["path"])),
    )
    return plot_entries, plot_manifest_load_error


@app.cell
def _(PLOT_SPECS, mo, plot_entries, plot_manifest_load_error, require, selected_record_plan):
    require(plot_manifest_load_error is not None, plot_manifest_load_error or "Plot manifest is invalid.")
    require(
        not plot_entries,
        "No `outputs/plots/plot_manifest.json` plots found yet. Run `uv run dense plot`.",
    )
    available_plot_names = sorted(list(PLOT_SPECS.keys()))
    generated_plot_names = sorted(
        {
            str(entry.get("plot_id") or "").strip()
            for entry in plot_entries
            if str(entry.get("plot_id") or "").strip()
        }
    )
    missing_plot_names = [name for name in available_plot_names if name not in generated_plot_names]
    if missing_plot_names:
        _joined = ",".join(missing_plot_names)
        mo.md(
            "Available but not generated: "
            + ", ".join(f"`{name}`" for name in missing_plot_names)
            + f". Run `uv run dense plot --only {_joined}` to generate them."
        )

    selected_plot_plan = str(selected_record_plan or "all")
    _filtered_entries = list(plot_entries)
    if selected_plot_plan != "all":
        _allowed_plans = {selected_plot_plan, "unscoped", "stage_a"}
        _filtered_entries = [_entry for _entry in plot_entries if str(_entry["plan_name"]) in _allowed_plans]
    require(not _filtered_entries, f"No plots found for plan `{selected_plot_plan}`.")

    label_to_entry = {}
    plot_options = []
    for _entry_index, _entry in enumerate(_filtered_entries):
        _variant = str(_entry["variant"]).strip()
        _label = str(_entry["plot_name"])
        _plan_name = str(_entry["plan_name"]).strip()
        if _plan_name and _plan_name != "unscoped":
            _label = f"{_plan_name} | {_label}"
        if _variant:
            _label = f"{_label} ({_variant})"
        _option_label = f"{_entry_index + 1}. {_label}"
        plot_options.append(_option_label)
        label_to_entry[_option_label] = _entry

    plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="Plot")
    return label_to_entry, plot_selector


@app.cell
def _(label_to_entry, plot_selector, require):
    _selected_plot_option = str(plot_selector.value or "")
    require(
        _selected_plot_option not in label_to_entry,
        "Selected plot is not available for the current plan filter.",
    )
    active_plot_entry = label_to_entry[_selected_plot_option]
    return active_plot_entry


@app.cell
def _(mo):
    plot_height_px = mo.ui.slider(
        360,
        1400,
        step=20,
        value=820,
        show_value=True,
        label="Plot viewport height (px)",
    )
    plot_height_px
    return plot_height_px


@app.cell
def _(active_plot_entry, mo, plot_height_px, plot_selector):
    _plan_name = str(active_plot_entry["plan_name"])
    _plot_name = str(active_plot_entry["plot_name"])
    _variant = str(active_plot_entry["variant"]).strip()
    _plot_path = active_plot_entry["path"]
    _suffix = str(_plot_path.suffix).lower()
    _variant_text = _variant if _variant else "-"
    _stage_b_fingerprint = _plot_name if _plan_name not in {"stage_a", "unscoped"} else "-"
    _controls = mo.hstack(
        [
            plot_selector,
            mo.md(f"Plan: `{_plan_name}`"),
            mo.md(f"Variant: `{_variant_text}`"),
            mo.md(f"Stage-B fingerprint: `{_stage_b_fingerprint}`"),
        ],
        justify="start",
        align="center",
        wrap=True,
        widths=[2, 1, 1, 2],
    )

    _content = [_controls]
    if _suffix == ".pdf":
        _content.append(mo.pdf(_plot_path, width="100%", height=f"{int(plot_height_px.value)}px"))
    elif _suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}:
        _content.append(mo.image(_plot_path))
    else:
        _content.append(
            mo.md(
                f"Plot preview is not supported for `{_plot_path.name}`. "
                f"Open the file from disk: `{_plot_path}`"
            )
        )
    mo.vstack(_content)
    return


@app.cell
def _(Path, df_window_filtered, export_button, export_format, export_path, mo, require, run_root):
    click_count = int(export_button.value or 0)
    status_text = ""
    if click_count > 0:
        selected_format = str(export_format.value or "").strip()
        require(
            selected_format not in {"parquet", "csv"},
            f"Export format must be parquet or csv, got `{selected_format}`.",
        )
        raw_path = str(export_path.value or "").strip()
        if not raw_path:
            raw_path = "outputs/notebooks/records_preview"

        destination = Path(raw_path).expanduser()
        if not destination.is_absolute():
            destination = run_root / destination

        if selected_format == "csv":
            if destination.suffix.lower() != ".csv":
                destination = destination.with_suffix(".csv")
        else:
            if destination.suffix.lower() != ".parquet":
                destination = destination.with_suffix(".parquet")

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if selected_format == "csv":
                df_window_filtered.to_csv(destination, index=False)
            else:
                df_window_filtered.to_parquet(destination, index=False)
        except Exception as exc:
            raise RuntimeError(f"Export failed while writing `{destination}`: {exc}") from exc

        status_text = f"Saved to `{destination}`."
    mo.md(status_text)
    return

"""


def _template_footer() -> str:
    return """
if __name__ == "__main__":
    app.run()
"""


def _template_sections() -> tuple[str, ...]:
    return (
        _template_header(),
        _template_cells(),
        _template_footer(),
    )


def render_notebook_template(context: NotebookTemplateContext) -> str:
    run_root_text = json.dumps(str(context.run_root.resolve()))
    cfg_path_text = json.dumps(str(context.cfg_path.resolve()))
    records_path_text = json.dumps(str(context.records_path.resolve()))
    output_source_text = json.dumps(str(context.output_source))
    usr_root_text = json.dumps(str(context.usr_root.resolve()) if context.usr_root is not None else "")
    usr_dataset_text = json.dumps(str(context.usr_dataset or ""))
    generated_with = "unknown"
    try:
        generated_with = importlib_metadata.version("marimo")
    except importlib_metadata.PackageNotFoundError:
        pass
    generated_with_text = json.dumps(generated_with)

    template = "".join(_template_sections())
    return (
        template.replace("__GENERATED_WITH__", generated_with_text)
        .replace("__RUN_ROOT__", run_root_text)
        .replace("__CFG_PATH__", cfg_path_text)
        .replace("__RECORDS_PATH__", records_path_text)
        .replace("__OUTPUT_SOURCE__", output_source_text)
        .replace("__USR_ROOT__", usr_root_text)
        .replace("__USR_DATASET__", usr_dataset_text)
    )
