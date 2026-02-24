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

from .notebook_template_cells import records_export_cell_template as _records_export_cell_template


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

app = marimo.App(width=\"compact\")

"""


def _template_cells() -> str:
    return """
@app.cell
def _():
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import subprocess

    import marimo as mo
    import pandas as pd
    import yaml
    from pyarrow.parquet import ParquetFile

    from dnadesign.baserender import load_records_from_parquet
    from dnadesign.baserender import render_record_figure
    from dnadesign.densegen import PLOT_SPECS, densegen_notebook_render_contract
    from dnadesign.densegen import build_run_summary_tables

    def require(condition: bool, message: str) -> None:
        if bool(condition):
            raise RuntimeError(message)

    return (
        ParquetFile,
        Path,
        hashlib,
        build_run_summary_tables,
        densegen_notebook_render_contract,
        json,
        PLOT_SPECS,
        load_records_from_parquet,
        mo,
        pd,
        require,
        render_record_figure,
        shutil,
        subprocess,
        yaml,
    )


@app.cell
def _(Path, densegen_notebook_render_contract):
    run_root = Path(__RUN_ROOT__)
    config_path = Path(__CFG_PATH__)
    workspace_name = str(config_path.parent.name or run_root.name)
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
def _(config_path, json, pd, records_path, require, run_manifest_path, run_root, workspace_name, yaml):
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
    try:
        _records_path_display = str(records_path.resolve().relative_to(run_root.resolve()))
    except Exception:
        _records_path_display = str(records_path)

    context_rows = [
        {"Field": "Workspace", "Value": workspace_name},
        {"Field": "Run root", "Value": str(config_path.parent)},
        {"Field": "Config path", "Value": str(config_path)},
        {"Field": "Records path", "Value": _records_path_display},
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
    duplicate_id_count = int(df_window[record_id_column].astype(str).duplicated().sum())
    require(
        duplicate_id_count > 0,
        "Duplicate record ids detected in the notebook preview window "
        f"({duplicate_id_count}). Resolve id collisions in `{preview_records_path.name}` and rerun.",
    )
    return df_window, preview_records_path, record_id_column


@app.cell
def _(build_run_summary_tables, pd, run_items, run_manifest):
    plan_quota_table, run_summary_table = build_run_summary_tables(
        run_manifest=run_manifest,
        run_items=run_items,
        pd=pd,
    )
    return plan_quota_table, run_summary_table


@app.cell
def _(context_table, mo, plan_quota_table, run_summary_table):
    _run_summary_blocks = [
        mo.md("### Workspace context"),
        mo.ui.table(context_table),
        mo.md("### Run summary"),
        mo.ui.table(run_summary_table),
    ]
    if not plan_quota_table.empty:
        _run_summary_blocks.extend(
            [
                mo.md("#### Plan quota breakdown"),
                mo.ui.table(plan_quota_table),
            ]
        )
    mo.vstack(_run_summary_blocks, align="stretch")
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
        record_plan_filter,
    )


@app.cell
def _(df_window_filtered, mo, run_root):
    export_format = mo.ui.dropdown(options=["parquet", "csv"], value="parquet", label="Dataset export format")
    default_export_path = run_root / "outputs" / "notebooks" / "records_preview.parquet"
    export_path = mo.ui.text(value=str(default_export_path), label="Dataset export path", full_width=True)
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
            mo.md(
                "Export writes the currently filtered records table from this notebook "
                "to the selected format and path."
            ),
            mo.ui.table(df_window_filtered.loc[:, list(df_window_filtered.columns)], page_size=10),
            export_controls,
        ]
    )
    return export_button, export_format, export_path


@app.cell
def _(
    df_window_filtered,
    get_active_record_index,
    json,
    mo,
    record_count,
    record_id_column,
    record_plan_filter,
    set_active_record_index,
):
    _raw_active_index = int(get_active_record_index() or 0)
    active_row_index = max(0, min(record_count - 1, _raw_active_index))
    if active_row_index != _raw_active_index:
        set_active_record_index(active_row_index)

    active_row = df_window_filtered.iloc[active_row_index]
    active_record_id = str(active_row[record_id_column])
    active_record_plan = str(active_row.get("densegen__plan") or "unscoped").strip() or "unscoped"
    def _promoter_site_summary(raw_value: object) -> str:
        if raw_value is None:
            return ""
        if hasattr(raw_value, "as_py"):
            raw_value = raw_value.as_py()
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return ""
            try:
                raw_value = json.loads(text)
            except Exception:
                return ""
        if not isinstance(raw_value, dict):
            return ""
        placements = raw_value.get("placements", [])
        if hasattr(placements, "tolist"):
            placements = placements.tolist()
        if not isinstance(placements, (list, tuple)):
            return ""
        summaries = []
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            upstream_seq = str(placement.get("upstream_seq") or "").strip().upper()
            downstream_seq = str(placement.get("downstream_seq") or "").strip().upper()
            if upstream_seq or downstream_seq:
                summaries.append(f"-35={upstream_seq or '?'} / -10={downstream_seq or '?'}")
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        return "; ".join(summaries[:2])
    active_record_core_summary = _promoter_site_summary(active_row.get("densegen__promoter_detail"))
    filtered_n = len(df_window_filtered)
    mo.vstack(
        [
            mo.md("### BaseRender preview"),
            mo.hstack([record_plan_filter], justify="start", align="center"),
        ]
    )
    return active_record_core_summary, active_record_id, active_record_plan, active_row_index, filtered_n


@app.cell
def _(
    active_record,
    active_record_core_summary,
    active_record_id,
    active_record_plan,
    active_row_index,
    contract,
    filtered_n,
    mo,
    next_record_button,
    prev_record_button,
    render_record_figure,
):
    baserender_figure = render_record_figure(active_record, style_preset=contract.style_preset)
    baserender_figure.patch.set_facecolor("white")
    for axis in baserender_figure.axes:
        axis.set_facecolor("white")
    _title_text = f"ID: {active_record_id} | Plan: {active_record_plan}"
    if active_record_core_summary:
        _title_text += f"\\nCore: {active_record_core_summary}"
    baserender_figure.suptitle(_title_text, y=0.995, fontsize=11)
    _record_status = mo.md(
        "<div style='text-align:center'>"
        + f"`{active_row_index + 1} / {filtered_n}` | `id: {active_record_id}`"
        + "</div>"
    )
    _prev_slot = mo.hstack([prev_record_button], justify="start", align="center")
    _next_slot = mo.hstack([next_record_button], justify="end", align="center")
    _nav_row = mo.hstack(
        [_prev_slot, _record_status, _next_slot],
        justify="space-between",
        align="center",
        widths=[1, 6, 1],
        wrap=False,
    )
    mo.vstack([_nav_row, baserender_figure], align="stretch")
    return


@app.cell
def _(
    contract,
    df_window_filtered,
    load_records_from_parquet,
    preview_records_path,
    record_id_column,
    require,
    schema_names,
):
    record_ids = [str(record_id) for record_id in df_window_filtered[record_id_column].tolist()]
    adapter_columns = dict(contract.adapter_columns)
    if "densegen__promoter_detail" in schema_names and "promoter_detail" not in adapter_columns:
        adapter_columns["promoter_detail"] = "densegen__promoter_detail"
    records = load_records_from_parquet(
        dataset_path=preview_records_path,
        record_ids=record_ids,
        adapter_kind=contract.adapter_kind,
        adapter_columns=adapter_columns,
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
def _(PLOT_SPECS, mo, plot_entries, plot_manifest_load_error, require):
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

    def compact_plan_label(plan_name: str) -> str:
        _plan_text = str(plan_name or "").strip()
        if not _plan_text:
            return "run-level"
        if _plan_text == "unscoped":
            return "run-level"
        if _plan_text == "stage_a":
            return "stage-a"
        _parts = [part for part in _plan_text.split("__") if part]
        _base_label = str(_parts[0] if _parts else _plan_text).strip().replace("__", "-")
        _variant_tokens = []
        for _token in _parts[1:]:
            _token = str(_token).strip()
            if not _token:
                continue
            if "=" in _token:
                _key, _value = _token.split("=", 1)
            elif "_" in _token:
                _key, _value = _token.split("_", 1)
            else:
                _key, _value = _token, ""
            _key = str(_key).strip()
            _value = str(_value).strip()
            if _key and _value:
                _variant_tokens.append(f"{_key}={_value}")
        if not _variant_tokens:
            return _base_label
        _variant_label = " ".join(_variant_tokens)
        return f"{_base_label} [{_variant_label}]"

    _plan_names = sorted({str(entry["plan_name"]) for entry in plot_entries})
    all_scope_label = "all scopes (all plots)"
    plan_label_to_name = {all_scope_label: "all"}
    for _plan_name in _plan_names:
        _label_candidate = compact_plan_label(_plan_name)
        if _label_candidate in plan_label_to_name and plan_label_to_name[_label_candidate] != _plan_name:
            _label_candidate = f"{_label_candidate} [{_plan_name}]"
        plan_label_to_name[_label_candidate] = _plan_name

    plan_options = list(plan_label_to_name.keys())
    plot_scope_filter = mo.ui.dropdown(options=plan_options, value=plan_options[0], label="")
    return all_scope_label, compact_plan_label, plan_label_to_name, plot_scope_filter


@app.cell
def _(all_scope_label, mo, plan_label_to_name, plot_entries, plot_scope_filter):
    selected_scope_label = str(plot_scope_filter.value or all_scope_label)
    selected_plot_scope = str(plan_label_to_name.get(selected_scope_label, "all"))
    entries_for_scope = list(plot_entries)
    if selected_plot_scope != "all":
        entries_for_scope = [_entry for _entry in plot_entries if str(_entry["plan_name"]) == selected_plot_scope]

    plot_ids_by_scope = {}
    _plot_ids_all = sorted(
        {
            str(entry.get("plot_id") or "").strip()
            for entry in plot_entries
            if str(entry.get("plot_id") or "").strip()
        }
    )
    plot_ids_by_scope["all"] = _plot_ids_all
    for _plan_name in sorted({str(entry["plan_name"]) for entry in plot_entries}):
        plot_ids_by_scope[_plan_name] = sorted(
            {
                str(entry.get("plot_id") or "").strip()
                for entry in plot_entries
                if str(entry["plan_name"]) == _plan_name and str(entry.get("plot_id") or "").strip()
            }
        )
    plot_id_options = list(plot_ids_by_scope.get(selected_plot_scope, []))
    if selected_plot_scope == "all":
        if not plot_id_options:
            plot_id_options = ["(no plot types)"]
    elif not plot_id_options:
        plot_id_options = ["(no plot types)"]
    plot_id_filter = mo.ui.dropdown(options=plot_id_options, value=plot_id_options[0], label="")
    scope_plot_types_text = ", ".join(f"`{plot_id}`" for plot_id in plot_id_options if plot_id != "(no plot types)")
    if not scope_plot_types_text:
        scope_plot_types_text = "`(none)`"
    if selected_plot_scope == "all":
        scope_plot_types_message = "Scope has plot types: " + scope_plot_types_text
    else:
        scope_plot_types_message = (
            "Scope has plot types: "
            + scope_plot_types_text
            + f" (`{selected_scope_label}`)"
        )
    if selected_plot_scope == "stage_a" and all(
        str(entry.get("plot_id") or "") == "stage_a_summary" for entry in entries_for_scope
    ):
        scope_plot_types_message += (
            ". Stage-A PWM sampling panels are absent for this run because only background pools were sampled."
        )
    return entries_for_scope, plot_id_filter, plot_scope_filter, scope_plot_types_message, selected_scope_label


@app.cell
def _(
    entries_for_scope,
    compact_plan_label,
    pd,
    plot_id_filter,
    plot_scope_filter,
    scope_plot_types_message,
    selected_scope_label,
):
    selected_plot_id = str(plot_id_filter.value or "")
    _filtered_entries = [
        _entry
        for _entry in entries_for_scope
        if str(_entry.get("plot_id") or "").strip() == selected_plot_id
    ]

    label_to_entry = {}
    plot_filter_message = ""

    plot_options = []
    if selected_plot_id == "(no plot types)":
        plot_filter_message = (
            "No plot types are available for scope `"
            + selected_scope_label
            + "`. Select another scope."
        )
        plot_options = ["(no plots for current filters)"]
    elif not _filtered_entries:
        plot_filter_message = (
            "No plots found for scope `"
            + selected_scope_label
            + "` and plot type `"
            + selected_plot_id
            + "`. Select another scope or plot type."
        )
        plot_options = ["(no plots for current filters)"]
    else:
        for _entry_index, _entry in enumerate(_filtered_entries):
            _plan = str(_entry["plan_name"])
            compact_plan_name = compact_plan_label(_plan)
            _variant = str(_entry["variant"]).strip()
            _label = str(_entry["plot_name"])
            if _variant:
                _label = f"{_label} ({_variant})"
            _option_label = f"{_entry_index + 1}. [{compact_plan_name}] {_label}"
            plot_options.append(_option_label)
            label_to_entry[_option_label] = _entry

    plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="")
    return (
        label_to_entry,
        plot_filter_message,
        plot_id_filter,
        scope_plot_types_message,
        plot_scope_filter,
        plot_selector,
    )


@app.cell
def _(label_to_entry, plot_filter_message, plot_selector):
    _selected_plot_option = str(plot_selector.value or "")
    active_plot_error = str(plot_filter_message or "").strip()
    active_plot_entry = None
    if not active_plot_error and _selected_plot_option not in label_to_entry:
        active_plot_error = "Selected plot is not available for the current plan filter."
    if not active_plot_error and _selected_plot_option in label_to_entry:
        active_plot_entry = label_to_entry[_selected_plot_option]
    return active_plot_entry, active_plot_error


@app.cell
def _(Path, hashlib, plot_manifest_path, shutil, subprocess):
    preview_dir = plot_manifest_path.parent / ".preview_png"
    _image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}

    def resolve_plot_preview_image(plot_path: Path) -> Path:
        source_path = Path(plot_path).expanduser().resolve()
        suffix = str(source_path.suffix).lower()
        if suffix in _image_suffixes:
            return source_path
        if suffix != ".pdf":
            raise RuntimeError(f"Unsupported plot format: `{source_path.suffix}`.")

        preview_dir.mkdir(parents=True, exist_ok=True)

        ghostscript = shutil.which("gs")
        pdftoppm = shutil.which("pdftoppm")
        magick = shutil.which("magick")
        convert = shutil.which("convert")
        sips = shutil.which("sips")
        command_signature = "|".join(
            label
            for label, enabled in (
                ("gs", bool(ghostscript)),
                ("pdftoppm", bool(pdftoppm)),
                ("magick", bool(magick)),
                ("convert", bool(convert)),
                ("sips", bool(sips)),
            )
            if enabled
        )
        preview_version = "preview-v2-gs450"
        digest = hashlib.sha1(f"{source_path}|{command_signature}|{preview_version}".encode("utf-8")).hexdigest()[:12]
        preview_path = preview_dir / f"{source_path.stem}__{digest}.png"
        if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
            return preview_path

        for stale_preview in preview_dir.glob(f"{source_path.stem}__*.png"):
            if stale_preview == preview_path:
                continue
            stale_preview.unlink(missing_ok=True)

        command_groups = []
        if ghostscript:
            command_groups.append(
                [
                    ghostscript,
                    "-dSAFER",
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=pngalpha",
                    "-r450",
                    "-dFirstPage=1",
                    "-dLastPage=1",
                    f"-sOutputFile={preview_path}",
                    str(source_path),
                ]
            )
        if pdftoppm:
            command_groups.append(
                [
                    pdftoppm,
                    "-png",
                    "-singlefile",
                    "-r",
                    "450",
                    str(source_path),
                    str(preview_path.with_suffix("")),
                ]
            )
        if magick:
            command_groups.append(
                [magick, "-density", "450", f"{source_path}[0]", "-quality", "100", str(preview_path)]
            )
        if convert:
            command_groups.append(
                [convert, "-density", "450", f"{source_path}[0]", "-quality", "100", str(preview_path)]
            )
        if sips:
            command_groups.append([sips, "-s", "format", "png", str(source_path), "--out", str(preview_path)])

        if preview_path.exists():
            preview_path.unlink()
        for command in command_groups:
            try:
                subprocess.run(command, check=True, capture_output=True)
            except Exception:
                continue
            if preview_path.exists() and preview_path.stat().st_size > 0:
                return preview_path
        raise RuntimeError(
            "Unable to render PDF plot preview. Install `gs`, `pdftoppm`, "
            "`magick`, `convert`, or `sips` for PNG previews."
        )

    return resolve_plot_preview_image


@app.cell
def _(
    active_plot_entry,
    active_plot_error,
    mo,
    plot_id_filter,
    plot_scope_filter,
    plot_selector,
    resolve_plot_preview_image,
    scope_plot_types_message,
):
    _controls = mo.hstack(
        [plot_scope_filter, plot_id_filter, plot_selector],
        justify="space-between",
        align="center",
        wrap=True,
        widths=[2, 2, 5],
    )
    _scope_help = mo.md(
        "Scope guide: `run-level` = run-health plots, `stage-a` = Stage-A pool diagnostics, "
        "`<plan>` = Stage-B plan-scoped plots."
    )
    _scope_types = mo.md(scope_plot_types_message)
    _content = [mo.md("### Plot gallery"), _scope_help, _scope_types, _controls]
    if active_plot_entry is None:
        _content.append(mo.md(str(active_plot_error or "No plot selected.")))
    else:
        _plan_name = str(active_plot_entry["plan_name"])
        _plot_id = str(active_plot_entry["plot_id"])
        _plot_name = str(active_plot_entry["plot_name"])
        _variant = str(active_plot_entry["variant"]).strip()
        _plot_path = active_plot_entry["path"]
        _variant_text = _variant if _variant else "none"
        try:
            _preview_path = resolve_plot_preview_image(_plot_path)
            _preview_error = ""
        except Exception as exc:
            _preview_path = None
            _preview_error = str(exc)
        if _preview_path is not None:
            _content.append(mo.image(str(_preview_path)))
        else:
            if str(getattr(_plot_path, "suffix", "")).lower() == ".pdf":
                _content.append(mo.pdf(str(_plot_path)))
                _content.append(
                    mo.md(
                        "PNG preview unavailable: "
                        + _preview_error
                        + ". Showing PDF directly for this plot."
                    )
                )
            else:
                _content.append(
                    mo.md(
                        "Preview unavailable: "
                        + _preview_error
                        + ". Install `gs`, `pdftoppm`, `magick`, `convert`, or `sips` for PNG plot previews."
                    )
                )
    mo.vstack(_content)
    return


@app.cell
def _(mo, run_root):
    plot_export_target = mo.ui.dropdown(
        options=["selected", "filtered", "all"],
        value="selected",
        label="",
    )
    plot_export_format = mo.ui.dropdown(options=["pdf", "png", "svg"], value="png", label="")
    default_plot_export_dir = run_root / "outputs" / "notebooks" / "plots_export"
    plot_export_path = mo.ui.text(value=str(default_plot_export_dir), label="Plot export directory", full_width=True)
    plot_export_button = mo.ui.run_button(label="Export", kind="neutral")
    mo.vstack(
        [
            mo.md("### Plot export"),
            mo.md("Export selected, filtered, or all plots into one format."),
            mo.hstack(
                [
                    plot_export_target,
                    plot_export_format,
                    plot_export_path,
                    plot_export_button,
                ],
                justify="space-between",
                align="end",
                widths=[2, 2, 6, 1],
                wrap=False,
            ),
        ],
        align="stretch",
    )
    return plot_export_button, plot_export_format, plot_export_path, plot_export_target


@app.cell
def _(
    Path,
    active_plot_entry,
    label_to_entry,
    mo,
    plot_entries,
    plot_export_button,
    plot_export_format,
    plot_export_path,
    plot_export_target,
    require,
    resolve_plot_preview_image,
    run_root,
    shutil,
    subprocess,
):
    _plot_click_count = int(plot_export_button.value or 0)
    _plot_status_text = ""
    if _plot_click_count > 0:
        _selected_target = str(plot_export_target.value or "selected").strip()
        require(
            _selected_target not in {"selected", "filtered", "all"},
            f"Plot export set must be selected|filtered|all, got `{_selected_target}`.",
        )
        _selected_format = str(plot_export_format.value or "").strip()
        require(
            _selected_format not in {"pdf", "png", "svg"},
            f"Plot export format must be pdf|png|svg, got `{_selected_format}`.",
        )
        _raw_export_dir = str(plot_export_path.value or "").strip()
        require(not _raw_export_dir, "Plot export directory cannot be empty.")
        _export_dir = Path(_raw_export_dir).expanduser()
        if not _export_dir.is_absolute():
            _export_dir = run_root / _export_dir
        _export_dir.mkdir(parents=True, exist_ok=True)

        if _selected_target == "selected":
            require(active_plot_entry is None, "No selected plot is available to export.")
            _entries = [active_plot_entry]
        elif _selected_target == "filtered":
            _entries = list(label_to_entry.values())
            require(not _entries, "No filtered plots are available to export.")
        else:
            _entries = list(plot_entries)
            require(not _entries, "No plots are available to export.")

    def _slug(value: str) -> str:
        text = str(value or "").strip().replace("__", "_")
        keep = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                keep.append(ch)
            else:
                keep.append("-")
        slug = "".join(keep).strip("-_")
        return slug or "plot"

    def _export_plot(source_path: Path, destination_path: Path, fmt: str) -> None:
        source_suffix = str(source_path.suffix).lower()
        if source_suffix == f".{fmt}":
            shutil.copy2(source_path, destination_path)
            return
        if fmt == "png":
            png_source = resolve_plot_preview_image(source_path)
            shutil.copy2(png_source, destination_path)
            return
        if fmt == "svg":
            if source_suffix == ".pdf":
                pdftocairo = shutil.which("pdftocairo")
                if not pdftocairo:
                    raise RuntimeError(
                        "SVG export from PDF requires `pdftocairo` to be installed and available in PATH."
                    )
                output_root = destination_path.with_suffix("")
                subprocess.run(
                    [pdftocairo, "-svg", str(source_path), str(output_root)],
                    check=True,
                    capture_output=True,
                )
                generated_svg = output_root.with_suffix(".svg")
                if not generated_svg.exists() or generated_svg.stat().st_size <= 0:
                    raise RuntimeError(f"Failed to export SVG for `{source_path.name}`.")
                if generated_svg != destination_path:
                    shutil.move(str(generated_svg), str(destination_path))
                return
            raise RuntimeError(
                f"Cannot export `{source_path.name}` to SVG. Only PDF sources can be exported to SVG."
            )
        if fmt == "pdf":
            magick = shutil.which("magick")
            convert = shutil.which("convert")
            commands = []
            if magick:
                commands.append([magick, str(source_path), str(destination_path)])
            if convert:
                commands.append([convert, str(source_path), str(destination_path)])
            for command in commands:
                try:
                    subprocess.run(command, check=True, capture_output=True)
                except Exception:
                    continue
                if destination_path.exists() and destination_path.stat().st_size > 0:
                    return
            raise RuntimeError(
                f"Cannot export `{source_path.name}` to PDF. Install `magick` or `convert` to enable this conversion."
            )
        raise RuntimeError(f"Unsupported plot export format `{fmt}`.")

    if _plot_click_count > 0:
        _exported_n = 0
        for _idx, _entry in enumerate(_entries):
            _source_path = Path(_entry["path"]).expanduser().resolve()
            _plan_name = _slug(str(_entry.get("plan_name") or "run"))
            _plot_name = _slug(str(_entry.get("plot_id") or _entry.get("plot_name") or _source_path.stem))
            _variant = _slug(str(_entry.get("variant") or "default"))
            _destination_path = (
                _export_dir / f"{_idx + 1:03d}__{_plan_name}__{_plot_name}__{_variant}.{_selected_format}"
            )
            _export_plot(_source_path, _destination_path, _selected_format)
            _exported_n += 1
        _plot_status_text = "Saved `" + str(_exported_n) + "` plot(s) to `" + str(_export_dir) + "`."
    mo.md(_plot_status_text)
    return


""" + _records_export_cell_template()


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
