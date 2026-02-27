"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_cells_template_base.py

Base marimo notebook cell template segment for DenseGen notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

NOTEBOOK_TEMPLATE_CELLS_BASE = r"""
@app.cell
def _():
    from functools import lru_cache
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
    from dnadesign.densegen.src.cli.notebook_export_paths import (
        resolve_baserender_export_destination,
        resolve_records_export_destination,
    )
    from dnadesign.densegen.src.cli.notebook_records_projection import (
        build_records_preview_table,
    )

    def require(condition: bool, message: str) -> None:
        if bool(condition):
            raise RuntimeError(message)

    return (
        ParquetFile,
        Path,
        lru_cache,
        hashlib,
        densegen_notebook_render_contract,
        json,
        PLOT_SPECS,
        resolve_baserender_export_destination,
        resolve_records_export_destination,
        build_records_preview_table,
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
    def _find_repo_root(start_path: Path) -> Path:
        candidate = Path(start_path).expanduser().resolve()
        for root in (candidate, *candidate.parents):
            if (root / "pyproject.toml").exists() or (root / ".git").exists():
                return root
        return candidate
    repo_root = _find_repo_root(run_root)
    def to_repo_relative_path(path: Path) -> str:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = run_root / candidate
        resolved = candidate.resolve()
        try:
            return str(resolved.relative_to(repo_root))
        except Exception:
            return str(resolved)
    workspace_name = str(config_path.parent.name or run_root.name)
    workspace_heading = __WORKSPACE_HEADING__
    workspace_run_details_payload = __WORKSPACE_RUN_DETAILS_PAYLOAD__
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
        repo_root,
        run_root,
        to_repo_relative_path,
        workspace_heading,
        workspace_run_details_payload,
        usr_dataset,
        usr_root,
        workspace_name,
    )


@app.cell
def _(mo, workspace_heading, workspace_run_details_payload):
    payload = (
        workspace_run_details_payload
        if isinstance(workspace_run_details_payload, dict)
        else {}
    )
    summary_lines = payload.get("summary_lines", [])
    section_items_raw = payload.get("sections", [])
    summary_text = "\\n".join(
        str(line).strip() for line in summary_lines if str(line).strip()
    )

    section_items: dict[str, object] = {}
    if isinstance(section_items_raw, list):
        for item in section_items_raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            body_md = str(item.get("body_md") or "").strip()
            if title:
                section_items[title] = mo.md(body_md or "_No details available._")

    _run_details_blocks: list[object] = [
        mo.md(f"# {workspace_heading}"),
        mo.md("## Run details"),
    ]
    if summary_text:
        _run_details_blocks.append(mo.md(summary_text))
    if section_items:
        _run_details_blocks.append(mo.accordion(section_items, multiple=True))
    mo.vstack(_run_details_blocks, align="stretch", gap=0.4)
    return


@app.cell
def _(config_path, json, require, run_manifest_path, yaml):
    config_load_error = None
    run_manifest_load_error = None

    if run_manifest_path.exists():
        try:
            json.loads(run_manifest_path.read_text())
        except Exception as exc:
            run_manifest_load_error = f"Failed to parse `run_manifest.json`: {exc}"
    require(run_manifest_load_error is not None, run_manifest_load_error or "Run manifest is invalid.")

    try:
        yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        config_load_error = str(exc)
    require(config_load_error is not None, f"Failed to parse `config.yaml`: {config_load_error}")
    return


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

    _default_plan_value = "all"

    record_plan_filter = mo.ui.dropdown(options=_plan_options, value=_default_plan_value, label="Record plan")
    return has_plan_column, record_plan_filter


@app.cell
def _(
    build_records_preview_table,
    df_window,
    get_active_record_index,
    has_plan_column,
    record_id_column,
    record_plan_filter,
    require,
    set_active_record_index,
):
    _selected_record_plan = str(record_plan_filter.value or "all")
    if _selected_record_plan == "all" or not has_plan_column:
        df_window_filtered = df_window.reset_index(drop=True)
    else:
        _mask = df_window["densegen__plan"].astype(str) == _selected_record_plan
        df_window_filtered = df_window[_mask].reset_index(drop=True)
    df_window_filtered = build_records_preview_table(df_window_filtered)
    require(
        df_window_filtered.empty,
        f"No records found for plan `{_selected_record_plan}` in preview window.",
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
def _(df_window_filtered, mo, record_plan_filter, run_root, to_repo_relative_path):
    export_format = mo.ui.dropdown(options=["parquet", "csv"], value="parquet", label="")
    default_export_path = run_root / "outputs" / "notebooks" / "records_preview.parquet"
    default_export_path_text = to_repo_relative_path(default_export_path)
    export_path = mo.ui.text(value=str(default_export_path_text), label="", full_width=True)
    export_button = mo.ui.run_button(label="Export", kind="neutral")
    export_controls = mo.hstack(
        [export_format, export_path, export_button],
        justify="start",
        align="end",
        gap=0.2,
        widths=[1.1, 8.0, 0.9],
        wrap=False,
    )
    _selected_record_plan = str(record_plan_filter.value or "all")
    export_details = mo.accordion(
        {
            "Dataset export details": mo.md(
                "\\n".join(
                    [
                        f"- Rows in view: `{len(df_window_filtered)}`",
                        f"- Columns in view: `{len(df_window_filtered.columns)}`",
                        f"- Record plan filter: `{_selected_record_plan}`",
                        "- Path behavior: relative export paths resolve from the repository root.",
                    ]
                )
            )
        },
        multiple=True,
    )
    mo.vstack(
        [
            mo.md("### Records preview"),
            mo.md(
                "Export writes the currently filtered records table from this notebook "
                "to the selected format and path."
            ),
            mo.ui.table(df_window_filtered.loc[:, list(df_window_filtered.columns)]),
            mo.md("Dataset export path"),
            export_controls,
            export_details,
        ]
    )
    return export_button, export_format, export_path


@app.cell
def _(json):
    def summarize_promoter_sites(raw_value: object) -> str:
        if raw_value is None:
            return ""
        if hasattr(raw_value, "as_py"):
            raw_value = raw_value.as_py()
        if hasattr(raw_value, "tolist"):
            raw_value = raw_value.tolist()
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return ""
            try:
                raw_value = json.loads(text)
            except Exception:
                return ""
        if isinstance(raw_value, dict):
            placements = raw_value.get("placements", [])
        elif isinstance(raw_value, (list, tuple)):
            placements = [
                entry
                for entry in raw_value
                if isinstance(entry, dict) and entry.get("part_kind") == "fixed_element"
            ]
        else:
            return ""
        if hasattr(placements, "tolist"):
            placements = placements.tolist()
        if not isinstance(placements, (list, tuple)):
            return ""
        upstream = None
        downstream = None
        summaries = []
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            role = str(placement.get("role") or "").strip().lower()
            if role == "upstream":
                upstream = str(placement.get("sequence") or placement.get("upstream_seq") or "").strip().upper()
            elif role == "downstream":
                downstream = str(placement.get("sequence") or placement.get("downstream_seq") or "").strip().upper()
            elif "upstream_seq" in placement or "downstream_seq" in placement:
                upstream = str(placement.get("upstream_seq") or "").strip().upper()
                downstream = str(placement.get("downstream_seq") or "").strip().upper()
            if upstream or downstream:
                summaries.append(f"-35={upstream or '?'} / -10={downstream or '?'}")
                upstream = None
                downstream = None
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        return "; ".join(summaries[:2])
    return summarize_promoter_sites


@app.cell
def _(contract, render_record_figure, workspace_name):
    def build_baserender_figure(record, *, core_summary: str):
        _legend_pad_px = 20.0
        _legend_patch_h = 13.0
        _title_font_size = 14
        _title_y = 0.968
        _figure = render_record_figure(
            record,
            style_preset=contract.style_preset,
            style_overrides={
                "dpi": 320,
                "padding_y": 10.0,
                "layout": {"outer_pad_cells": 0.62},
                "sequence": {"to_kmer_gap_cells": 0.38},
                "legend_pad_px": _legend_pad_px,
                "legend_patch_w": 28.0,
                "legend_patch_h": _legend_patch_h,
                "legend_font_size": 14,
                "legend_gap_patch_text": 11.0,
                "legend_gap_x": 44.0,
            },
        )
        _figure.patch.set_facecolor("white")
        for _axis in _figure.axes:
            _axis.set_facecolor("white")

        _record_id = str(getattr(record, "id", "") or "unknown")
        _header_text = f"{workspace_name} | sequence {_record_id}"

        _axis = _figure.axes[0] if _figure.axes else None
        if _axis is None:
            raise RuntimeError("BaseRender preview figure expected one axes for title placement.")
        _axis.text(
            0.5,
            _title_y,
            _header_text,
            transform=_axis.transAxes,
            ha="center",
            va="top",
            fontsize=_title_font_size,
            color="#111827",
            zorder=20.0,
            clip_on=False,
        )
        return _figure

    return build_baserender_figure


@app.cell
def _(
    df_window_filtered,
    get_active_record_index,
    mo,
    record_count,
    record_id_column,
    record_plan_filter,
    set_active_record_index,
    summarize_promoter_sites,
):
    _raw_active_index = int(get_active_record_index() or 0)
    active_row_index = max(0, min(record_count - 1, _raw_active_index))
    if active_row_index != _raw_active_index:
        set_active_record_index(active_row_index)

    active_row = df_window_filtered.iloc[active_row_index]
    active_record_id = str(active_row[record_id_column])
    active_record_core_summary = summarize_promoter_sites(active_row.get("densegen__parts_detail"))
    filtered_n = len(df_window_filtered)
    mo.vstack(
        [
            mo.md("### BaseRender preview"),
            mo.hstack([record_plan_filter], justify="start", align="center"),
        ]
    )
    return active_record_core_summary, active_record_id, active_row_index, filtered_n


@app.cell
def _(
    active_record_id,
    active_row_index,
    run_root,
    filtered_n,
    mo,
    next_record_button,
    prev_record_button,
    render_baserender_preview_path,
    to_repo_relative_path,
):
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
    _baserender_image = mo.image(
        render_baserender_preview_path(active_record_id),
        rounded=True,
        style={
            "border-radius": "14px",
            "width": "100%",
            "max-height": "460px",
            "height": "auto",
            "background": "white",
            "display": "block",
        },
    )
    baserender_export_format = mo.ui.dropdown(options=["png", "pdf"], value="png", label="")
    default_baserender_export_path = run_root / "outputs" / "notebooks" / "baserender_preview.png"
    default_baserender_export_path_text = to_repo_relative_path(default_baserender_export_path)
    baserender_export_path = mo.ui.text(value=str(default_baserender_export_path_text), label="", full_width=True)
    baserender_export_button = mo.ui.run_button(label="Export", kind="neutral")
    _baserender_export_controls = mo.hstack(
        [baserender_export_format, baserender_export_path, baserender_export_button],
        justify="start",
        align="end",
        gap=0.2,
        widths=[1.1, 8.0, 0.9],
        wrap=False,
    )
    mo.vstack(
        [
            _nav_row,
            _baserender_image,
            mo.md("BaseRender export path"),
            _baserender_export_controls,
        ],
        align="stretch",
    )
    return baserender_export_button, baserender_export_format, baserender_export_path


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
def _(
    Path,
    build_baserender_figure,
    df_window_filtered,
    hashlib,
    lru_cache,
    record_id_column,
    records_by_id,
    run_root,
    summarize_promoter_sites,
):
    import matplotlib.pyplot as plt

    preview_meta_by_id = {}
    for _, _row in df_window_filtered.iterrows():
        _record_id = str(_row[record_id_column])
        preview_meta_by_id[_record_id] = {
            "core_summary": summarize_promoter_sites(_row.get("densegen__parts_detail")),
        }

    preview_cache_dir = run_root / "outputs" / "notebooks" / ".baserender_preview_cache"
    preview_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(record_id: str) -> Path:
        digest = hashlib.sha1(str(record_id).encode("utf-8")).hexdigest()[:16]
        return preview_cache_dir / f"{digest}.png"

    @lru_cache(maxsize=64)
    def render_baserender_preview_path(record_id: str) -> str:
        _record_key = str(record_id)
        _meta = preview_meta_by_id.get(_record_key)
        if _meta is None:
            raise RuntimeError(f"BaseRender preview metadata missing record `{_record_key}`.")
        _record = records_by_id.get(_record_key)
        if _record is None:
            raise RuntimeError(f"BaseRender preview cache missing record `{_record_key}`.")
        _destination = _cache_path(_record_key)
        _figure = build_baserender_figure(
            _record,
            core_summary=str(_meta.get("core_summary") or ""),
        )
        _figure.savefig(
            _destination,
            format="png",
            dpi=_figure.dpi,
        )
        plt.close(_figure)
        return str(_destination)

    return render_baserender_preview_path


@app.cell
def _(active_row_index, df_window_filtered, record_id_column, render_baserender_preview_path):
    prefetch_indices = (active_row_index, active_row_index - 1, active_row_index + 1)
    for _prefetch_index in prefetch_indices:
        if _prefetch_index < 0 or _prefetch_index >= len(df_window_filtered):
            continue
        _prefetch_id = str(df_window_filtered.iloc[_prefetch_index][record_id_column])
        render_baserender_preview_path(_prefetch_id)
    return


@app.cell
def _(active_record_id, records_by_id, require):
    require(active_record_id not in records_by_id, f"Record `{active_record_id}` missing from preview cache.")
    active_record = records_by_id[active_record_id]
    return active_record


"""
