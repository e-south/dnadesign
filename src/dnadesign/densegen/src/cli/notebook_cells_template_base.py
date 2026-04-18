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
    from io import BytesIO
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    import textwrap

    import marimo as mo
    import pandas as pd
    import yaml
    from pyarrow.parquet import ParquetFile

    from dnadesign.baserender import adapt_records
    from dnadesign.baserender import render_record_figure
    from dnadesign.densegen import PLOT_SPECS, densegen_notebook_render_contract
    from dnadesign.densegen.src.integrations.baserender.notebook_contract import (
        densegen_baserender_title_text,
        densegen_video_subtitle_text,
    )
    from dnadesign.densegen.src.cli.notebook_export_paths import (
        resolve_baserender_export_destination,
        resolve_records_export_destination,
    )
    from dnadesign.densegen.src.cli.notebook_records_projection import (
        build_records_preview_table,
    )
    from dnadesign.densegen.src.core.record_metadata_recovery import (
        recover_densegen_metadata_from_source,
    )
    from dnadesign.densegen.src.core.record_values import (
        coerce_list_of_dicts,
    )
    from dnadesign.densegen.src.viz.plot_inventory import (
        HIDDEN_VISUAL_PLOT_TYPES,
        base_plot_id,
        compact_plan_label,
        describe_visual_plot_type,
        load_current_inventory_strict,
        notebook_visible_plot_ids,
        plot_missing_hint,
        plot_required_artifacts,
        resolve_plot_availability,
        resolve_plot_record,
    )

    def require(condition: bool, message: str) -> None:
        if bool(condition):
            raise RuntimeError(message)

    def consume_click(click_count: int, last_handled: int) -> tuple[bool, int]:
        _click_count = max(0, int(click_count or 0))
        _last_handled = max(0, int(last_handled or 0))
        if _click_count <= _last_handled:
            return False, _last_handled
        return True, _click_count

    def _load_usr_preview_dataframe(
        *,
        pd,
        preview_limit: int,
        required_columns: set[str],
        usr_root: Path,
        usr_dataset: str,
    ) -> tuple[pd.DataFrame, int, list[str], Path]:
        from dnadesign.usr import Dataset

        ds = Dataset(usr_root, str(usr_dataset))
        require(not ds.records_path.exists(), f"USR records not found: {ds.records_path}")
        stats = ds.stats()
        total_rows = int(getattr(stats, "rows", 0) or 0)
        require(total_rows <= 0, f"`{ds.records_path.name}` is empty.")
        window_n = min(total_rows, max(1, int(preview_limit)))
        remaining = int(window_n)
        batches = []
        for batch in ds.scan(
            columns=None,
            include_overlays=True,
            include_deleted=False,
            batch_size=min(1024, max(1, int(preview_limit))),
        ):
            frame = batch.to_pandas()
            if remaining < len(frame):
                frame = frame.iloc[:remaining]
            batches.append(frame)
            remaining -= len(frame)
            if remaining <= 0:
                break
        df_window = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        missing = sorted(required_columns - set(df_window.columns))
        return df_window, total_rows, missing, ds.records_path

    def load_preview_records(
        *,
        ParquetFile,
        pd,
        preview_limit: int,
        output_source: str,
        records_path: Path,
        recover_densegen_metadata_from_source,
        required_columns: set[str],
        usr_dataset: str | None,
        usr_root: Path | None,
    ) -> dict[str, object]:
        if str(output_source or "").strip() == "usr":
            require(
                usr_root is None or not str(usr_dataset or "").strip(),
                "Notebook source is USR but generation context does not include a dataset path.",
            )
            df_window, total_rows, missing, preview_source_path = _load_usr_preview_dataframe(
                pd=pd,
                preview_limit=preview_limit,
                required_columns=required_columns,
                usr_root=usr_root,
                usr_dataset=str(usr_dataset),
            )
        else:
            preview_source_path = records_path
            try:
                parquet_file = ParquetFile(preview_source_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to open `{preview_source_path.name}`: {exc}") from exc
            schema_names = set(parquet_file.schema_arrow.names)
            missing = sorted(required_columns - schema_names)
            total_rows = int(parquet_file.metadata.num_rows or 0)
            require(total_rows <= 0, f"`{preview_source_path.name}` is empty.")
            window_n = min(total_rows, max(1, int(preview_limit)))
            remaining = int(window_n)
            batches = []
            preview_columns = list(parquet_file.schema_arrow.names)
            for batch in parquet_file.iter_batches(
                columns=preview_columns,
                batch_size=min(1024, max(1, int(preview_limit))),
            ):
                frame = batch.to_pandas()
                if remaining < len(frame):
                    frame = frame.iloc[:remaining]
                batches.append(frame)
                remaining -= len(frame)
                if remaining <= 0:
                    break
            df_window = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=preview_columns)
        require(
            bool(missing),
            f"`{preview_source_path.name}` missing required columns: {missing}. "
            "DenseGen BaseRender preview requires id, sequence, and densegen placement detail.",
        )
        df_window = recover_densegen_metadata_from_source(df_window)
        if "densegen__used_tfbs_detail" in df_window.columns:
            df_window["densegen__used_tfbs_detail"] = [
                coerce_list_of_dicts(value) for value in df_window["densegen__used_tfbs_detail"].tolist()
            ]
        require(df_window.empty, "No rows available in preview window.")
        return {
            "preview_rows": df_window.reset_index(drop=True),
            "preview_source_path": preview_source_path,
            "preview_strategy": "head_window",
            "preview_total_rows": int(total_rows),
            "preview_window_limit": max(1, int(preview_limit)),
        }

    return (
        BytesIO,
        HIDDEN_VISUAL_PLOT_TYPES,
        ParquetFile,
        Path,
        adapt_records,
        base_plot_id,
        compact_plan_label,
        consume_click,
        describe_visual_plot_type,
        densegen_baserender_title_text,
        densegen_notebook_render_contract,
        densegen_video_subtitle_text,
        json,
        load_current_inventory_strict,
        load_preview_records,
        notebook_visible_plot_ids,
        PLOT_SPECS,
        plot_missing_hint,
        plot_required_artifacts,
        resolve_plot_availability,
        resolve_plot_record,
        resolve_baserender_export_destination,
        resolve_records_export_destination,
        build_records_preview_table,
        coerce_list_of_dicts,
        mo,
        pd,
        require,
        recover_densegen_metadata_from_source,
        render_record_figure,
        shutil,
        subprocess,
        tempfile,
        textwrap,
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
    workspace_plan_names = __WORKSPACE_PLAN_NAMES__
    workspace_run_details_payload = __WORKSPACE_RUN_DETAILS_PAYLOAD__
    records_path = Path(__RECORDS_PATH__)
    output_source = __OUTPUT_SOURCE__
    usr_root_text = __USR_ROOT__
    usr_root = Path(usr_root_text) if usr_root_text else None
    usr_dataset = __USR_DATASET__
    contract = densegen_notebook_render_contract()
    record_window_limit = int(contract.record_window_limit)
    run_manifest_path = run_root / "outputs" / "meta" / "run_manifest.json"
    plot_inventory_path = run_root / "outputs" / "plots" / "current_inventory.json"
    return (
        config_path,
        contract,
        output_source,
        plot_inventory_path,
        record_window_limit,
        records_path,
        run_manifest_path,
        repo_root,
        run_root,
        to_repo_relative_path,
        workspace_heading,
        workspace_plan_names,
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
    load_preview_records,
    output_source,
    pd,
    record_window_limit,
    records_path,
    require,
    usr_dataset,
    usr_root,
):
    record_id_column = str(contract.adapter_columns["id"])
    preview_payload = load_preview_records(
        ParquetFile=ParquetFile,
        pd=pd,
        preview_limit=int(record_window_limit),
        output_source=str(output_source or "").strip(),
        records_path=records_path,
        recover_densegen_metadata_from_source=recover_densegen_metadata_from_source,
        required_columns={
        contract.adapter_columns["id"],
        contract.adapter_columns["sequence"],
        contract.adapter_columns["annotations"],
        },
        usr_dataset=usr_dataset,
        usr_root=usr_root,
    )
    df_window = preview_payload["preview_rows"]
    preview_source_path = preview_payload["preview_source_path"]
    preview_strategy = str(preview_payload["preview_strategy"])
    preview_total_rows = int(preview_payload["preview_total_rows"])
    preview_window_limit = int(preview_payload["preview_window_limit"])
    require(df_window.empty, "No rows available in preview window.")
    duplicate_id_count = int(df_window[record_id_column].astype(str).duplicated().sum())
    require(
        duplicate_id_count > 0,
        "Duplicate record ids detected in the notebook preview window "
        f"({duplicate_id_count}). Resolve id collisions in `{preview_source_path.name}` and rerun.",
    )
    return (
        df_window,
        preview_strategy,
        preview_total_rows,
        preview_window_limit,
        record_id_column,
    )


@app.cell
def _(mo):
    get_active_record_index, set_active_record_index = mo.state(0)
    return get_active_record_index, set_active_record_index


@app.cell
def _(mo):
    get_baserender_display_payload, set_baserender_display_payload = mo.state({})
    return get_baserender_display_payload, set_baserender_display_payload


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
    has_plan_column,
    preview_strategy,
    preview_total_rows,
    preview_window_limit,
    record_id_column,
    record_plan_filter,
    require,
):
    _selected_record_plan = str(record_plan_filter.value or "all")
    if _selected_record_plan == "all" or not has_plan_column:
        preview_rows_filtered = df_window.reset_index(drop=True)
    else:
        _mask = df_window["densegen__plan"].astype(str) == _selected_record_plan
        preview_rows_filtered = df_window[_mask].reset_index(drop=True)
    df_window_filtered = build_records_preview_table(preview_rows_filtered)
    require(
        df_window_filtered.empty,
        f"No records found for plan `{_selected_record_plan}` in preview window.",
    )
    record_count = int(len(df_window_filtered[record_id_column]))
    require(record_count <= 0, "No records are available in the selected preview window.")
    return (
        df_window_filtered,
        preview_rows_filtered,
        preview_strategy,
        preview_total_rows,
        preview_window_limit,
        record_count,
        record_plan_filter,
    )


@app.cell
def _(get_active_record_index, record_count, set_active_record_index):
    _raw_active_index = int(get_active_record_index() or 0)
    active_record_index_default = max(0, min(record_count - 1, _raw_active_index))
    if active_record_index_default != _raw_active_index:
        set_active_record_index(active_record_index_default)
    return active_record_index_default


@app.cell
def _(
    active_record_index_default,
    mo,
    record_count,
    set_active_record_index,
):

    record_index_slider = mo.ui.slider(
        1, record_count, value=active_record_index_default + 1, step=1, label="Record"
    )
    record_index_jump = mo.ui.number(value=active_record_index_default + 1, label="Jump to")

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
        next_record_button,
        prev_record_button,
        record_index_jump,
        record_index_slider,
    )


@app.cell
def _(
    get_active_record_index,
    record_count,
    record_index_jump,
    record_index_slider,
    set_active_record_index,
):
    _raw_active_index = int(get_active_record_index() or 0)
    synced_active_index = max(0, min(record_count - 1, _raw_active_index))
    _slider_index = max(0, min(record_count - 1, int(record_index_slider.value or (synced_active_index + 1)) - 1))
    _jump_raw = record_index_jump.value
    try:
        _jump_index = max(0, min(record_count - 1, int(_jump_raw) - 1))
    except Exception:
        _jump_index = synced_active_index
    if _slider_index != synced_active_index:
        set_active_record_index(_slider_index)
    elif _jump_index != synced_active_index:
        set_active_record_index(_jump_index)
    return


@app.cell
def _(
    df_window_filtered,
    mo,
    preview_strategy,
    preview_total_rows,
    preview_window_limit,
    record_plan_filter,
    run_root,
    to_repo_relative_path,
):
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
                "\n".join(
                    [
                        f"- Rows in view: `{len(df_window_filtered)}`",
                        f"- Accepted records in source: `{int(preview_total_rows)}`",
                        f"- Preview window limit: `{int(preview_window_limit)}` rows",
                        f"- Preview strategy: `{str(preview_strategy)}` (head-window)",
                        f"- Columns in view: `{len(df_window_filtered.columns)}`",
                        f"- Record plan filter: `{_selected_record_plan}`",
                        "- Record plan filter applies to preview rows only within the head-window sample.",
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
                f"Showing first `{min(int(preview_total_rows), int(preview_window_limit))}` rows "
                f"of `{int(preview_total_rows)}` accepted records from a `{str(preview_strategy)}` "
                f"preview capped at `{int(preview_window_limit)}` rows. "
                "Record plan filter applies to preview rows only within this head-window."
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
def _(
    contract,
    densegen_baserender_title_text,
    densegen_video_subtitle_text,
    render_record_figure,
    summarize_promoter_sites,
    textwrap,
    workspace_heading,
    workspace_name,
):
    _workspace_title = str(workspace_heading or "").strip()
    if not _workspace_title:
        _workspace_title = densegen_baserender_title_text(workspace_name=str(workspace_name or ""))

    def build_baserender_request(*, record, preview_row):
        return {
            "record": record,
            "core_summary": summarize_promoter_sites(preview_row.get("densegen__parts_detail")),
            "plan_summary": str(preview_row.get("densegen__plan") or "").strip(),
        }

    def build_baserender_figure(request: dict[str, object]):
        record = request["record"]
        core_summary = str(request.get("core_summary") or "")
        plan_summary = str(request.get("plan_summary") or "")
        _contract_style_overrides = dict(getattr(contract, "style_overrides", {}) or {})
        _base_typography_size = float(
            max(
                _contract_style_overrides.get("font_size_seq", 18),
                _contract_style_overrides.get("font_size_label", 18),
                _contract_style_overrides.get("legend_font_size", 18),
            )
        )
        _legend_pad_px = float(_contract_style_overrides.get("legend_pad_px", 20.0))
        _legend_height_px = float(_contract_style_overrides.get("legend_height_px", 70.0))
        _legend_patch_w = float(_contract_style_overrides.get("legend_patch_w", 24.0))
        _legend_patch_h = float(_contract_style_overrides.get("legend_patch_h", 14.0))
        _legend_gap_patch_text = float(_contract_style_overrides.get("legend_gap_patch_text", 7.0))
        _legend_gap_x = float(_contract_style_overrides.get("legend_gap_x", 20.0))
        _legend_vertical_align = float(_contract_style_overrides.get("legend_vertical_align", 0.5))
        _uniform_display_font_size = bool(_contract_style_overrides.get("uniform_display_font_size", False))
        _title_font_size = _base_typography_size
        _record_id = str(getattr(record, "id", "") or "unknown")
        _header_title = _workspace_title
        _header_subtitle = densegen_video_subtitle_text(record_id=_record_id, plan_name=str(plan_summary or ""))
        _header_title_wrapped = textwrap.fill(
            _header_title,
            width=42,
            break_long_words=False,
            break_on_hyphens=False,
        )
        _header_subtitle_wrapped = textwrap.fill(
            _header_subtitle,
            width=76,
            break_long_words=False,
            break_on_hyphens=False,
        )
        _line_count = max(1, len(_header_title_wrapped.splitlines())) + max(
            1,
            len(_header_subtitle_wrapped.splitlines()),
        )
        _style_overrides = dict(_contract_style_overrides)
        _style_overrides.update(
            {
                "dpi": 320,
                "font_size_seq": _base_typography_size,
                "font_size_label": _base_typography_size,
                "legend": True,
                "legend_mode": "bottom",
                "legend_height_px": _legend_height_px,
                "padding_y": 12.0,
                "layout": {"outer_pad_cells": 0.62},
                "sequence": {"to_kmer_gap_cells": 0.38},
                "legend_pad_px": _legend_pad_px,
                "legend_patch_w": _legend_patch_w,
                "legend_patch_h": _legend_patch_h,
                "legend_font_size": _base_typography_size,
                "legend_gap_patch_text": _legend_gap_patch_text,
                "legend_gap_x": _legend_gap_x,
                "legend_vertical_align": _legend_vertical_align,
                "uniform_display_font_size": _uniform_display_font_size,
            }
        )
        _style_overrides["padding_y"] = max(float(_style_overrides.get("padding_y", 12.0)), 8.0 + 4.0 * _line_count)
        _figure = render_record_figure(
            record,
            style_preset=contract.style_preset,
            style_overrides=_style_overrides,
        )
        _figure.patch.set_facecolor("white")
        for _axis in _figure.axes:
            _axis.set_facecolor("white")

        _axis = _figure.axes[0] if _figure.axes else None
        if _axis is None:
            raise RuntimeError("BaseRender preview figure expected one axes for title placement.")

        _figure.text(
            0.5,
            0.985,
            f"{_header_title_wrapped}\n{_header_subtitle_wrapped}",
            transform=_figure.transFigure,
            ha="center",
            va="top",
            fontsize=_title_font_size,
            color="#111827",
            zorder=20.0,
            clip_on=False,
        )
        return _figure

    return build_baserender_figure, build_baserender_request


@app.cell
def _(
    preview_rows_filtered,
    get_active_record_index,
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

    active_preview_row = preview_rows_filtered.iloc[active_row_index]
    active_record_id = str(active_preview_row[record_id_column])
    filtered_n = len(preview_rows_filtered)
    mo.vstack(
        [
            mo.md("### BaseRender preview"),
            mo.hstack([record_plan_filter], justify="start", align="center"),
        ]
    )
    return active_preview_row, active_record_id, active_row_index, filtered_n


@app.cell
def _(
    active_record_id,
    active_row_index,
    filtered_n,
    get_baserender_display_payload,
    mo,
    next_record_button,
    prev_record_button,
    record_index_jump,
    record_index_slider,
    run_root,
    to_repo_relative_path,
):
    _record_status = mo.md(
        "<div style='text-align:center'>"
        + f"`{active_row_index + 1} / {filtered_n}` | `id: {active_record_id}`"
        + "</div>"
    )
    _slider_row = mo.hstack([record_index_slider], justify="start", align="center")
    _jump_slot = mo.hstack([record_index_jump], justify="center", align="center")
    _center_slot = mo.vstack([_record_status, _jump_slot], align="center", gap=0.2)
    _prev_slot = mo.hstack([prev_record_button], justify="start", align="center")
    _next_slot = mo.hstack([next_record_button], justify="end", align="center")
    _nav_row = mo.hstack(
        [_prev_slot, _center_slot, _next_slot],
        justify="space-between",
        align="center",
        widths=[1, 6, 1],
        wrap=False,
    )
    _display_payload = dict(get_baserender_display_payload() or {})
    _display_caption = str(_display_payload.get("caption") or "").strip()
    if not _display_caption:
        _display_caption = (
            f"DenseGen BaseRender preview for record {active_record_id}. "
            "Annotated TFBS placements and fixed promoter-core elements are highlighted."
        )
    _display_image_bytes = _display_payload.get("image_bytes")
    if _display_image_bytes:
        _baserender_image = mo.image(
            _display_image_bytes,
            alt=_display_caption,
            caption=_display_caption,
            rounded=True,
            style={
                "border-radius": "14px",
                "width": "100%",
                "height": "auto",
                "max-height": "560px",
                "object-fit": "contain",
                "background": "white",
                "display": "block",
                "margin": "0 auto",
            },
        )
    else:
        _baserender_image = mo.md(
            "<div style='min-height:420px; width:100%; border-radius:14px; "
            "border:1px solid #e5e7eb; background:#ffffff;'></div>"
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
            _slider_row,
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
    adapt_records,
    contract,
    preview_rows_filtered,
    record_id_column,
    require,
):
    record_ids = [str(record_id) for record_id in preview_rows_filtered[record_id_column].tolist()]
    adapter_columns = dict(contract.adapter_columns)
    records = adapt_records(
        preview_rows_filtered.to_dict(orient="records"),
        adapter_kind=contract.adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=contract.adapter_policies,
    )
    records_by_id = {record.id: record for record in records}
    require(
        len(records_by_id) != len(records),
        "Preview rows contain duplicate record ids. Resolve id collisions and rerun.",
    )
    missing_ids = [record_id for record_id in record_ids if record_id not in records_by_id]
    require(
        bool(missing_ids),
        "Preview rows are missing records from the selected window: "
        + ", ".join(f"`{record_id}`" for record_id in missing_ids[:8])
        + (" ..." if len(missing_ids) > 8 else ""),
    )
    return records_by_id


@app.cell
def _(
    BytesIO,
    build_baserender_figure,
    lru_cache,
    records_by_id,
):
    import matplotlib.pyplot as plt
    from PIL import Image, ImageChops

    def _trim_white_border(image: Image.Image, *, pad_px: int = 8) -> Image.Image:
        _background = Image.new(image.mode, image.size, (255, 255, 255, 255))
        _difference = ImageChops.difference(image, _background)
        _bbox = _difference.getbbox()
        if _bbox is None:
            return image
        _left = max(0, int(_bbox[0]) - int(pad_px))
        _top = max(0, int(_bbox[1]) - int(pad_px))
        _right = min(int(image.size[0]), int(_bbox[2]) + int(pad_px))
        _bottom = min(int(image.size[1]), int(_bbox[3]) + int(pad_px))
        return image.crop((_left, _top, _right, _bottom))

    @lru_cache(maxsize=64)
    def render_baserender_preview_image(
        record_id: str,
        core_summary: str,
        plan_summary: str,
    ) -> bytes:
        _buffer = BytesIO()
        _figure = build_baserender_figure(
            {
                "record": records_by_id[str(record_id)],
                "core_summary": str(core_summary or ""),
                "plan_summary": str(plan_summary or ""),
            }
        )
        _figure.savefig(
            _buffer,
            format="png",
            dpi=_figure.dpi,
            bbox_inches="tight",
            pad_inches=0.0,
            facecolor="white",
        )
        plt.close(_figure)
        _image = Image.open(BytesIO(_buffer.getvalue())).convert("RGBA")
        _cropped = _trim_white_border(_image, pad_px=8)
        _cropped_buffer = BytesIO()
        _cropped.save(_cropped_buffer, format="PNG")
        return _cropped_buffer.getvalue()

    return render_baserender_preview_image


@app.cell
def _(
    active_baserender_request,
    active_record_id,
    get_baserender_display_payload,
    render_baserender_preview_image,
    set_baserender_display_payload,
):
    _core_summary = str(active_baserender_request.get("core_summary") or "").strip()
    _caption = (
        f"DenseGen BaseRender preview for record {active_record_id}. "
        + (
            _core_summary
            if _core_summary
            else "Annotated TFBS placements and fixed promoter-core elements are highlighted."
        )
    )
    baserender_preview_image_bytes = render_baserender_preview_image(
        str(active_record_id),
        _core_summary,
        str(active_baserender_request.get("plan_summary") or "").strip(),
    )
    _payload = {
        "record_id": str(active_record_id),
        "image_bytes": baserender_preview_image_bytes,
        "caption": _caption,
    }
    if dict(get_baserender_display_payload() or {}) != _payload:
        set_baserender_display_payload(_payload)
    return baserender_preview_image_bytes


@app.cell
def _(active_record_id, records_by_id, require):
    require(active_record_id not in records_by_id, f"Record `{active_record_id}` missing from preview rows.")
    active_record = records_by_id[active_record_id]
    return active_record


@app.cell
def _(active_preview_row, active_record, build_baserender_request):
    active_baserender_request = build_baserender_request(
        record=active_record,
        preview_row=active_preview_row,
    )
    return active_baserender_request


"""
