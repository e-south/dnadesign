"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/api.py

Baserender vNext public API for job execution and record rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from .adapters import build_adapter, required_source_columns
from .config import (
    AdapterCfg,
    SequenceRowsJobV3,
    load_sequence_rows_job_from_mapping,
    resolve_style,
)
from .config import (
    validate_sequence_rows_job as _validate_sequence_rows_job,
)
from .core import Record, SchemaError, ensure
from .io import iter_parquet_rows
from .runner import run_sequence_rows_job as _run_sequence_rows_job
from .runtime import initialize_runtime
from .showcase_style import cruncher_showcase_style_overrides as _cruncher_showcase_style_overrides


def load_record_from_parquet(
    dataset_path: str | Path,
    *,
    record_id: str,
    adapter_kind: str,
    adapter_columns: Mapping[str, object],
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    match_column: str | None = None,
) -> Record:
    initialize_runtime()

    cfg = AdapterCfg(
        kind=str(adapter_kind),
        columns=dict(adapter_columns),
        policies={} if adapter_policies is None else dict(adapter_policies),
    )
    adapter = build_adapter(cfg, alphabet=alphabet)
    source_columns = required_source_columns(cfg)

    if match_column is None:
        raw_match = cfg.columns.get("id")
        ensure(
            raw_match is not None,
            "adapter columns must include 'id' when match_column is not provided",
            SchemaError,
        )
        key_col = str(raw_match)
    else:
        key_col = str(match_column)
        ensure(key_col.strip() != "", "match_column must be non-empty", SchemaError)
        if key_col not in source_columns:
            source_columns = [*source_columns, key_col]

    target_id = str(record_id)
    for row_index, row in enumerate(iter_parquet_rows(dataset_path, columns=source_columns)):
        if str(row.get(key_col)) != target_id:
            continue
        return adapter.apply(row, row_index=row_index)

    raise SchemaError(f"Record '{target_id}' not found in dataset by column '{key_col}'")


def render_record_figure(
    record: Record,
    *,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
):
    initialize_runtime()
    style = resolve_style(
        preset=style_preset,
        overrides={} if style_overrides is None else dict(style_overrides),
    )
    from .render import Palette

    palette = Palette(style.palette)
    from .render import render_record

    return render_record(record, renderer_name=renderer_name, style=style, palette=palette)


def _figure_rgba(fig):
    import numpy as np

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape((height, width, 4)).copy()


def render_record_grid_figure(
    records: Iterable[Record],
    *,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
    ncols: int = 3,
):
    import matplotlib.pyplot as plt

    initialize_runtime()
    records_list = list(records)
    ensure(len(records_list) > 0, "render_record_grid_figure requires at least one record", SchemaError)
    ensure(isinstance(ncols, int) and ncols >= 1, "ncols must be >= 1", SchemaError)

    panel_images: list[object] = []
    for record in records_list:
        panel = render_record_figure(
            record,
            renderer_name=renderer_name,
            style_preset=style_preset,
            style_overrides=style_overrides,
        )
        panel_images.append(_figure_rgba(panel))
        plt.close(panel)

    max_h = max(image.shape[0] for image in panel_images)
    max_w = max(image.shape[1] for image in panel_images)
    cols = min(ncols, len(panel_images))
    rows = int(math.ceil(len(panel_images) / cols))
    dpi = 120
    fig_w = (cols * max_w) / dpi
    fig_h = (rows * max_h) / dpi
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    flat_axes = list(axes.flat)

    for idx, image in enumerate(panel_images):
        ax = flat_axes[idx]
        ax.imshow(image)
        ax.set_axis_off()

    for ax in flat_axes[len(panel_images) :]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)
    return fig


def render_parquet_record_figure(
    dataset_path: str | Path,
    *,
    record_id: str,
    adapter_kind: str,
    adapter_columns: Mapping[str, object],
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    match_column: str | None = None,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
):
    record = load_record_from_parquet(
        dataset_path,
        record_id=record_id,
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
        match_column=match_column,
    )
    return render_record_figure(
        record,
        renderer_name=renderer_name,
        style_preset=style_preset,
        style_overrides=style_overrides,
    )


def validate_sequence_rows_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> SequenceRowsJobV3:
    return _validate_sequence_rows_job(job_or_path, caller_root=caller_root)


def run_sequence_rows_job(job_or_path: SequenceRowsJobV3 | str, *, caller_root: str | Path | None = None):
    return _run_sequence_rows_job(job_or_path, caller_root=caller_root)


def validate_cruncher_showcase_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> SequenceRowsJobV3:
    # Backward-compatible alias; sequence_rows_v3 is the canonical contract surface.
    return validate_sequence_rows_job(job_or_path, caller_root=caller_root)


def run_cruncher_showcase_job(job_or_path: SequenceRowsJobV3 | str, *, caller_root: str | Path | None = None):
    # Backward-compatible alias; sequence_rows_v3 is the canonical contract surface.
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)


def _check_job_kind(kind: str | None) -> None:
    if kind is None:
        return
    normalized = str(kind).strip().lower()
    ensure(
        normalized in {"sequence_rows_v3", "cruncher_showcase_v3"},
        "kind must be one of: sequence_rows_v3, cruncher_showcase_v3",
        SchemaError,
    )


def validate_job(
    path_or_dict: str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    caller_root: str | Path | None = None,
) -> SequenceRowsJobV3:
    _check_job_kind(kind)
    if isinstance(path_or_dict, Mapping):
        return load_sequence_rows_job_from_mapping(path_or_dict, caller_root=caller_root)
    return validate_sequence_rows_job(path_or_dict, caller_root=caller_root)


def run_job(
    path_or_dict: SequenceRowsJobV3 | str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    strict: bool | None = None,
    caller_root: str | Path | None = None,
):
    _check_job_kind(kind)
    if isinstance(path_or_dict, SequenceRowsJobV3):
        job = path_or_dict
    elif isinstance(path_or_dict, Mapping):
        job = load_sequence_rows_job_from_mapping(path_or_dict, caller_root=caller_root)
    else:
        job = validate_sequence_rows_job(path_or_dict, caller_root=caller_root)

    if strict is not None:
        job = replace(job, run=replace(job.run, strict=bool(strict)))
    return run_sequence_rows_job(job, caller_root=caller_root)


def render(
    record_or_records: Record | Iterable[Record],
    *,
    renderer: str = "sequence_rows",
    style: Mapping[str, object] | None = None,
    grid: Mapping[str, object] | None = None,
):
    if style is None:
        style_preset = None
        style_overrides: Mapping[str, object] | None = None
    else:
        style_preset_raw = style.get("preset") if isinstance(style, Mapping) else None
        style_preset = None if style_preset_raw is None else str(style_preset_raw)
        if isinstance(style, Mapping) and "overrides" in style:
            overrides_raw = style.get("overrides") or {}
            if not isinstance(overrides_raw, Mapping):
                raise SchemaError("style.overrides must be a mapping")
            style_overrides = dict(overrides_raw)
        elif isinstance(style, Mapping):
            style_overrides = {k: v for k, v in style.items() if k not in {"preset", "overrides"}}
        else:
            raise SchemaError("style must be a mapping")

    if isinstance(record_or_records, Record):
        return render_record_figure(
            record_or_records,
            renderer_name=renderer,
            style_preset=style_preset,
            style_overrides=style_overrides,
        )

    ncols = 3
    if grid is not None:
        if not isinstance(grid, Mapping):
            raise SchemaError("grid must be a mapping")
        unknown = sorted(str(k) for k in grid.keys() if str(k) != "ncols")
        if unknown:
            raise SchemaError(f"grid contains unknown keys: {unknown}; allowed keys: ['ncols']")
        if "ncols" in grid:
            ncols = int(grid["ncols"])
    if ncols < 1:
        raise SchemaError("grid.ncols must be >= 1")
    return render_record_grid_figure(
        record_or_records,
        renderer_name=renderer,
        style_preset=style_preset,
        style_overrides=style_overrides,
        ncols=ncols,
    )


def cruncher_showcase_style_overrides() -> Mapping[str, object]:
    return _cruncher_showcase_style_overrides()
