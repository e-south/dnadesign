"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/api.py

Baserender vNext orchestration API for Job v3 validation and execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .adapters import build_adapter, required_source_columns
from .config import (
    AdapterCfg,
    ImagesOutputCfg,
    JobV3,
    VideoOutputCfg,
    load_job_v3,
    output_kind,
    resolve_style,
    validate_job_v3,
)
from .core import Record, SchemaError, SkipRecord, ensure
from .io import iter_parquet_rows
from .pipeline import apply_selection, apply_transforms, enforce_selection_policy, load_transforms
from .reporting import RunReport
from .runtime import initialize_runtime


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


def _figure_rgba(fig) -> np.ndarray:
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
    initialize_runtime()
    records_list = list(records)
    ensure(len(records_list) > 0, "render_record_grid_figure requires at least one record", SchemaError)
    ensure(isinstance(ncols, int) and ncols >= 1, "ncols must be >= 1", SchemaError)

    panel_images: list[np.ndarray] = []
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


def _iter_records(job: JobV3, report: RunReport) -> Iterator[Record]:
    adapter = build_adapter(job.input.adapter, alphabet=job.input.alphabet)
    transforms = load_transforms(job.pipeline.plugins)
    columns = required_source_columns(job.input.adapter)

    for row_index, row in enumerate(iter_parquet_rows(job.input.path, columns=columns)):
        report.total_rows_seen += 1
        try:
            record = adapter.apply(row, row_index=row_index)
            record = apply_transforms(record, transforms)
            yield record
        except SkipRecord as skip:
            report.note_skip_row(str(skip) or "skip_record")


def _sample_or_limit_unselected(records: Iterable[Record], job: JobV3) -> Iterable[Record] | list[Record]:
    sample = job.input.sample
    if sample is not None:
        if sample.mode == "first_n":
            return islice(records, sample.n)
        import random

        materialized = list(records)
        rng = random.Random(int(sample.seed))
        n = min(sample.n, len(materialized))
        idxs = sorted(rng.sample(range(len(materialized)), n))
        return [materialized[i] for i in idxs]

    if job.input.limit is not None:
        return islice(records, job.input.limit)

    return records


def run_job_v3(job_or_path: JobV3 | str, *, caller_root: str | Path | None = None) -> RunReport:
    initialize_runtime()
    job = (
        job_or_path
        if isinstance(job_or_path, JobV3)
        else load_job_v3(
            job_or_path,
            caller_root=caller_root,
        )
    )

    report = RunReport(
        job_name=job.name,
        input_path=str(job.input.path),
        selection_path=str(job.selection.path) if job.selection else None,
    )

    style = resolve_style(preset=job.render.style_preset, overrides=job.render.style_overrides)
    from .render import Palette

    palette = Palette(style.palette)

    records: Iterable[Record] | list[Record] = _iter_records(job, report)

    if job.selection is not None:
        selected, missing = apply_selection(list(records), job.selection)
        report.missing_selection_keys = missing
        enforce_selection_policy(job.selection, missing)
        records = selected
    else:
        records = _sample_or_limit_unselected(records, job)

    img_output = output_kind(job, "images")
    vid_output = output_kind(job, "video")

    if isinstance(vid_output, VideoOutputCfg):
        from .outputs import write_images, write_video

        materialized = list(records)
        report.yielded_records = len(materialized)
        if not materialized:
            raise SchemaError("No records to render after adapter, transforms, and selection")
        if isinstance(img_output, ImagesOutputCfg):
            out_dir = write_images(
                materialized,
                output=img_output,
                renderer_name=job.render.renderer,
                style=style,
                palette=palette,
            )
            report.outputs["images_dir"] = str(out_dir)
        out_path = write_video(
            materialized,
            output=vid_output,
            renderer_name=job.render.renderer,
            style=style,
            palette=palette,
        )
        report.outputs["video_path"] = str(out_path)
    elif isinstance(img_output, ImagesOutputCfg):
        from .outputs import write_images

        if isinstance(records, list):
            materialized = records
            report.yielded_records = len(materialized)
            if not materialized:
                raise SchemaError("No records to render after adapter, transforms, and selection")
            out_dir = write_images(
                materialized,
                output=img_output,
                renderer_name=job.render.renderer,
                style=style,
                palette=palette,
            )
            report.outputs["images_dir"] = str(out_dir)
        else:
            iterator = iter(records)
            try:
                first = next(iterator)
            except StopIteration as exc:
                raise SchemaError("No records to render after adapter, transforms, and selection") from exc

            emitted = 0

            def _counted_records() -> Iterator[Record]:
                nonlocal emitted
                emitted += 1
                yield first
                for record in iterator:
                    emitted += 1
                    yield record

            out_dir = write_images(
                _counted_records(),
                output=img_output,
                renderer_name=job.render.renderer,
                style=style,
                palette=palette,
            )
            report.outputs["images_dir"] = str(out_dir)
            report.yielded_records = emitted
    else:
        raise SchemaError("No supported outputs configured")

    if job.run.emit_report and job.run.report_path is not None:
        report.write(job.run.report_path)
        report.outputs["report_path"] = str(job.run.report_path)

    if job.run.strict or job.run.fail_on_skips:
        if report.has_skips():
            raise SchemaError("Run completed with skipped rows/records; strict mode is enabled")

    return report


def validate_job(job_or_path: str, *, caller_root: str | Path | None = None) -> JobV3:
    return validate_job_v3(job_or_path, caller_root=caller_root)
