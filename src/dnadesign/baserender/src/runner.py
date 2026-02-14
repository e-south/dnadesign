"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/runner.py

Sequence-rows job orchestration for adapter, pipeline, selection, and output execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from .adapters import build_adapter, required_source_columns
from .config import (
    ImagesOutputCfg,
    SequenceRowsJobV3,
    VideoOutputCfg,
    load_sequence_rows_job,
    output_kind,
    resolve_style,
)
from .core import Record, SchemaError, SkipRecord
from .io import iter_parquet_rows
from .pipeline import apply_selection, apply_transforms, enforce_selection_policy, load_transforms
from .reporting import RunReport
from .runtime import initialize_runtime


def _iter_records(job: SequenceRowsJobV3, report: RunReport) -> Iterator[Record]:
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


def _sample_or_limit_unselected(records: Iterable[Record], job: SequenceRowsJobV3) -> Iterable[Record] | list[Record]:
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


def run_sequence_rows_job(
    job_or_path: SequenceRowsJobV3 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    initialize_runtime()
    job = (
        job_or_path
        if isinstance(job_or_path, SequenceRowsJobV3)
        else load_sequence_rows_job(
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


def run_cruncher_showcase_job(
    job_or_path: SequenceRowsJobV3 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    # Backward-compatible alias; sequence_rows_v3 is the canonical contract surface.
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)
