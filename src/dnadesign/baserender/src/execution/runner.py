"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/execution/runner.py

Sequence-rows job orchestration for adapter, pipeline, selection, and output execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

from ..config import (
    ImagesOutputCfg,
    InputEnvelope,
    RenderJobV4,
    VideoOutputCfg,
    load_render_job,
    output_kind,
    render_contract_descriptor,
    resolve_style,
    validate_adapter_output_compatibility,
    validate_adapter_renderer_compatibility,
    validate_output_configuration,
    validate_render_contract_renderer,
)
from ..core import Record, SchemaError, SkipRecord
from ..integrations import adapter_contract, build_adapter, finalize_adapter, required_source_columns
from ..io import iter_json_rows, iter_jsonl_rows, iter_parquet_rows
from ..pipeline import apply_selection, apply_transforms, enforce_selection_policy, load_transforms
from ..reporting import RunReport
from ..runtime import initialize_runtime

_BundlePublication = CreateOnlyDirectoryPublication


def _render_spec(job: RenderJobV4, style) -> dict[str, object]:
    """Return the portable choices needed to identify one render."""

    style_payload = asdict(style)
    style_bytes = json.dumps(style_payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "dnadesign.baserender.render_spec.v1",
        "contract_kind": job.contract.kind,
        "adapter_kind": job.input.adapter.kind,
        "alphabet": str(job.input.alphabet),
        "renderer": job.render.renderer,
        "options": dict(job.render.options),
        "style_sha256": hashlib.sha256(style_bytes).hexdigest(),
    }


def _output_destination(output: ImagesOutputCfg | VideoOutputCfg) -> Path:
    if isinstance(output, ImagesOutputCfg):
        if output.path is not None:
            return output.path
        assert output.dir is not None
        return output.dir
    return output.path


def _prepare_bundle_publication(bundle_root: Path, *, sensitivity: str = "public") -> _BundlePublication:
    try:
        return CreateOnlyDirectoryPublication.prepare(bundle_root, sensitivity=sensitivity)
    except PublicationError as exc:
        raise SchemaError(str(exc).replace("Artifact bundle", "Render bundle")) from exc


def _staged_job(job: RenderJobV4, publication: _BundlePublication) -> RenderJobV4:
    staged_outputs: list[ImagesOutputCfg | VideoOutputCfg] = []
    for output in job.outputs:
        final = _output_destination(output).resolve()
        staging = publication.stage / final.relative_to(publication.final)
        if isinstance(output, ImagesOutputCfg):
            staged_outputs.append(
                replace(
                    output,
                    dir=staging if output.dir is not None else None,
                    path=staging if output.path is not None else None,
                )
            )
        else:
            staged_outputs.append(replace(output, path=staging))
    return replace(job, outputs=tuple(staged_outputs))


def _publish_bundle(publication: _BundlePublication) -> None:
    try:
        publication.publish(required_manifest="manifest.json")
    except PublicationError as exc:
        raise SchemaError(str(exc).replace("Artifact bundle", "Render bundle")) from exc


def _iter_records(job: RenderJobV4, report: RunReport, *, source_content: bytes) -> Iterator[Record]:
    adapter = build_adapter(job.input.adapter, alphabet=job.input.alphabet)
    transforms = load_transforms(job.pipeline.plugins)
    if job.input.kind == "parquet":
        rows: Iterable[dict] = iter_parquet_rows(
            job.input.path,
            columns=required_source_columns(job.input.adapter),
            content=source_content,
        )
    elif job.input.kind == "json":
        rows = iter_json_rows(job.input.path, content=source_content)
    else:
        rows = iter_jsonl_rows(job.input.path, content=source_content)

    for row_index, row in enumerate(rows):
        report.total_rows_seen += 1
        try:
            record = adapter.apply(row, row_index=row_index)
            record = apply_transforms(record, transforms)
            yield record
        except SkipRecord as skip:
            report.note_skip_row(str(skip) or "skip_record")
    finalize_adapter(adapter)


def _validate_input_envelope(
    job: RenderJobV4,
    *,
    source_content: bytes,
    envelope: InputEnvelope | None,
) -> None:
    if envelope is None:
        return
    if job.input.kind not in envelope.accepted_input_kinds:
        allowed = ", ".join(envelope.accepted_input_kinds)
        raise SchemaError(f"contract input kind must be one of: {allowed}")

    record_count = 0
    base_count = 0
    for row in iter_json_rows(job.input.path, content=source_content):
        record_count += 1
        if record_count > envelope.max_records:
            raise SchemaError(f"Render source exceeds the maximum of {envelope.max_records} records")
        value: object = row
        for field in envelope.base_field_path:
            if not isinstance(value, dict) or field not in value:
                value = None
                break
            value = value[field]
        if isinstance(value, str):
            base_count += len(value)
            if base_count > envelope.max_bases:
                raise SchemaError(f"Render source exceeds the maximum of {envelope.max_bases} bases")


def _sample_or_limit_unselected(records: Iterable[Record], job: RenderJobV4) -> Iterable[Record] | list[Record]:
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


def _materialize_before_strict_outputs(
    records: Iterable[Record] | list[Record],
    job: RenderJobV4,
    report: RunReport,
) -> list[Record]:
    materialized = records if isinstance(records, list) else list(records)
    report.yielded_records = len(materialized)
    if (job.run.strict or job.run.fail_on_skips) and report.has_skips():
        raise SchemaError("Run completed with skipped rows/records; strict mode is enabled")
    if not materialized:
        raise SchemaError("No records to render after adapter, transforms, and selection")
    return materialized


def run_render_job(
    job_or_path: RenderJobV4 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    initialize_runtime()
    job = (
        job_or_path
        if isinstance(job_or_path, RenderJobV4)
        else load_render_job(
            job_or_path,
            caller_root=caller_root,
        )
    )

    validate_adapter_renderer_compatibility(job.input, job.render)
    validate_render_contract_renderer(job.contract.kind, job.render.renderer, field="contract.kind")
    validate_output_configuration(job.bundle.path, job.outputs)
    validate_adapter_output_compatibility(job.input.adapter.kind, job.outputs)
    descriptor = render_contract_descriptor(job.contract.kind)
    adapter_descriptor = adapter_contract(job.input.adapter.kind)
    report = RunReport(
        job_name=job.name,
        input_path=str(job.input.path),
        selection_path=str(job.selection.path) if job.selection else None,
    )
    envelope = adapter_descriptor.input_envelope or descriptor.input_envelope
    try:
        if envelope is None:
            report.capture_source_evidence()
        else:
            report.capture_source_evidence(max_bytes=envelope.max_bytes)
    except ValueError as exc:
        raise SchemaError(str(exc)) from exc
    _validate_input_envelope(job, source_content=report.source_content("input"), envelope=envelope)

    style = resolve_style(preset=job.render.style_preset, overrides=job.render.style_overrides)
    report.render_spec = _render_spec(job, style)
    from ..render import Palette

    palette = Palette(style.palette)

    records: Iterable[Record] | list[Record] = _iter_records(
        job,
        report,
        source_content=report.source_content("input"),
    )
    if adapter_descriptor.validation_scope == "document":
        records = list(records)

    if job.selection is not None:
        selected, missing = apply_selection(
            list(records),
            job.selection,
            source_content=report.source_content("selection"),
            max_rows=envelope.max_records if envelope is not None else None,
        )
        report.missing_selection_keys = missing
        enforce_selection_policy(job.selection, missing)
        records = selected
    else:
        records = _sample_or_limit_unselected(records, job)

    records = _materialize_before_strict_outputs(records, job, report)
    report.release_source_content()

    sensitivity = "private" if "private" in {descriptor.sensitivity, adapter_descriptor.sensitivity} else "public"
    publication = _prepare_bundle_publication(job.bundle.path, sensitivity=sensitivity)
    original_job = job
    try:
        job = _staged_job(job, publication)
        img_output = output_kind(job, "images")
        vid_output = output_kind(job, "video")
    except Exception:
        publication.close()
        raise

    try:
        if isinstance(vid_output, VideoOutputCfg):
            from ..outputs import (
                effective_video_frames_per_record,
                planned_video_frame_count,
                write_images,
                write_video,
            )

            materialized = records
            report.yielded_records = len(materialized)
            planned_frame_count = planned_video_frame_count(materialized, output=vid_output)
            effective_frames_per_record = effective_video_frames_per_record(materialized, output=vid_output)
            if isinstance(img_output, ImagesOutputCfg):
                image_kwargs = {
                    "output": img_output,
                    "renderer_name": job.render.renderer,
                    "style": style,
                    "palette": palette,
                }
                if job.render.options:
                    image_kwargs["renderer_options"] = job.render.options
                write_images(materialized, **image_kwargs)
                original_images = output_kind(original_job, "images")
                assert isinstance(original_images, ImagesOutputCfg)
                final_images = _output_destination(original_images).resolve()
                report.outputs["images_path" if original_images.path is not None else "images_dir"] = str(final_images)
            write_video(
                materialized,
                output=vid_output,
                renderer_name=job.render.renderer,
                style=style,
                palette=palette,
            )
            original_video = output_kind(original_job, "video")
            assert isinstance(original_video, VideoOutputCfg)
            report.outputs["video_path"] = str(original_video.path.resolve())
            report.output_metrics["video"] = {
                "record_count": len(materialized),
                "planned_frame_count": planned_frame_count,
                "fps": int(vid_output.fps),
                "frames_per_record": effective_frames_per_record,
            }
        elif isinstance(img_output, ImagesOutputCfg):
            from ..outputs import write_images

            materialized = records
            image_kwargs = {
                "output": img_output,
                "renderer_name": job.render.renderer,
                "style": style,
                "palette": palette,
            }
            if job.render.options:
                image_kwargs["renderer_options"] = job.render.options
            write_images(materialized, **image_kwargs)
            report.yielded_records = len(materialized)
            original_images = output_kind(original_job, "images")
            assert isinstance(original_images, ImagesOutputCfg)
            final_images = _output_destination(original_images).resolve()
            report.outputs["images_path" if original_images.path is not None else "images_dir"] = str(final_images)
        else:
            raise SchemaError("No supported outputs configured")

        report.outputs["bundle_root"] = str(original_job.bundle.path)
        report.outputs["manifest_path"] = str(original_job.bundle.path / "manifest.json")
        report.verify_source_evidence()
        report.write_portable_manifest(
            publication.stage / "manifest.json",
            bundle_root=original_job.bundle.path,
            staging_root=publication.stage,
        )
        _publish_bundle(publication)
    finally:
        publication.close()

    return report
