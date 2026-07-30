"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/execution/runner.py

Sequence-rows job orchestration for adapter, pipeline, selection, and output execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from ..adapters import build_adapter, required_source_columns
from ..config import (
    ImagesOutputCfg,
    RenderJobV3,
    VideoOutputCfg,
    load_sequence_rows_job,
    output_kind,
    resolve_style,
)
from ..core import Record, SchemaError, SkipRecord
from ..io import iter_json_rows, iter_jsonl_rows, iter_parquet_rows
from ..pipeline import apply_selection, apply_transforms, enforce_selection_policy, load_transforms
from ..reporting import RunReport
from ..runtime import initialize_runtime


@dataclass(frozen=True)
class _PublicationTarget:
    final: Path
    staging: Path


def _output_destination(output: ImagesOutputCfg | VideoOutputCfg) -> Path:
    if isinstance(output, ImagesOutputCfg):
        if output.path is not None:
            return output.path
        assert output.dir is not None
        return output.dir
    return output.path


def _publication_targets(job: RenderJobV3) -> tuple[_PublicationTarget, ...]:
    destinations = [
        (_output_destination(output).resolve(), isinstance(output, ImagesOutputCfg) and output.dir is not None)
        for output in job.outputs
    ]
    if job.run.emit_report and job.run.report_path is not None:
        destinations.append((job.run.report_path.resolve(), False))

    for index, (left, _left_is_dir) in enumerate(destinations):
        for right, _right_is_dir in destinations[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise SchemaError(f"Batch output destinations must not overlap: {left} and {right}")

    token = uuid.uuid4().hex
    return tuple(
        _PublicationTarget(
            final=destination,
            staging=(
                destination.parent / f".{destination.name}.staging-{token}"
                if is_directory
                else destination.parent / f".{destination.stem}.staging-{token}{destination.suffix}"
            ),
        )
        for destination, is_directory in destinations
    )


def _validate_publication_conflicts(
    targets: tuple[_PublicationTarget, ...],
    *,
    conflict_policy: str,
) -> None:
    if conflict_policy == "replace":
        return
    conflicts = [target.final for target in targets if target.final.exists() or target.final.is_symlink()]
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise SchemaError(
            f"Batch output already exists: {rendered}; set run.conflict_policy to 'replace' to replace it"
        )


def _publication_lock_path(target: _PublicationTarget) -> Path:
    return target.final.parent / f".{target.final.name}.publication.lock"


def _reserve_publication_targets(targets: tuple[_PublicationTarget, ...]) -> tuple[Path, ...]:
    locks: list[Path] = []
    try:
        for target in sorted(targets, key=lambda item: item.final.as_posix()):
            lock_path = _publication_lock_path(target)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise SchemaError(f"Batch publication is already in progress for: {target.final}") from exc
            try:
                os.write(descriptor, f"pid={os.getpid()}\n".encode())
            finally:
                os.close(descriptor)
            locks.append(lock_path)
    except Exception:
        _release_publication_locks(tuple(locks))
        raise
    return tuple(locks)


def _release_publication_locks(locks: tuple[Path, ...]) -> None:
    for lock_path in reversed(locks):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            continue


def _staged_job(job: RenderJobV3, targets: tuple[_PublicationTarget, ...]) -> RenderJobV3:
    by_final = {target.final: target.staging for target in targets}
    staged_outputs: list[ImagesOutputCfg | VideoOutputCfg] = []
    for output in job.outputs:
        final = _output_destination(output).resolve()
        staging = by_final[final]
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
    staged_report = None
    if job.run.emit_report and job.run.report_path is not None:
        staged_report = by_final[job.run.report_path.resolve()]
    return replace(
        job,
        outputs=tuple(staged_outputs),
        run=replace(job.run, report_path=staged_report),
    )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _cleanup_staging(targets: tuple[_PublicationTarget, ...]) -> None:
    for target in targets:
        _remove_path(target.staging)


def _publish_batch(targets: tuple[_PublicationTarget, ...], *, conflict_policy: str) -> None:
    missing = [target.staging for target in targets if not target.staging.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise SchemaError(f"Batch output staging is incomplete: {rendered}")

    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        if conflict_policy == "replace":
            for target in targets:
                if not (target.final.exists() or target.final.is_symlink()):
                    continue
                backup = target.final.parent / f".{target.final.name}.backup-{token}"
                os.replace(target.final, backup)
                backups[target.final] = backup
        for target in targets:
            os.replace(target.staging, target.final)
            published.append(target.final)
    except Exception:
        for path in reversed(published):
            _remove_path(path)
        for final, backup in backups.items():
            if backup.exists() or backup.is_symlink():
                os.replace(backup, final)
        raise
    else:
        for backup in backups.values():
            _remove_path(backup)


def _iter_records(job: RenderJobV3, report: RunReport) -> Iterator[Record]:
    adapter = build_adapter(job.input.adapter, alphabet=job.input.alphabet)
    transforms = load_transforms(job.pipeline.plugins)
    if job.input.kind == "parquet":
        rows: Iterable[dict] = iter_parquet_rows(job.input.path, columns=required_source_columns(job.input.adapter))
    elif job.input.kind == "json":
        rows = iter_json_rows(job.input.path)
    else:
        rows = iter_jsonl_rows(job.input.path)

    for row_index, row in enumerate(rows):
        report.total_rows_seen += 1
        try:
            record = adapter.apply(row, row_index=row_index)
            record = apply_transforms(record, transforms)
            yield record
        except SkipRecord as skip:
            report.note_skip_row(str(skip) or "skip_record")


def _sample_or_limit_unselected(records: Iterable[Record], job: RenderJobV3) -> Iterable[Record] | list[Record]:
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
    job: RenderJobV3,
    report: RunReport,
) -> list[Record] | Iterable[Record]:
    if not (job.run.strict or job.run.fail_on_skips):
        return records

    materialized = records if isinstance(records, list) else list(records)
    report.yielded_records = len(materialized)
    if report.has_skips():
        raise SchemaError("Run completed with skipped rows/records; strict mode is enabled")
    return materialized


def run_sequence_rows_job(
    job_or_path: RenderJobV3 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    initialize_runtime()
    job = (
        job_or_path
        if isinstance(job_or_path, RenderJobV3)
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
    from ..render import Palette

    palette = Palette(style.palette)

    records: Iterable[Record] | list[Record] = _iter_records(job, report)

    if job.selection is not None:
        selected, missing = apply_selection(list(records), job.selection)
        report.missing_selection_keys = missing
        enforce_selection_policy(job.selection, missing)
        records = selected
    else:
        records = _sample_or_limit_unselected(records, job)

    records = _materialize_before_strict_outputs(records, job, report)

    targets = _publication_targets(job)
    _validate_publication_conflicts(targets, conflict_policy=job.run.conflict_policy)
    publication_locks = _reserve_publication_targets(targets)
    try:
        # Recheck after acquiring the reservation to close the preflight/publication race.
        _validate_publication_conflicts(targets, conflict_policy=job.run.conflict_policy)
        original_job = job
        job = _staged_job(job, targets)
        img_output = output_kind(job, "images")
        vid_output = output_kind(job, "video")
    except Exception:
        _release_publication_locks(publication_locks)
        raise

    try:
        if isinstance(vid_output, VideoOutputCfg):
            from ..outputs import (
                effective_video_frames_per_record,
                planned_video_frame_count,
                write_images,
                write_video,
            )

            materialized = list(records)
            report.yielded_records = len(materialized)
            if not materialized:
                raise SchemaError("No records to render after adapter, transforms, and selection")
            planned_frame_count = planned_video_frame_count(materialized, output=vid_output)
            effective_frames_per_record = effective_video_frames_per_record(materialized, output=vid_output)
            if isinstance(img_output, ImagesOutputCfg):
                write_images(
                    materialized,
                    output=img_output,
                    renderer_name=job.render.renderer,
                    style=style,
                    palette=palette,
                )
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

            if isinstance(records, list):
                materialized = records
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

                materialized = list(_counted_records())
                report.yielded_records = emitted

            if not materialized:
                raise SchemaError("No records to render after adapter, transforms, and selection")
            write_images(
                materialized,
                output=img_output,
                renderer_name=job.render.renderer,
                style=style,
                palette=palette,
            )
            report.yielded_records = len(materialized)
            original_images = output_kind(original_job, "images")
            assert isinstance(original_images, ImagesOutputCfg)
            final_images = _output_destination(original_images).resolve()
            report.outputs["images_path" if original_images.path is not None else "images_dir"] = str(final_images)
        else:
            raise SchemaError("No supported outputs configured")

        if job.run.emit_report and job.run.report_path is not None:
            original_report = original_job.run.report_path
            assert original_report is not None
            report.outputs["report_path"] = str(original_report.resolve())
            report.write(job.run.report_path)

        _publish_batch(targets, conflict_policy=job.run.conflict_policy)
    except Exception:
        _cleanup_staging(targets)
        raise
    finally:
        _release_publication_locks(publication_locks)

    return report


def run_cruncher_showcase_job(
    job_or_path: RenderJobV3 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    # Backward-compatible alias; sequence_rows_v3 is the canonical contract surface.
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)
