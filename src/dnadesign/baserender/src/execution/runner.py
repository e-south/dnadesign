"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/execution/runner.py

Sequence-rows job orchestration for adapter, pipeline, selection, and output execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from ..adapters import build_adapter, required_source_columns
from ..config import (
    ImagesOutputCfg,
    RenderJobV4,
    VideoOutputCfg,
    load_render_job,
    output_kind,
    resolve_style,
)
from ..core import Record, SchemaError, SkipRecord
from ..io import iter_json_rows, iter_jsonl_rows, iter_parquet_rows
from ..pipeline import apply_selection, apply_transforms, enforce_selection_policy, load_transforms
from ..reporting import RunReport
from ..runtime import initialize_runtime


@dataclass(frozen=True)
class _BundlePublication:
    final: Path
    render_stage: Path
    adjacent_stage_name: str
    parent_descriptor: int


def _output_destination(output: ImagesOutputCfg | VideoOutputCfg) -> Path:
    if isinstance(output, ImagesOutputCfg):
        if output.path is not None:
            return output.path
        assert output.dir is not None
        return output.dir
    return output.path


def _open_or_create_directory(path: Path) -> int:
    if not path.is_absolute():
        raise SchemaError(f"Publication output parent must be absolute: {path}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, directory_flags)
    try:
        for part in path.parts[1:]:
            try:
                child_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _target_parent_matches_anchor(publication: _BundlePublication) -> bool:
    try:
        current = os.stat(publication.final.parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    anchored = os.fstat(publication.parent_descriptor)
    return stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (
        anchored.st_dev,
        anchored.st_ino,
    )


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _prepare_bundle_publication(bundle_root: Path) -> _BundlePublication:
    final = bundle_root.resolve(strict=False)
    parent_descriptor = _open_or_create_directory(final.parent)
    if _entry_exists_at(parent_descriptor, final.name):
        os.close(parent_descriptor)
        raise SchemaError(f"Render bundle already exists and is immutable: {final}")
    render_stage = Path(tempfile.mkdtemp(prefix="dnadesign-baserender-stage-"))
    return _BundlePublication(
        final=final,
        render_stage=render_stage,
        adjacent_stage_name=f".{final.name}.staging-{uuid.uuid4().hex}",
        parent_descriptor=parent_descriptor,
    )


def _staged_job(job: RenderJobV4, publication: _BundlePublication) -> RenderJobV4:
    staged_outputs: list[ImagesOutputCfg | VideoOutputCfg] = []
    for output in job.outputs:
        final = _output_destination(output).resolve()
        staging = publication.render_stage / final.relative_to(publication.final)
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


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _remove_entry_at(parent_descriptor: int, name: str) -> None:
    try:
        entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(entry_stat.st_mode):
        shutil.rmtree(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _copy_file_to_directory(source: Path, parent_descriptor: int, name: str) -> None:
    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise SchemaError(f"Batch output staging must contain regular files: {source}")
        destination_descriptor = os.open(
            name,
            destination_flags,
            stat.S_IMODE(source_stat.st_mode),
            dir_fd=parent_descriptor,
        )
        try:
            with os.fdopen(os.dup(source_descriptor), "rb") as source_handle:
                with os.fdopen(os.dup(destination_descriptor), "wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _copy_directory_to_directory(source: Path, parent_descriptor: int, name: str) -> None:
    os.mkdir(name, dir_fd=parent_descriptor)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    destination_descriptor = os.open(name, directory_flags, dir_fd=parent_descriptor)
    try:
        for entry in os.scandir(source):
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise SchemaError(f"Batch output staging must not contain symlinks: {entry_path}")
            if entry.is_dir(follow_symlinks=False):
                _copy_directory_to_directory(entry_path, destination_descriptor, entry.name)
            elif entry.is_file(follow_symlinks=False):
                _copy_file_to_directory(entry_path, destination_descriptor, entry.name)
            else:
                raise SchemaError(f"Batch output staging contains an unsupported entry: {entry_path}")
    finally:
        os.close(destination_descriptor)


def _rename_directory_create_only(parent_descriptor: int, source: str, destination: str) -> None:
    """Atomically install one directory without replacing an existing entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source_bytes, parent_descriptor, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source_bytes, parent_descriptor, destination_bytes, 0x1)
    else:
        raise SchemaError("This platform does not support atomic create-only bundle publication")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SchemaError(f"Render bundle already exists and is immutable: {destination}")
    raise OSError(error, os.strerror(error), destination)


def _publish_bundle(publication: _BundlePublication) -> None:
    manifest = publication.render_stage / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise SchemaError(f"Render bundle staging is incomplete: {manifest}")
    if not _target_parent_matches_anchor(publication):
        raise SchemaError(f"Render bundle parent changed during publication: {publication.final.parent}")
    if _entry_exists_at(publication.parent_descriptor, publication.adjacent_stage_name):
        raise SchemaError(f"Render bundle staging already exists: {publication.adjacent_stage_name}")

    try:
        _copy_directory_to_directory(
            publication.render_stage,
            publication.parent_descriptor,
            publication.adjacent_stage_name,
        )
        if not _target_parent_matches_anchor(publication):
            raise SchemaError(f"Render bundle parent changed during publication: {publication.final.parent}")
        _rename_directory_create_only(
            publication.parent_descriptor,
            publication.adjacent_stage_name,
            publication.final.name,
        )
    except Exception:
        _remove_entry_at(publication.parent_descriptor, publication.adjacent_stage_name)
        raise


def _iter_records(job: RenderJobV4, report: RunReport) -> Iterator[Record]:
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
) -> list[Record] | Iterable[Record]:
    if not (job.run.strict or job.run.fail_on_skips):
        return records

    materialized = records if isinstance(records, list) else list(records)
    report.yielded_records = len(materialized)
    if report.has_skips():
        raise SchemaError("Run completed with skipped rows/records; strict mode is enabled")
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

    publication = _prepare_bundle_publication(job.bundle.path)
    original_job = job
    try:
        job = _staged_job(job, publication)
        img_output = output_kind(job, "images")
        vid_output = output_kind(job, "video")
    except Exception:
        _remove_path(publication.render_stage)
        os.close(publication.parent_descriptor)
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

        report.outputs["bundle_root"] = str(original_job.bundle.path)
        report.outputs["manifest_path"] = str(original_job.bundle.path / "manifest.json")
        report.write(publication.render_stage / "manifest.json")
        _publish_bundle(publication)
    finally:
        _remove_path(publication.render_stage)
        os.close(publication.parent_descriptor)

    return report
