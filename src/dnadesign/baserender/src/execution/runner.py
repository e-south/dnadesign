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
import stat
import unicodedata
import uuid
from dataclasses import dataclass, replace
from hashlib import sha256
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
    is_directory: bool = False
    parent_descriptor: int | None = None
    publication_staging_name: str | None = None


@dataclass(frozen=True)
class _PublicationLock:
    descriptor: int


@dataclass(frozen=True)
class _PublicationClaim:
    path: Path
    key: str
    exclusive: bool


_LOCK_REGISTRY_ROOT = Path("/var/tmp")


def _case_variant(name: str) -> str | None:
    for index, character in enumerate(name):
        if character.isalpha():
            replacement = character.upper() if character != character.upper() else character.lower()
            if replacement != character:
                return f"{name[:index]}{replacement}{name[index + 1 :]}"
    return None


def _filesystem_is_case_sensitive(path: Path) -> bool:
    existing = path.resolve(strict=False)
    while not existing.exists():
        if existing.parent == existing:
            return False
        existing = existing.parent

    variant_name = _case_variant(existing.name)
    if variant_name is None:
        return False
    variant = existing.with_name(variant_name)
    try:
        return not os.path.samefile(existing, variant)
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _canonical_path_key(path: Path) -> str:
    normalized = unicodedata.normalize("NFC", path.resolve(strict=False).as_posix())
    return normalized if _filesystem_is_case_sensitive(path) else normalized.casefold()


def _registry_name() -> str:
    if not hasattr(os, "getuid"):
        raise SchemaError("Transactional publication locking requires a POSIX runtime")
    return f"dnadesign-baserender-locks-{os.getuid()}"


def _publication_lock_registry_path(*, root: Path = _LOCK_REGISTRY_ROOT) -> Path:
    return root / _registry_name()


def _open_publication_lock_registry(*, root: Path = _LOCK_REGISTRY_ROOT) -> int:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    root_descriptor = os.open(root, directory_flags)
    namespace_name = _registry_name()
    try:
        created = False
        try:
            os.mkdir(namespace_name, mode=0o700, dir_fd=root_descriptor)
            created = True
        except FileExistsError:
            pass
        try:
            descriptor = os.open(namespace_name, directory_flags, dir_fd=root_descriptor)
        except OSError as exc:
            raise SchemaError("Publication lock registry must be a trusted directory") from exc
    finally:
        os.close(root_descriptor)

    registry_stat = os.fstat(descriptor)
    if created:
        os.fchmod(descriptor, 0o700)
        registry_stat = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(registry_stat.st_mode)
        or registry_stat.st_uid != os.getuid()
        or stat.S_IMODE(registry_stat.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise SchemaError("Publication lock registry must be owned by the current user with mode 0700")
    return descriptor


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
        left_key = Path(_canonical_path_key(left))
        for right, _right_is_dir in destinations[index + 1 :]:
            right_key = Path(_canonical_path_key(right))
            if left_key == right_key or left_key in right_key.parents or right_key in left_key.parents:
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
            is_directory=is_directory,
            publication_staging_name=f".{destination.name}.staging-{token}",
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


def _publication_lock_name(key: str) -> str:
    return f"{sha256(key.encode()).hexdigest()}.lock"


def _publication_claims(targets: tuple[_PublicationTarget, ...]) -> tuple[_PublicationClaim, ...]:
    claims: dict[str, _PublicationClaim] = {}
    for target in targets:
        destination = target.final.resolve(strict=False)
        for path, exclusive in [(ancestor, False) for ancestor in destination.parents] + [(destination, True)]:
            key = _canonical_path_key(path)
            existing = claims.get(key)
            claims[key] = _PublicationClaim(
                path=path,
                key=key,
                exclusive=exclusive or (existing.exclusive if existing is not None else False),
            )
    return tuple(claims[key] for key in sorted(claims))


def _reserve_publication_targets(targets: tuple[_PublicationTarget, ...]) -> tuple[_PublicationLock, ...]:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - guarded above on supported runtimes
        raise SchemaError("Transactional publication locking requires POSIX file locks") from exc

    locks: list[_PublicationLock] = []
    registry_descriptor = _open_publication_lock_registry()
    try:
        for claim in _publication_claims(targets):
            lock_name = _publication_lock_name(claim.key)
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_name, flags, 0o600, dir_fd=registry_descriptor)
            try:
                mode = fcntl.LOCK_EX if claim.exclusive else fcntl.LOCK_SH
                fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise SchemaError(
                    f"Batch publication is already in progress for an overlapping path: {claim.path}"
                ) from exc
            except Exception:
                os.close(descriptor)
                raise
            locks.append(_PublicationLock(descriptor=descriptor))
    except Exception:
        _release_publication_locks(tuple(locks))
        raise
    finally:
        os.close(registry_descriptor)
    return tuple(locks)


def _release_publication_locks(locks: tuple[_PublicationLock, ...]) -> None:
    import fcntl

    for lock in reversed(locks):
        try:
            fcntl.flock(lock.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock.descriptor)


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


def _anchor_publication_targets(targets: tuple[_PublicationTarget, ...]) -> tuple[_PublicationTarget, ...]:
    anchored: list[_PublicationTarget] = []
    try:
        for target in targets:
            anchored.append(
                replace(
                    target,
                    parent_descriptor=_open_or_create_directory(target.final.parent),
                )
            )
    except Exception:
        _close_publication_target_anchors(tuple(anchored))
        raise
    return tuple(anchored)


def _close_publication_target_anchors(targets: tuple[_PublicationTarget, ...]) -> None:
    for target in targets:
        if target.parent_descriptor is not None:
            os.close(target.parent_descriptor)


def _target_parent_matches_anchor(target: _PublicationTarget) -> bool:
    assert target.parent_descriptor is not None
    try:
        current = os.stat(target.final.parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    anchored = os.fstat(target.parent_descriptor)
    return stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (
        anchored.st_dev,
        anchored.st_ino,
    )


def _prepare_render_staging(targets: tuple[_PublicationTarget, ...]) -> tuple[tuple[_PublicationTarget, ...], Path]:
    registry_descriptor = _open_publication_lock_registry()
    staging_name = f"staging-{uuid.uuid4().hex}"
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=registry_descriptor)
    finally:
        os.close(registry_descriptor)
    staging_root = _publication_lock_registry_path() / staging_name
    prepared = tuple(
        replace(
            target,
            staging=staging_root
            / (
                f"{index}-{target.final.name}"
                if target.is_directory
                else f"{index}-{target.final.stem}{target.final.suffix}"
            ),
        )
        for index, target in enumerate(targets)
    )
    return prepared, staging_root


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _validate_anchored_publication_conflicts(
    targets: tuple[_PublicationTarget, ...],
    *,
    conflict_policy: str,
) -> None:
    if conflict_policy == "replace":
        return
    conflicts = []
    for target in targets:
        assert target.parent_descriptor is not None
        if _entry_exists_at(target.parent_descriptor, target.final.name):
            conflicts.append(target.final)
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise SchemaError(
            f"Batch output already exists: {rendered}; set run.conflict_policy to 'replace' to replace it"
        )


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


def _materialize_adjacent_staging(target: _PublicationTarget) -> None:
    assert target.parent_descriptor is not None
    assert target.publication_staging_name is not None
    if _entry_exists_at(target.parent_descriptor, target.publication_staging_name):
        raise SchemaError(f"Batch publication staging already exists: {target.publication_staging_name}")
    if target.is_directory:
        if not target.staging.is_dir() or target.staging.is_symlink():
            raise SchemaError(f"Batch output staging is incomplete: {target.staging}")
        _copy_directory_to_directory(target.staging, target.parent_descriptor, target.publication_staging_name)
    else:
        if not target.staging.is_file() or target.staging.is_symlink():
            raise SchemaError(f"Batch output staging is incomplete: {target.staging}")
        _copy_file_to_directory(target.staging, target.parent_descriptor, target.publication_staging_name)


def _publish_batch(targets: tuple[_PublicationTarget, ...], *, conflict_policy: str) -> None:
    missing = [target.staging for target in targets if not target.staging.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise SchemaError(f"Batch output staging is incomplete: {rendered}")

    token = uuid.uuid4().hex
    backups: list[tuple[_PublicationTarget, str]] = []
    published: list[_PublicationTarget] = []
    try:
        for target in targets:
            if not _target_parent_matches_anchor(target):
                raise SchemaError(f"Batch output parent changed during publication: {target.final.parent}")
            _materialize_adjacent_staging(target)
        if conflict_policy == "replace":
            for target in targets:
                assert target.parent_descriptor is not None
                if not _entry_exists_at(target.parent_descriptor, target.final.name):
                    continue
                backup_name = f".{target.final.name}.backup-{token}"
                os.replace(
                    target.final.name,
                    backup_name,
                    src_dir_fd=target.parent_descriptor,
                    dst_dir_fd=target.parent_descriptor,
                )
                backups.append((target, backup_name))
        for target in targets:
            assert target.parent_descriptor is not None
            assert target.publication_staging_name is not None
            os.replace(
                target.publication_staging_name,
                target.final.name,
                src_dir_fd=target.parent_descriptor,
                dst_dir_fd=target.parent_descriptor,
            )
            published.append(target)
    except Exception:
        for target in reversed(published):
            assert target.parent_descriptor is not None
            _remove_entry_at(target.parent_descriptor, target.final.name)
        for target, backup_name in backups:
            assert target.parent_descriptor is not None
            if _entry_exists_at(target.parent_descriptor, backup_name):
                os.replace(
                    backup_name,
                    target.final.name,
                    src_dir_fd=target.parent_descriptor,
                    dst_dir_fd=target.parent_descriptor,
                )
        for target in targets:
            if target.parent_descriptor is not None and target.publication_staging_name is not None:
                _remove_entry_at(target.parent_descriptor, target.publication_staging_name)
        raise
    else:
        for target, backup_name in backups:
            assert target.parent_descriptor is not None
            _remove_entry_at(target.parent_descriptor, backup_name)


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
    staging_root: Path | None = None
    try:
        targets = _anchor_publication_targets(targets)
        _validate_anchored_publication_conflicts(targets, conflict_policy=job.run.conflict_policy)
        targets, staging_root = _prepare_render_staging(targets)
        original_job = job
        job = _staged_job(job, targets)
        img_output = output_kind(job, "images")
        vid_output = output_kind(job, "video")
    except Exception:
        _close_publication_target_anchors(targets)
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
    finally:
        _cleanup_staging(targets)
        if staging_root is not None:
            _remove_path(staging_root)
        _close_publication_target_anchors(targets)
        _release_publication_locks(publication_locks)

    return report


def run_cruncher_showcase_job(
    job_or_path: RenderJobV3 | str,
    *,
    caller_root: str | Path | None = None,
) -> RunReport:
    # Backward-compatible alias; sequence_rows_v3 is the canonical contract surface.
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)
