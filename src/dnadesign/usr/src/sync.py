"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/sync.py

Dataset sync operations and verification flow for USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .config import SSHRemoteConfig, get_remote
from .diff import (
    DiffSummary,
    compute_diff,
    compute_file_diff,
    resolve_verify_mode,
    verify_primary_match,
)
from .errors import VerificationError
from .events import record_event
from .locks import dataset_write_lock
from .remote import RemoteDatasetStat, SSHRemote


@dataclass
class SyncOptions:
    primary_only: bool = False
    skip_snapshots: bool = False
    dry_run: bool = False
    assume_yes: bool = False
    verify: str = "auto"
    verify_sidecars: bool = False


@dataclass(frozen=True)
class SidecarState:
    meta_mtime: str | None
    events_lines: int
    snapshot_names: tuple[str, ...]


_SNAPSHOT_RE = re.compile(r"^records-\d{8}T\d{6,}\.parquet$")


def _make_pull_staging_dir(root: Path, dataset: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "__")
    return Path(tempfile.mkdtemp(prefix=f".usr-pull-{safe_dataset}-", dir=str(root)))


def _copy_file_atomic(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.usr-sync-", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _collect_staged_entries(staged: Path, *, skip_snapshots: bool) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for src_path in sorted(staged.rglob("*")):
        rel = src_path.relative_to(staged)
        if not rel.parts:
            continue
        rel_text = rel.as_posix()
        if rel_text in {"records.parquet", ".usr.lock"}:
            continue
        if skip_snapshots and rel.parts[0] == "_snapshots":
            continue
        if src_path.is_symlink():
            raise VerificationError(f"Staged pull payload contains symlink entry: {rel_text}")
        if src_path.is_dir() or src_path.is_file():
            entries.append((src_path, rel))
            continue
        raise VerificationError(f"Staged pull payload contains unsupported entry type: {rel_text}")
    return entries


def _snapshot_names_from_dir(snapshot_dir: Path) -> tuple[str, ...]:
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return ()
    return tuple(
        sorted([item.name for item in snapshot_dir.iterdir() if item.is_file() and _SNAPSHOT_RE.match(item.name)])
    )


def _local_sidecar_state(dataset_dir: Path) -> SidecarState:
    from .diff import events_tail_count, file_mtime

    dataset_dir = Path(dataset_dir)
    return SidecarState(
        meta_mtime=file_mtime(dataset_dir / "meta.md"),
        events_lines=events_tail_count(dataset_dir / ".events.log"),
        snapshot_names=_snapshot_names_from_dir(dataset_dir / "_snapshots"),
    )


def _remote_sidecar_state(remote_stat: RemoteDatasetStat) -> SidecarState:
    return SidecarState(
        meta_mtime=remote_stat.meta_mtime,
        events_lines=int(remote_stat.events_lines),
        snapshot_names=tuple(sorted(remote_stat.snapshot_names)),
    )


def _verify_sidecar_state_match(local: SidecarState, remote: SidecarState, *, context: str) -> None:
    mismatches: list[str] = []
    if local.meta_mtime != remote.meta_mtime:
        mismatches.append(f"meta.md mtime local={local.meta_mtime or '-'} remote={remote.meta_mtime or '-'}")
    if local.events_lines != remote.events_lines:
        mismatches.append(f".events.log lines local={local.events_lines} remote={remote.events_lines}")
    if local.snapshot_names != remote.snapshot_names:
        mismatches.append(f"_snapshots names local={list(local.snapshot_names)} remote={list(remote.snapshot_names)}")
    if mismatches:
        raise VerificationError(f"{context}: sidecar mismatch; " + "; ".join(mismatches))


def _ensure_sidecar_verify_compatible(opts: SyncOptions) -> None:
    if not opts.verify_sidecars:
        return
    if opts.primary_only or opts.skip_snapshots:
        raise VerificationError(
            "--verify-sidecars requires full dataset transfer (no --primary-only/--skip-snapshots)."
        )


def _remote_dataset_lock(remote: SSHRemote, dataset: str):
    lock_fn = getattr(remote, "dataset_transfer_lock", None)
    if lock_fn is None:
        return nullcontext()
    return lock_fn(dataset)


def _plan_diff_with_remote(
    remote: SSHRemote,
    root: Path,
    dataset: str,
    *,
    verify: str,
) -> tuple[DiffSummary, RemoteDatasetStat]:
    remote_stat = remote.stat_dataset(dataset, verify=verify)
    verify_mode, notes = resolve_verify_mode(verify, remote_stat.primary)
    summary = compute_diff(
        Path(root) / dataset,
        remote_stat,
        dataset,
        verify_mode=verify_mode,
        verify_notes=notes,
    )
    return summary, remote_stat


def _promote_staged_pull(staged: Path, dest: Path, *, primary_only: bool, skip_snapshots: bool) -> None:
    staged = Path(staged)
    dest = Path(dest)
    staged_primary = staged / "records.parquet"
    if not staged_primary.exists():
        raise VerificationError(f"Staged pull payload missing records.parquet: {staged_primary}")
    if staged_primary.is_symlink():
        raise VerificationError(f"Staged pull payload contains symlink entry: {staged_primary.name}")
    if not staged_primary.is_file():
        raise VerificationError(f"Staged pull payload contains unsupported records entry: {staged_primary.name}")

    staged_entries = _collect_staged_entries(staged, skip_snapshots=skip_snapshots)

    dest.mkdir(parents=True, exist_ok=True)
    _copy_file_atomic(staged_primary, dest / "records.parquet")
    if primary_only:
        return

    kept_paths: set[str] = {"records.parquet"}
    for src_path, rel in staged_entries:
        rel_text = rel.as_posix()
        kept_paths.add(rel_text)
        dst_path = dest / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        _copy_file_atomic(src_path, dst_path)

    keep_with_parents: set[str] = {".usr.lock"}
    for rel_text in kept_paths:
        keep_with_parents.add(rel_text)
        parent = Path(rel_text).parent
        while str(parent) != ".":
            keep_with_parents.add(parent.as_posix())
            parent = parent.parent

    for local_path in sorted(dest.rglob("*"), key=lambda p: (len(p.parts), p.as_posix()), reverse=True):
        rel = local_path.relative_to(dest)
        rel_text = rel.as_posix()
        if rel_text in keep_with_parents:
            continue
        if skip_snapshots and rel.parts and rel.parts[0] == "_snapshots":
            continue
        if local_path.is_file() or local_path.is_symlink():
            local_path.unlink()
            continue
        try:
            local_path.rmdir()
        except OSError:
            pass


def _verify_after_pull(local_dir: Path, summary: DiffSummary) -> None:
    from .diff import parquet_stats

    local = parquet_stats(
        local_dir / "records.parquet",
        include_sha=summary.verify_mode == "hash",
        include_parquet=summary.verify_mode == "parquet",
    )
    verify_primary_match(local, summary.primary_remote, summary.verify_mode, context="post-pull")


def _verify_after_push(remote: SSHRemote, dataset: str, summary_before: DiffSummary) -> RemoteDatasetStat:
    # Probe remote again and ensure it now matches local
    after: RemoteDatasetStat = remote.stat_dataset(dataset, verify=summary_before.verify_mode)
    # Create a synthetic summary comparing local (previous local) to new remote
    from .diff import FileStat

    # local side (from before push) is summary_before.primary_local
    local = summary_before.primary_local
    remote_now = FileStat(
        exists=after.primary.exists,
        size=after.primary.size,
        sha256=after.primary.sha256,
        rows=after.primary.rows,
        cols=after.primary.cols,
        mtime=after.primary.mtime,
    )

    verify_primary_match(local, remote_now, summary_before.verify_mode, context="post-push")
    return after


def plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    summary, _ = _plan_diff_with_remote(rmt, root, dataset, verify=verify)
    return summary


def plan_diff_file(local_file: Path, remote_name: str, *, remote_path: str, verify: str) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    rstat = rmt.stat_file(remote_path, verify=verify)
    verify_mode, notes = resolve_verify_mode(verify, rstat)
    return compute_file_diff(local_file, rstat, str(local_file), verify_mode=verify_mode, verify_notes=notes)


def execute_pull(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    _ensure_sidecar_verify_compatible(opts)

    summary, remote_before = _plan_diff_with_remote(rmt, root, dataset, verify=opts.verify)
    if not summary.primary_remote.exists:
        raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
    # Transfer only when changes are detected
    if not summary.has_change and summary.primary_remote.exists:
        return summary

    dest = Path(root) / dataset
    if opts.dry_run:
        rmt.pull_to_local(
            dataset,
            dest,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=True,
        )
        return summary

    with dataset_write_lock(dest):
        with _remote_dataset_lock(rmt, dataset):
            summary, remote_before = _plan_diff_with_remote(rmt, root, dataset, verify=opts.verify)
            if not summary.primary_remote.exists:
                raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
            if not summary.has_change and summary.primary_remote.exists:
                return summary

            staged_dir = _make_pull_staging_dir(root, dataset)
            try:
                rmt.pull_to_local(
                    dataset,
                    staged_dir,
                    primary_only=opts.primary_only,
                    skip_snapshots=opts.skip_snapshots,
                    dry_run=False,
                )
                _verify_after_pull(staged_dir, summary)
                if opts.verify_sidecars:
                    _verify_sidecar_state_match(
                        _local_sidecar_state(staged_dir),
                        _remote_sidecar_state(remote_before),
                        context="post-pull-sidecars",
                    )
                _promote_staged_pull(
                    staged_dir,
                    dest,
                    primary_only=opts.primary_only,
                    skip_snapshots=opts.skip_snapshots,
                )
            finally:
                shutil.rmtree(staged_dir, ignore_errors=True)
            record_event(
                dest / ".events.log",
                "pull",
                dataset=dataset,
                args={
                    "from": remote_name,
                    "verify": summary.verify_mode,
                    "verify_sidecars": bool(opts.verify_sidecars),
                    "rows": summary.primary_remote.rows,
                    "cols": summary.primary_remote.cols,
                },
                target_path=dest / "records.parquet",
                dataset_root=root,
            )
    return summary


def execute_pull_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
    if opts.verify_sidecars:
        raise VerificationError("--verify-sidecars is a dataset-only option.")
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    # Prepare synthetic summary-before (for verification thresholds)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path, verify=opts.verify)
    if not before.changes and before.primary_remote.exists:
        return before
    rmt.pull_file(remote_path, local_file, dry_run=opts.dry_run)
    if not opts.dry_run:
        from .diff import file_stats

        local_now = file_stats(
            local_file,
            include_sha=before.verify_mode == "hash",
            include_parquet=before.verify_mode == "parquet",
        )
        verify_primary_match(local_now, before.primary_remote, before.verify_mode, context="post-pull-file")
        record_event(
            local_file.parent / ".events.log",
            "pull_file",
            dataset=str(local_file.parent),
            args={"from": remote_name, "path": str(local_file), "verify": before.verify_mode},
            target_path=local_file,
            dataset_root=local_file.parent,
        )
    return before


def execute_push(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    _ensure_sidecar_verify_compatible(opts)

    summary, _ = _plan_diff_with_remote(rmt, root, dataset, verify=opts.verify)
    if not summary.primary_local.exists:
        raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
    # If nothing to change, return
    if not summary.has_change and summary.primary_remote.exists:
        return summary

    src = Path(root) / dataset
    if opts.dry_run:
        rmt.push_from_local(
            dataset,
            src,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=True,
        )
        return summary

    with dataset_write_lock(src):
        with _remote_dataset_lock(rmt, dataset):
            summary, _ = _plan_diff_with_remote(rmt, root, dataset, verify=opts.verify)
            if not summary.primary_local.exists:
                raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
            if not summary.has_change and summary.primary_remote.exists:
                return summary

            local_sidecars = _local_sidecar_state(src) if opts.verify_sidecars else None
            rmt.push_from_local(
                dataset,
                src,
                primary_only=opts.primary_only,
                skip_snapshots=opts.skip_snapshots,
                dry_run=False,
            )
            remote_after = _verify_after_push(rmt, dataset, summary)
            if opts.verify_sidecars and local_sidecars is not None:
                _verify_sidecar_state_match(
                    local_sidecars,
                    _remote_sidecar_state(remote_after),
                    context="post-push-sidecars",
                )
            record_event(
                src / ".events.log",
                "push",
                dataset=dataset,
                args={
                    "to": remote_name,
                    "verify": summary.verify_mode,
                    "verify_sidecars": bool(opts.verify_sidecars),
                },
                target_path=src / "records.parquet",
                dataset_root=root,
            )
    return summary


def execute_push_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
    if opts.verify_sidecars:
        raise VerificationError("--verify-sidecars is a dataset-only option.")
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path, verify=opts.verify)
    if not before.changes and before.primary_remote.exists:
        return before
    rmt.push_file(local_file, remote_path, dry_run=opts.dry_run)
    if not opts.dry_run:
        # Re-stat remote and verify
        after = plan_diff_file(local_file, remote_name, remote_path=remote_path, verify=before.verify_mode)
        verify_primary_match(after.primary_local, after.primary_remote, before.verify_mode, context="post-push-file")
        record_event(
            local_file.parent / ".events.log",
            "push_file",
            dataset=str(local_file.parent),
            args={"to": remote_name, "path": str(local_file), "verify": before.verify_mode},
            target_path=local_file,
            dataset_root=local_file.parent,
        )
    return before
