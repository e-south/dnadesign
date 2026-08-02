"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sync/__init__.py

Dataset sync operations and verification flow for USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..contracts import VerificationError
from ..events.append import event_log_lock
from ..storage.locking import dataset_write_lock
from .remote import execution as sync_execution
from .remote.config import get_remote
from .remote.diff import (
    DiffSummary,
    FileStat,
    compute_diff,
    event_content_changed,
    parquet_stats,
    resolve_verify_mode,
    verify_primary_match,
)
from .remote.remote import RemoteDatasetStat, SSHRemote
from .remote.sidecars import ensure_sidecar_verify_compatible
from .remote.transfer import capture_validated_event_log_content_revision


@dataclass
class SyncOptions:
    primary_only: bool = False
    skip_snapshots: bool = False
    dry_run: bool = False
    assume_yes: bool = False
    verify: str = "auto"
    verify_sidecars: bool = False
    verify_derived_hashes: bool = False


def _ensure_sidecar_verify_compatible(opts: SyncOptions) -> None:
    ensure_sidecar_verify_compatible(
        verify_sidecars=opts.verify_sidecars,
        verify_derived_hashes=opts.verify_derived_hashes,
        primary_only=opts.primary_only,
        skip_snapshots=opts.skip_snapshots,
    )


def _remote_dataset_lock(remote: SSHRemote, dataset: str):
    lock_fn = getattr(remote, "dataset_transfer_lock", None)
    if lock_fn is None:
        raise VerificationError(f"Remote transport cannot lease the dataset for transfer: {dataset}")
    return lock_fn(dataset)


def _remote_event_log_lock(remote: SSHRemote, dataset: str):
    lock_fn = getattr(remote, "event_log_transfer_lock", None)
    if lock_fn is None:
        raise VerificationError(f"Remote transport cannot lease the event log for full push: {dataset}")
    return lock_fn(dataset)


def _remote_event_log_revision(remote: SSHRemote, dataset: str, event_lease: object):
    revision_fn = getattr(remote, "event_log_revision", None)
    if revision_fn is None:
        raise VerificationError(f"Remote transport cannot verify the event log for full push: {dataset}")
    return revision_fn(dataset, event_lease=event_lease)


def _remote_event_log_observation(remote: SSHRemote, dataset: str):
    observe_fn = getattr(remote, "observe_event_log_revision", None)
    if observe_fn is None:
        raise VerificationError(f"Remote transport cannot inspect the event-log content revision: {dataset}")
    return observe_fn(dataset)


def _plan_diff_with_remote(
    remote: SSHRemote,
    root: Path,
    dataset: str,
    *,
    verify: str,
    include_derived_hashes: bool = False,
    include_event_content: bool = True,
) -> tuple[DiffSummary, RemoteDatasetStat]:
    remote_stat = remote.stat_dataset(dataset, verify=verify, include_derived_hashes=include_derived_hashes)
    verify_mode, notes = resolve_verify_mode(verify, remote_stat.primary)
    summary = compute_diff(
        Path(root) / dataset,
        remote_stat,
        dataset,
        verify_mode=verify_mode,
        verify_notes=notes,
    )
    if include_event_content:
        local_event_revision = capture_validated_event_log_content_revision(Path(root) / dataset / ".events.log")
        remote_event_revision = _remote_event_log_observation(remote, dataset)
        content_diff = event_content_changed(local_event_revision, remote_event_revision)
        summary.changes["events_content_diff"] = content_diff
        summary.has_change = bool(summary.has_change or content_diff)
    return summary, remote_stat


def _verify_after_pull(local_dir: Path, summary: DiffSummary) -> None:
    local = parquet_stats(
        local_dir / "records.parquet",
        include_sha=summary.verify_mode == "hash",
        include_parquet=summary.verify_mode == "parquet",
    )
    verify_primary_match(local, summary.primary_remote, summary.verify_mode, context="post-pull")


def _verify_after_push(
    remote: SSHRemote, dataset: str, summary_before: DiffSummary, *, include_derived_hashes: bool = False
) -> RemoteDatasetStat:
    after: RemoteDatasetStat = remote.stat_dataset(
        dataset, verify=summary_before.verify_mode, include_derived_hashes=include_derived_hashes
    )
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


def _runtime() -> sync_execution.SyncRuntime:
    return sync_execution.SyncRuntime(
        get_remote=get_remote,
        remote_cls=SSHRemote,
        ensure_sidecar_verify_compatible=_ensure_sidecar_verify_compatible,
        remote_dataset_lock=_remote_dataset_lock,
        plan_diff_with_remote=_plan_diff_with_remote,
        verify_after_pull=_verify_after_pull,
        verify_after_push=_verify_after_push,
        dataset_write_lock=dataset_write_lock,
        event_log_lock=event_log_lock,
        remote_event_log_lock=_remote_event_log_lock,
        remote_event_log_revision=_remote_event_log_revision,
        remote_event_log_observation=_remote_event_log_observation,
    )


def plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str) -> DiffSummary:
    return sync_execution.plan_diff(root, dataset, remote_name, verify=verify, runtime=_runtime())


def plan_diff_file(local_file: Path, remote_name: str, *, remote_path: str, verify: str) -> DiffSummary:
    return sync_execution.plan_diff_file(
        local_file,
        remote_name,
        remote_path=remote_path,
        verify=verify,
        runtime=_runtime(),
    )


def execute_pull(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    return sync_execution.execute_pull(root, dataset, remote_name, opts, runtime=_runtime())


def execute_pull_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
    return sync_execution.execute_pull_file(local_file, remote_name, remote_path, opts, runtime=_runtime())


def execute_push(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    return sync_execution.execute_push(root, dataset, remote_name, opts, runtime=_runtime())


def execute_push_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
    return sync_execution.execute_push_file(local_file, remote_name, remote_path, opts, runtime=_runtime())


__all__ = [
    "SyncOptions",
    "execute_pull",
    "execute_pull_file",
    "execute_push",
    "execute_push_file",
    "plan_diff",
    "plan_diff_file",
]
