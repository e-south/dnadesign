"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/sync.py

Dataset sync operations and verification flow for USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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


def _verify_after_pull(local_dir: Path, summary: DiffSummary) -> None:
    from .diff import parquet_stats

    local = parquet_stats(
        local_dir / "records.parquet",
        include_sha=summary.verify_mode == "hash",
        include_parquet=summary.verify_mode == "parquet",
    )
    verify_primary_match(local, summary.primary_remote, summary.verify_mode, context="post-pull")


def _verify_after_push(remote: SSHRemote, dataset: str, summary_before: DiffSummary) -> None:
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


def plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    rstat = rmt.stat_dataset(dataset, verify=verify)
    verify_mode, notes = resolve_verify_mode(verify, rstat.primary)
    return compute_diff(Path(root) / dataset, rstat, dataset, verify_mode=verify_mode, verify_notes=notes)


def plan_diff_file(local_file: Path, remote_name: str, *, remote_path: str, verify: str) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    rstat = rmt.stat_file(remote_path, verify=verify)
    verify_mode, notes = resolve_verify_mode(verify, rstat)
    return compute_file_diff(local_file, rstat, str(local_file), verify_mode=verify_mode, verify_notes=notes)


def execute_pull(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)

    summary = plan_diff(root, dataset, remote_name, verify=opts.verify)
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
        rmt.pull_to_local(
            dataset,
            dest,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=False,
        )
        _verify_after_pull(dest, summary)
        record_event(
            dest / ".events.log",
            "pull",
            dataset=dataset,
            args={
                "from": remote_name,
                "verify": summary.verify_mode,
                "rows": summary.primary_remote.rows,
                "cols": summary.primary_remote.cols,
            },
            target_path=dest / "records.parquet",
        )
    return summary


def execute_pull_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
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
        )
    return before


def execute_push(root: Path, dataset: str, remote_name: str, opts: SyncOptions) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)

    summary = plan_diff(root, dataset, remote_name, verify=opts.verify)
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
        rmt.push_from_local(
            dataset,
            src,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=False,
        )
        _verify_after_push(rmt, dataset, summary)
        record_event(
            src / ".events.log",
            "push",
            dataset=dataset,
            args={"to": remote_name, "verify": summary.verify_mode},
            target_path=src / "records.parquet",
        )
    return summary


def execute_push_file(local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions) -> DiffSummary:
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
        )
    return before
