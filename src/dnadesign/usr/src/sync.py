"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/sync.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SSHRemoteConfig, get_remote
from .diff import DiffSummary, compute_diff, compute_file_diff
from .errors import VerificationError
from .io import append_event
from .remote import RemoteDatasetStat, SSHRemote


@dataclass
class SyncOptions:
    primary_only: bool = False
    skip_snapshots: bool = False
    dry_run: bool = False
    assume_yes: bool = False


def _verify_after_pull(local_dir: Path, summary: DiffSummary) -> None:
    # Recompute local stats and ensure they match remote where known
    from .diff import parquet_stats

    local = parquet_stats(local_dir / "records.parquet")
    remote = summary.primary_remote

    # Prefer SHA check if remote SHA is known
    if remote.sha256 and local.sha256 and remote.sha256 != local.sha256:
        raise VerificationError(
            f"Post-transfer SHA mismatch: local={local.sha256} remote={remote.sha256}"
        )
    # Fallback to size
    if (
        (remote.size is not None)
        and (local.size is not None)
        and remote.size != local.size
    ):
        raise VerificationError(
            f"Post-transfer size mismatch: local={local.size} remote={remote.size}"
        )
    # Optional shape check if remote provided rows/cols
    if (
        (remote.rows is not None)
        and (local.rows is not None)
        and remote.rows != local.rows
    ):
        raise VerificationError(
            f"Post-transfer row count mismatch: local={local.rows} remote={remote.rows}"
        )
    if (
        (remote.cols is not None)
        and (local.cols is not None)
        and remote.cols != local.cols
    ):
        raise VerificationError(
            f"Post-transfer col count mismatch: local={local.cols} remote={remote.cols}"
        )


def _verify_after_push(
    remote: SSHRemote, dataset: str, summary_before: DiffSummary
) -> None:
    # Probe remote again and ensure it now matches local
    after: RemoteDatasetStat = remote.stat_dataset(dataset)
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

    # Verify by SHA if available
    if local.sha256 and remote_now.sha256 and local.sha256 != remote_now.sha256:
        raise VerificationError(
            f"Post-push SHA mismatch: local={local.sha256} remote={remote_now.sha256}"
        )
    if (
        (local.size is not None)
        and (remote_now.size is not None)
        and local.size != remote_now.size
    ):
        raise VerificationError(
            f"Post-push size mismatch: local={local.size} remote={remote_now.size}"
        )


def plan_diff(root: Path, dataset: str, remote_name: str) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    rstat = rmt.stat_dataset(dataset)
    return compute_diff(Path(root) / dataset, rstat, dataset)


def plan_diff_file(
    local_file: Path, remote_name: str, *, remote_path: str
) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    rstat = rmt.stat_file(remote_path)
    return compute_file_diff(local_file, rstat, str(local_file))


def execute_pull(
    root: Path, dataset: str, remote_name: str, opts: SyncOptions
) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)

    summary = plan_diff(root, dataset, remote_name)
    # Transfer only when changes are detected
    if not summary.has_change and summary.primary_remote.exists:
        return summary

    dest = Path(root) / dataset
    rmt.pull_to_local(
        dataset,
        dest,
        primary_only=opts.primary_only,
        skip_snapshots=opts.skip_snapshots,
        dry_run=opts.dry_run,
    )
    if not opts.dry_run:
        _verify_after_pull(dest, summary)
        append_event(
            dest / ".events.log",
            {
                "action": "pull",
                "from": remote_name,
                "sha": summary.primary_remote.sha256,
                "rows": summary.primary_remote.rows,
                "cols": summary.primary_remote.cols,
            },
        )
    return summary


def execute_pull_file(
    local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions
) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    # Prepare synthetic summary-before (for verification thresholds)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path)
    if not before.changes and before.primary_remote.exists:
        return before
    rmt.pull_file(remote_path, local_file, dry_run=opts.dry_run)
    if not opts.dry_run:
        # Verify by SHA/size/rows/cols if present
        after = plan_diff_file(local_file, remote_name, remote_path=remote_path)
        if before.primary_remote.sha256 and after.primary_local.sha256:
            if before.primary_remote.sha256 != after.primary_local.sha256:
                raise VerificationError("Post-transfer SHA mismatch for file.")
        elif before.primary_remote.size and after.primary_local.size:
            if before.primary_remote.size != after.primary_local.size:
                raise VerificationError("Post-transfer size mismatch for file.")
        # log
        append_event(
            local_file.parent / ".events.log",
            {
                "action": "pull_file",
                "from": remote_name,
                "path": str(local_file),
                "sha": after.primary_local.sha256,
                "rows": after.primary_local.rows,
                "cols": after.primary_local.cols,
            },
        )
    return before


def execute_push(
    root: Path, dataset: str, remote_name: str, opts: SyncOptions
) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)

    summary = plan_diff(root, dataset, remote_name)
    # If nothing to change, return
    if not summary.has_change and summary.primary_remote.exists:
        return summary

    src = Path(root) / dataset
    rmt.push_from_local(
        dataset,
        src,
        primary_only=opts.primary_only,
        skip_snapshots=opts.skip_snapshots,
        dry_run=opts.dry_run,
    )
    if not opts.dry_run:
        _verify_after_push(rmt, dataset, summary)
        append_event(
            src / ".events.log",
            {
                "action": "push",
                "to": remote_name,
                "sha": summary.primary_local.sha256,
                "rows": summary.primary_local.rows,
                "cols": summary.primary_local.cols,
            },
        )
    return summary


def execute_push_file(
    local_file: Path, remote_name: str, remote_path: str, opts: SyncOptions
) -> DiffSummary:
    cfg: SSHRemoteConfig = get_remote(remote_name)
    rmt = SSHRemote(cfg)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path)
    if not before.changes and before.primary_remote.exists:
        return before
    rmt.push_file(local_file, remote_path, dry_run=opts.dry_run)
    if not opts.dry_run:
        # Re-stat remote and verify
        after = plan_diff_file(local_file, remote_name, remote_path=remote_path)
        if after.primary_local.sha256 and after.primary_remote.sha256:
            if after.primary_local.sha256 != after.primary_remote.sha256:
                raise VerificationError("Post-push SHA mismatch for file.")
        elif after.primary_local.size and after.primary_remote.size:
            if after.primary_local.size != after.primary_remote.size:
                raise VerificationError("Post-push size mismatch for file.")
        append_event(
            local_file.parent / ".events.log",
            {
                "action": "push_file",
                "to": remote_name,
                "path": str(local_file),
                "sha": after.primary_local.sha256,
                "rows": after.primary_local.rows,
                "cols": after.primary_local.cols,
            },
        )
    return before
