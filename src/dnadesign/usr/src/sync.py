"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/sync.py

Dataset sync operations and verification flow for USR remotes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
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
from .sync_sidecars import (
    ensure_sidecar_verify_compatible,
    local_sidecar_state,
    remote_sidecar_state,
    verify_sidecar_state_match,
)
from .sync_transfer import make_pull_staging_dir, promote_staged_pull


@dataclass
class SyncOptions:
    primary_only: bool = False
    skip_snapshots: bool = False
    dry_run: bool = False
    assume_yes: bool = False
    verify: str = "auto"
    verify_sidecars: bool = False
    verify_derived_hashes: bool = False


_SYNC_ONLY_ACTIONS = {"pull", "push", "pull_file", "push_file"}


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
        return nullcontext()
    return lock_fn(dataset)


def _plan_diff_with_remote(
    remote: SSHRemote,
    root: Path,
    dataset: str,
    *,
    verify: str,
    include_derived_hashes: bool = False,
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
    return summary, remote_stat


def _event_delta_requires_push(events_path: Path, *, remote_lines: int) -> bool:
    events_path = Path(events_path)
    if not events_path.exists():
        return False
    start_line = max(0, int(remote_lines))
    with events_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            if index < start_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"Failed to parse local event log line {index + 1}: {events_path}") from exc
            action = str(payload.get("action", "")).strip()
            if action and action not in _SYNC_ONLY_ACTIONS:
                return True
    return False


def _verify_after_pull(local_dir: Path, summary: DiffSummary) -> None:
    from .diff import parquet_stats

    local = parquet_stats(
        local_dir / "records.parquet",
        include_sha=summary.verify_mode == "hash",
        include_parquet=summary.verify_mode == "parquet",
    )
    verify_primary_match(local, summary.primary_remote, summary.verify_mode, context="post-pull")


def _verify_after_push(
    remote: SSHRemote, dataset: str, summary_before: DiffSummary, *, include_derived_hashes: bool = False
) -> RemoteDatasetStat:
    # Probe remote again and ensure it now matches local
    after: RemoteDatasetStat = remote.stat_dataset(
        dataset, verify=summary_before.verify_mode, include_derived_hashes=include_derived_hashes
    )
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
    summary, _ = _plan_diff_with_remote(rmt, root, dataset, verify=verify, include_derived_hashes=False)
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

    summary, remote_before = _plan_diff_with_remote(
        rmt, root, dataset, verify=opts.verify, include_derived_hashes=opts.verify_derived_hashes
    )
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
            summary, remote_before = _plan_diff_with_remote(
                rmt, root, dataset, verify=opts.verify, include_derived_hashes=opts.verify_derived_hashes
            )
            if not summary.primary_remote.exists:
                raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
            if not summary.has_change and summary.primary_remote.exists:
                return summary

            staged_dir = make_pull_staging_dir(root, dataset)
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
                    verify_sidecar_state_match(
                        local_sidecar_state(staged_dir, include_derived_hashes=opts.verify_derived_hashes),
                        remote_sidecar_state(remote_before, include_derived_hashes=opts.verify_derived_hashes),
                        context="post-pull-sidecars",
                    )
                promote_staged_pull(
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
                    "verify_derived_hashes": bool(opts.verify_derived_hashes),
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

    summary, _ = _plan_diff_with_remote(
        rmt, root, dataset, verify=opts.verify, include_derived_hashes=opts.verify_derived_hashes
    )
    if not summary.primary_local.exists:
        raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
    # If nothing to change, return
    if not summary.has_change and summary.primary_remote.exists:
        src = Path(root) / dataset
        if not _event_delta_requires_push(src / ".events.log", remote_lines=summary.events_remote_lines):
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
            summary, _ = _plan_diff_with_remote(
                rmt, root, dataset, verify=opts.verify, include_derived_hashes=opts.verify_derived_hashes
            )
            if not summary.primary_local.exists:
                raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
            if not summary.has_change and summary.primary_remote.exists:
                if not _event_delta_requires_push(src / ".events.log", remote_lines=summary.events_remote_lines):
                    return summary

            local_sidecars = (
                local_sidecar_state(src, include_derived_hashes=opts.verify_derived_hashes)
                if opts.verify_sidecars
                else None
            )
            rmt.push_from_local(
                dataset,
                src,
                primary_only=opts.primary_only,
                skip_snapshots=opts.skip_snapshots,
                dry_run=False,
            )
            remote_after = _verify_after_push(rmt, dataset, summary, include_derived_hashes=opts.verify_derived_hashes)
            if opts.verify_sidecars and local_sidecars is not None:
                verify_sidecar_state_match(
                    local_sidecars,
                    remote_sidecar_state(remote_after, include_derived_hashes=opts.verify_derived_hashes),
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
                    "verify_derived_hashes": bool(opts.verify_derived_hashes),
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
