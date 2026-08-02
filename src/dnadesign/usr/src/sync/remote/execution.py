"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sync/remote/execution.py

Remote sync execution orchestration helpers used by the root sync facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...contracts import VerificationError
from .diff import (
    DiffSummary,
    compute_file_diff,
    event_content_changed,
    file_stats,
    resolve_verify_mode,
    verify_primary_match,
)
from .remote import SSHRemote
from .sidecars import local_sidecar_state, remote_sidecar_state, verify_sidecar_state_match
from .transfer import (
    EventLogContentRevision,
    capture_validated_event_log_content_revision,
    capture_validated_event_log_revision,
    digest_event_log_prefix,
    make_pull_staging_dir,
    promote_staged_pull,
)


@dataclass(frozen=True)
class SyncRuntime:
    get_remote: Callable[[str], object]
    remote_cls: Callable[[object], SSHRemote]
    ensure_sidecar_verify_compatible: Callable[[object], None]
    remote_dataset_lock: Callable[[SSHRemote, str], object]
    plan_diff_with_remote: Callable[..., tuple[DiffSummary, object]]
    verify_after_pull: Callable[[Path, DiffSummary], None]
    verify_after_push: Callable[..., object]
    dataset_write_lock: Callable[[Path], object]
    event_log_lock: Callable[[Path], object]
    remote_event_log_lock: Callable[[SSHRemote, str], object]
    remote_event_log_revision: Callable[[SSHRemote, str, object], EventLogContentRevision]
    remote_event_log_observation: Callable[[SSHRemote, str], EventLogContentRevision]


def _remote_for_name(runtime: SyncRuntime, remote_name: str) -> SSHRemote:
    cfg = runtime.get_remote(remote_name)
    return runtime.remote_cls(cfg)


def _verify_remote_event_prefix(
    local_event_path: Path,
    local_revision: EventLogContentRevision,
    remote_revision: EventLogContentRevision,
) -> None:
    if not remote_revision.exists:
        return
    if (
        not local_revision.exists
        or remote_revision.size_bytes > local_revision.size_bytes
        or not remote_revision.sha256
    ):
        raise VerificationError("Remote event log is not a prefix of the locked local event log")
    prefix_digest = digest_event_log_prefix(local_event_path, remote_revision.size_bytes)
    if prefix_digest != remote_revision.sha256:
        raise VerificationError("Remote event log is not a prefix of the locked local event log")


def _verify_remote_event_match(
    local_revision: EventLogContentRevision,
    remote_revision: EventLogContentRevision,
) -> None:
    if remote_revision != local_revision:
        raise VerificationError("Remote event log does not match the transferred local event revision")


def _apply_event_content_plan(
    summary: DiffSummary,
    local_revision: EventLogContentRevision,
    remote_revision: EventLogContentRevision,
) -> bool:
    changed = event_content_changed(local_revision, remote_revision)
    summary.changes["events_content_diff"] = changed
    summary.has_change = bool(summary.has_change or changed)
    return changed


def _verify_staged_pull_event_history(
    staged_event_path: Path,
    local_revision: EventLogContentRevision,
) -> None:
    """Require a full pull to preserve the complete local append-only prefix."""

    remote_revision = capture_validated_event_log_content_revision(staged_event_path)
    if not local_revision.exists:
        return
    if (
        not remote_revision.exists
        or remote_revision.size_bytes < local_revision.size_bytes
        or not local_revision.sha256
    ):
        raise VerificationError("Remote event log does not extend the local event history")
    prefix_digest = digest_event_log_prefix(staged_event_path, local_revision.size_bytes)
    if prefix_digest != local_revision.sha256:
        raise VerificationError("Remote event log does not extend the local event history")


def plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str, runtime: SyncRuntime) -> DiffSummary:
    remote = _remote_for_name(runtime, remote_name)
    summary, _ = runtime.plan_diff_with_remote(
        remote,
        root,
        dataset,
        verify=verify,
        include_derived_hashes=False,
    )
    return summary


def plan_diff_file(
    local_file: Path,
    remote_name: str,
    *,
    remote_path: str,
    verify: str,
    runtime: SyncRuntime,
) -> DiffSummary:
    remote = _remote_for_name(runtime, remote_name)
    remote_stat = remote.stat_file(remote_path, verify=verify)
    verify_mode, notes = resolve_verify_mode(verify, remote_stat)
    return compute_file_diff(local_file, remote_stat, str(local_file), verify_mode=verify_mode, verify_notes=notes)


def execute_pull(root: Path, dataset: str, remote_name: str, opts, *, runtime: SyncRuntime) -> DiffSummary:
    remote = _remote_for_name(runtime, remote_name)
    runtime.ensure_sidecar_verify_compatible(opts)

    summary, remote_before = runtime.plan_diff_with_remote(
        remote,
        root,
        dataset,
        verify=opts.verify,
        include_derived_hashes=opts.verify_derived_hashes,
        include_event_content=False,
    )
    if not summary.primary_remote.exists:
        raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
    if not opts.primary_only:
        _apply_event_content_plan(
            summary,
            capture_validated_event_log_content_revision(Path(root) / dataset / ".events.log"),
            runtime.remote_event_log_observation(remote, dataset),
        )
    if not summary.has_change and summary.primary_remote.exists:
        return summary

    dest = Path(root) / dataset
    if opts.dry_run:
        remote.pull_to_local(
            dataset,
            dest,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=True,
        )
        return summary

    with runtime.dataset_write_lock(dest):
        with runtime.remote_dataset_lock(remote, dataset):
            summary, remote_before = runtime.plan_diff_with_remote(
                remote,
                root,
                dataset,
                verify=opts.verify,
                include_derived_hashes=opts.verify_derived_hashes,
                include_event_content=False,
            )
            local_event_revision = EventLogContentRevision(exists=False, size_bytes=0, sha256=None)
            expected_event_revision = None
            if not opts.primary_only:
                with runtime.event_log_lock(dest / ".events.log"):
                    expected_event_revision = capture_validated_event_log_revision(dest / ".events.log")
                    local_event_revision = expected_event_revision.content_revision()
            if not summary.primary_remote.exists:
                raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
            if not opts.primary_only:
                _apply_event_content_plan(
                    summary,
                    local_event_revision,
                    runtime.remote_event_log_observation(remote, dataset),
                )
            if not summary.has_change and summary.primary_remote.exists:
                return summary

            staged_dir = make_pull_staging_dir(root, dataset)
            try:
                remote.pull_to_local(
                    dataset,
                    staged_dir,
                    primary_only=opts.primary_only,
                    skip_snapshots=opts.skip_snapshots,
                    dry_run=False,
                )
                runtime.verify_after_pull(staged_dir, summary)
                if opts.verify_sidecars:
                    verify_sidecar_state_match(
                        local_sidecar_state(staged_dir, include_derived_hashes=opts.verify_derived_hashes),
                        remote_sidecar_state(remote_before, include_derived_hashes=opts.verify_derived_hashes),
                        context="post-pull-sidecars",
                    )
                if not opts.primary_only:
                    _verify_staged_pull_event_history(staged_dir / ".events.log", local_event_revision)
                promote_staged_pull(
                    staged_dir,
                    dest,
                    primary_only=opts.primary_only,
                    skip_snapshots=opts.skip_snapshots,
                    expected_event_revision=expected_event_revision,
                )
            finally:
                shutil.rmtree(staged_dir, ignore_errors=True)
    return summary


def execute_pull_file(
    local_file: Path,
    remote_name: str,
    remote_path: str,
    opts,
    *,
    runtime: SyncRuntime,
) -> DiffSummary:
    if opts.verify_sidecars:
        raise VerificationError("--verify-sidecars is a dataset-only option.")
    remote = _remote_for_name(runtime, remote_name)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path, verify=opts.verify, runtime=runtime)
    if not before.changes and before.primary_remote.exists:
        return before
    remote.pull_file(remote_path, local_file, dry_run=opts.dry_run)
    if not opts.dry_run:
        local_now = file_stats(
            local_file,
            include_sha=before.verify_mode == "hash",
            include_parquet=before.verify_mode == "parquet",
        )
        verify_primary_match(local_now, before.primary_remote, before.verify_mode, context="post-pull-file")
    return before


def execute_push(root: Path, dataset: str, remote_name: str, opts, *, runtime: SyncRuntime) -> DiffSummary:
    remote = _remote_for_name(runtime, remote_name)
    runtime.ensure_sidecar_verify_compatible(opts)
    src = Path(root) / dataset

    summary, _ = runtime.plan_diff_with_remote(
        remote,
        root,
        dataset,
        verify=opts.verify,
        include_derived_hashes=opts.verify_derived_hashes,
        include_event_content=False,
    )
    if not summary.primary_local.exists:
        raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
    observed_local_event_revision = None
    observed_remote_event_revision = None
    if not opts.primary_only:
        observed_local_event_revision = capture_validated_event_log_content_revision(src / ".events.log")
        observed_remote_event_revision = runtime.remote_event_log_observation(remote, dataset)
        _apply_event_content_plan(
            summary,
            observed_local_event_revision,
            observed_remote_event_revision,
        )
    if not summary.has_change and summary.primary_remote.exists and opts.primary_only:
        return summary

    if opts.dry_run:
        if not opts.primary_only:
            assert observed_local_event_revision is not None
            assert observed_remote_event_revision is not None
            _verify_remote_event_prefix(
                src / ".events.log",
                observed_local_event_revision,
                observed_remote_event_revision,
            )
            summary.verify_notes.append(
                "Dry-run observed a valid remote event-log prefix without mutation; an actual push rechecks it "
                "under transaction locks."
            )
        remote.push_from_local(
            dataset,
            src,
            primary_only=opts.primary_only,
            skip_snapshots=opts.skip_snapshots,
            dry_run=True,
        )
        return summary

    with runtime.dataset_write_lock(src):
        with runtime.remote_dataset_lock(remote, dataset):
            # Full-transfer lock order is local dataset, remote dataset, local
            # event log, then remote event log. The remote lease spans the
            # definitive plan, prefix proof, copy, and post-copy verification.
            local_event_lock = nullcontext() if opts.primary_only else runtime.event_log_lock(src / ".events.log")
            with local_event_lock:
                remote_event_lock = (
                    nullcontext(None) if opts.primary_only else runtime.remote_event_log_lock(remote, dataset)
                )
                with remote_event_lock as event_lease:
                    summary, _ = runtime.plan_diff_with_remote(
                        remote,
                        root,
                        dataset,
                        verify=opts.verify,
                        include_derived_hashes=opts.verify_derived_hashes,
                        include_event_content=False,
                    )
                    if not summary.primary_local.exists:
                        raise VerificationError(
                            f"Refusing push for dataset '{dataset}': local records.parquet is missing."
                        )

                    local_event_revision = None
                    if not opts.primary_only:
                        local_event_revision = capture_validated_event_log_content_revision(src / ".events.log")
                        remote_event_revision = runtime.remote_event_log_revision(remote, dataset, event_lease)
                        _verify_remote_event_prefix(
                            src / ".events.log",
                            local_event_revision,
                            remote_event_revision,
                        )
                        _apply_event_content_plan(summary, local_event_revision, remote_event_revision)

                    if not summary.has_change and summary.primary_remote.exists:
                        return summary

                    local_sidecars = (
                        local_sidecar_state(src, include_derived_hashes=opts.verify_derived_hashes)
                        if opts.verify_sidecars
                        else None
                    )
                    if opts.primary_only:
                        remote.push_from_local(
                            dataset,
                            src,
                            primary_only=True,
                            skip_snapshots=opts.skip_snapshots,
                            dry_run=False,
                        )
                    else:
                        remote.push_from_local(
                            dataset,
                            src,
                            primary_only=False,
                            skip_snapshots=opts.skip_snapshots,
                            dry_run=False,
                            event_lease=event_lease,
                        )
                    remote_after = runtime.verify_after_push(
                        remote,
                        dataset,
                        summary,
                        include_derived_hashes=opts.verify_derived_hashes,
                    )
                    if not opts.primary_only and local_event_revision is not None:
                        _verify_remote_event_match(
                            local_event_revision,
                            runtime.remote_event_log_revision(remote, dataset, event_lease),
                        )
                    if opts.verify_sidecars and local_sidecars is not None:
                        verify_sidecar_state_match(
                            local_sidecars,
                            remote_sidecar_state(remote_after, include_derived_hashes=opts.verify_derived_hashes),
                            context="post-push-sidecars",
                        )

    return summary


def execute_push_file(
    local_file: Path,
    remote_name: str,
    remote_path: str,
    opts,
    *,
    runtime: SyncRuntime,
) -> DiffSummary:
    if opts.verify_sidecars:
        raise VerificationError("--verify-sidecars is a dataset-only option.")
    remote = _remote_for_name(runtime, remote_name)
    before = plan_diff_file(local_file, remote_name, remote_path=remote_path, verify=opts.verify, runtime=runtime)
    if not before.changes and before.primary_remote.exists:
        return before
    remote.push_file(local_file, remote_path, dry_run=opts.dry_run)
    if not opts.dry_run:
        after = plan_diff_file(
            local_file, remote_name, remote_path=remote_path, verify=before.verify_mode, runtime=runtime
        )
        verify_primary_match(after.primary_local, after.primary_remote, before.verify_mode, context="post-push-file")
    return before


__all__ = [
    "SyncRuntime",
    "execute_pull",
    "execute_pull_file",
    "execute_push",
    "execute_push_file",
    "plan_diff",
    "plan_diff_file",
]
