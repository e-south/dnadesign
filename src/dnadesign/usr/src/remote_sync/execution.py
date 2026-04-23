"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/remote_sync/execution.py

Remote sync execution orchestration helpers used by the root sync facade.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..errors import VerificationError
from .diff import (
    DiffSummary,
    compute_file_diff,
    file_stats,
    resolve_verify_mode,
    verify_primary_match,
)
from .remote import SSHRemote
from .sidecars import local_sidecar_state, remote_sidecar_state, verify_sidecar_state_match
from .transfer import make_pull_staging_dir, promote_staged_pull


@dataclass(frozen=True)
class SyncRuntime:
    get_remote: Callable[[str], object]
    remote_cls: Callable[[object], SSHRemote]
    ensure_sidecar_verify_compatible: Callable[[object], None]
    remote_dataset_lock: Callable[[SSHRemote, str], object]
    plan_diff_with_remote: Callable[..., tuple[DiffSummary, object]]
    verify_after_pull: Callable[[Path, DiffSummary], None]
    verify_after_push: Callable[..., object]
    event_delta_requires_push: Callable[[Path, int], bool]
    dataset_write_lock: Callable[[Path], object]
    record_event: Callable[..., None]


def _remote_for_name(runtime: SyncRuntime, remote_name: str) -> SSHRemote:
    cfg = runtime.get_remote(remote_name)
    return runtime.remote_cls(cfg)


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
    )
    if not summary.primary_remote.exists:
        raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
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
            )
            if not summary.primary_remote.exists:
                raise VerificationError(f"Refusing pull for dataset '{dataset}': remote records.parquet is missing.")
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
                promote_staged_pull(
                    staged_dir,
                    dest,
                    primary_only=opts.primary_only,
                    skip_snapshots=opts.skip_snapshots,
                )
            finally:
                shutil.rmtree(staged_dir, ignore_errors=True)
            runtime.record_event(
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
        runtime.record_event(
            local_file.parent / ".events.log",
            "pull_file",
            dataset=str(local_file.parent),
            args={"from": remote_name, "path": str(local_file), "verify": before.verify_mode},
            target_path=local_file,
            dataset_root=local_file.parent,
        )
    return before


def execute_push(root: Path, dataset: str, remote_name: str, opts, *, runtime: SyncRuntime) -> DiffSummary:
    remote = _remote_for_name(runtime, remote_name)
    runtime.ensure_sidecar_verify_compatible(opts)

    summary, _ = runtime.plan_diff_with_remote(
        remote,
        root,
        dataset,
        verify=opts.verify,
        include_derived_hashes=opts.verify_derived_hashes,
    )
    if not summary.primary_local.exists:
        raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
    if not summary.has_change and summary.primary_remote.exists:
        src = Path(root) / dataset
        if not runtime.event_delta_requires_push(src / ".events.log", summary.events_remote_lines):
            return summary

    src = Path(root) / dataset
    if opts.dry_run:
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
            summary, _ = runtime.plan_diff_with_remote(
                remote,
                root,
                dataset,
                verify=opts.verify,
                include_derived_hashes=opts.verify_derived_hashes,
            )
            if not summary.primary_local.exists:
                raise VerificationError(f"Refusing push for dataset '{dataset}': local records.parquet is missing.")
            if not summary.has_change and summary.primary_remote.exists:
                if not runtime.event_delta_requires_push(src / ".events.log", summary.events_remote_lines):
                    return summary

            local_sidecars = (
                local_sidecar_state(src, include_derived_hashes=opts.verify_derived_hashes)
                if opts.verify_sidecars
                else None
            )
            remote.push_from_local(
                dataset,
                src,
                primary_only=opts.primary_only,
                skip_snapshots=opts.skip_snapshots,
                dry_run=False,
            )
            remote_after = runtime.verify_after_push(
                remote,
                dataset,
                summary,
                include_derived_hashes=opts.verify_derived_hashes,
            )
            if opts.verify_sidecars and local_sidecars is not None:
                verify_sidecar_state_match(
                    local_sidecars,
                    remote_sidecar_state(remote_after, include_derived_hashes=opts.verify_derived_hashes),
                    context="post-push-sidecars",
                )
            runtime.record_event(
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
        runtime.record_event(
            local_file.parent / ".events.log",
            "push_file",
            dataset=str(local_file.parent),
            args={"to": remote_name, "path": str(local_file), "verify": before.verify_mode},
            target_path=local_file,
            dataset_root=local_file.parent,
        )
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
