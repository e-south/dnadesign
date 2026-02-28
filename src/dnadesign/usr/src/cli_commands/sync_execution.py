"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync_execution.py

Execution orchestration helpers for USR sync command workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class SyncExecutionDeps:
    is_dataset_dir_target: Callable[[str | None], bool]
    resolve_dataset_dir_target: Callable[[Path, Path], tuple[Path, str]]
    strict_bootstrap_id_enabled: Callable[[object], bool]
    enforce_strict_bootstrap_dataset_id: Callable[[Path, str, bool], None]
    resolve_dataset_id_for_diff_or_pull: Callable[[Path, str | None, bool], str | None]
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    opts_from_args: Callable[[object, bool], object]
    plan_diff_file: Callable[..., object]
    plan_diff: Callable[..., object]
    resolve_remote_path_for_file: Callable[[Path, object], str]
    print_verify_notes: Callable[[object], None]
    print_diff: Callable[[object, bool], None]
    confirm_or_abort: Callable[[object, bool], None]
    print_sync_audit: Callable[[object, str, bool, bool, bool], None]


@dataclass(frozen=True)
class SyncRunResult:
    summary: object
    verify_sidecars: bool
    verify_derived_hashes: bool


def assert_dataset_only_flags_for_file_mode(args) -> None:
    if (
        args.primary_only
        or args.skip_snapshots
        or bool(getattr(args, "verify_sidecars", False))
        or bool(getattr(args, "no_verify_sidecars", False))
        or bool(getattr(args, "verify_derived_hashes", False))
        or bool(getattr(args, "no_verify_derived_hashes", False))
    ):
        raise SystemExit(
            "--primary-only/--skip-snapshots/--verify-sidecars/--no-verify-sidecars/--verify-derived-hashes/--no-verify-derived-hashes are dataset-only flags (not valid in FILE mode)."  # noqa
        )


def run_file_sync(args, *, action: str, execute_file: Callable, deps: SyncExecutionDeps) -> SyncRunResult:
    assert_dataset_only_flags_for_file_mode(args)
    local_file = Path(args.dataset).resolve()
    if local_file.is_dir():
        raise SystemExit("FILE mode: pass a file path, not a directory.")
    remote_path = deps.resolve_remote_path_for_file(local_file, args)
    opts = deps.opts_from_args(args, True)
    summary = deps.plan_diff_file(local_file, args.remote, remote_path=remote_path, verify=opts.verify)
    deps.print_verify_notes(summary)
    deps.print_diff(summary, bool(getattr(args, "rich", False)))
    deps.confirm_or_abort(summary, bool(args.yes))
    summary = execute_file(local_file, args.remote, remote_path, opts)
    deps.print_sync_audit(summary, action, bool(args.dry_run), False, False)
    return SyncRunResult(summary=summary, verify_sidecars=False, verify_derived_hashes=False)


def resolve_pull_dataset_target(args, *, deps: SyncExecutionDeps) -> tuple[Path, str] | None:
    target = args.dataset
    if deps.is_dataset_dir_target(target):
        return deps.resolve_dataset_dir_target(Path(target), Path(args.root))
    root = Path(args.root)
    if deps.strict_bootstrap_id_enabled(args):
        deps.enforce_strict_bootstrap_dataset_id(
            root,
            str(getattr(args, "dataset", "")),
            bool(getattr(args, "rich", False)),
        )
    ds_name = deps.resolve_dataset_id_for_diff_or_pull(
        root,
        getattr(args, "dataset", None),
        bool(getattr(args, "rich", False)),
    )
    if not ds_name:
        return None
    return root, ds_name


def resolve_push_dataset_target(args, *, deps: SyncExecutionDeps) -> tuple[Path, str] | None:
    target = args.dataset
    if deps.is_dataset_dir_target(target):
        return deps.resolve_dataset_dir_target(Path(target), Path(args.root))
    ds_name = deps.resolve_dataset_name_interactive(
        Path(args.root),
        getattr(args, "dataset", None),
        bool(getattr(args, "rich", False)),
    )
    if not ds_name:
        return None
    return Path(args.root), ds_name


def run_dataset_sync(
    args, *, action: str, resolve_target: Callable, execute_dataset: Callable, deps: SyncExecutionDeps
) -> Optional[SyncRunResult]:
    resolved = resolve_target(args)
    if not resolved:
        return None
    dataset_root, dataset = resolved
    opts = deps.opts_from_args(args, False)
    summary = deps.plan_diff(dataset_root, dataset, args.remote, verify=opts.verify)
    deps.print_verify_notes(summary)
    deps.print_diff(summary, bool(getattr(args, "rich", False)))
    deps.confirm_or_abort(summary, bool(args.yes))
    summary = execute_dataset(dataset_root, dataset, args.remote, opts)
    deps.print_sync_audit(
        summary,
        action,
        bool(args.dry_run),
        opts.verify_sidecars,
        opts.verify_derived_hashes,
    )
    return SyncRunResult(
        summary=summary,
        verify_sidecars=bool(opts.verify_sidecars),
        verify_derived_hashes=bool(opts.verify_derived_hashes),
    )
