"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync.py

USR CLI remote diff/pull/push command implementations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..config import get_remote
from ..dataset import normalize_dataset_id
from ..errors import SequencesError, UserAbort
from ..sync import (
    SyncOptions,
    execute_pull,
    execute_pull_file,
    execute_push,
    execute_push_file,
    plan_diff,
    plan_diff_file,
)
from . import sync_execution as sync_execution_commands
from . import sync_output as sync_output_commands
from . import sync_targets as sync_targets_commands
from .datasets import resolve_dataset_name_interactive
from .sync_policy import resolve_sync_verify, resolve_verify_derived_hashes, resolve_verify_sidecars


def _print_diff(summary, *, use_rich: bool | None = None) -> None:
    sync_output_commands.print_diff(summary, use_rich=use_rich)


def _print_verify_notes(summary) -> None:
    sync_output_commands.print_verify_notes(summary)


def _confirm_or_abort(summary, *, assume_yes: bool) -> None:
    if not summary.has_change:
        print("Already up to date.")
        return
    if assume_yes:
        return
    print("\nChanges detected. Proceed with overwrite?")
    answer = input("[Enter = Yes / n = No] : ").strip().lower()
    if answer in {"n", "no"}:
        raise UserAbort("User cancelled.")


def _print_sync_audit(
    summary, *, action: str, dry_run: bool, verify_sidecars: bool, verify_derived_hashes: bool
) -> None:
    sync_output_commands.print_sync_audit(
        summary,
        action=action,
        dry_run=dry_run,
        verify_sidecars=verify_sidecars,
        verify_derived_hashes=verify_derived_hashes,
    )


def _opts_from_args(args, *, file_mode: bool) -> SyncOptions:
    verify_sidecars = resolve_verify_sidecars(args, file_mode=file_mode)
    return SyncOptions(
        primary_only=bool(args.primary_only),
        skip_snapshots=bool(args.skip_snapshots),
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
        verify=resolve_sync_verify(args),
        verify_sidecars=verify_sidecars,
        verify_derived_hashes=resolve_verify_derived_hashes(args, file_mode=file_mode, verify_sidecars=verify_sidecars),
    )


def _execution_deps() -> sync_execution_commands.SyncExecutionDeps:
    return sync_execution_commands.SyncExecutionDeps(
        is_dataset_dir_target=_is_dataset_dir_target,
        resolve_dataset_dir_target=_resolve_dataset_dir_target,
        strict_bootstrap_id_enabled=_strict_bootstrap_id_enabled,
        enforce_strict_bootstrap_dataset_id=lambda root, dataset, use_rich: _enforce_strict_bootstrap_dataset_id(
            root, dataset, use_rich=use_rich
        ),
        resolve_dataset_id_for_diff_or_pull=lambda root, dataset, use_rich: _resolve_dataset_id_for_diff_or_pull(
            root, dataset, use_rich=use_rich
        ),
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        opts_from_args=lambda args, file_mode: _opts_from_args(args, file_mode=file_mode),
        plan_diff_file=plan_diff_file,
        plan_diff=plan_diff,
        resolve_remote_path_for_file=_resolve_remote_path_for_file,
        print_verify_notes=_print_verify_notes,
        print_diff=lambda summary, use_rich: _print_diff(summary, use_rich=use_rich),
        confirm_or_abort=lambda summary, assume_yes: _confirm_or_abort(summary, assume_yes=assume_yes),
        print_sync_audit=lambda summary, action, dry_run, verify_sidecars, verify_derived_hashes: _print_sync_audit(
            summary,
            action=action,
            dry_run=dry_run,
            verify_sidecars=verify_sidecars,
            verify_derived_hashes=verify_derived_hashes,
        ),
    )


def _is_file_mode_target(target: str | None) -> bool:
    return sync_targets_commands.is_file_mode_target(target)


def _is_dataset_dir_target(target: str | None) -> bool:
    return sync_targets_commands.is_dataset_dir_target(target)


def _find_registry_root(path: Path) -> Path | None:
    return sync_targets_commands.find_registry_root(path)


def _resolve_dataset_dir_target(dataset_dir: Path, root: Path) -> tuple[Path, str]:
    return sync_targets_commands.resolve_dataset_dir_target(dataset_dir, root)


def _resolve_remote_path_for_file(local_file: Path, args) -> str:
    return sync_targets_commands.resolve_remote_path_for_file(local_file, args, get_remote=get_remote)


def _resolve_dataset_id_for_diff_or_pull(root: Path, dataset: str | None, *, use_rich: bool) -> str | None:
    return sync_targets_commands.resolve_dataset_id_for_diff_or_pull(
        root,
        dataset,
        use_rich=use_rich,
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        normalize_dataset_id=normalize_dataset_id,
        sequences_error_type=SequencesError,
    )


def _env_flag_true(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _strict_bootstrap_id_enabled(args) -> bool:
    return bool(getattr(args, "strict_bootstrap_id", False)) or _env_flag_true("USR_SYNC_STRICT_BOOTSTRAP_ID")


def _enforce_strict_bootstrap_dataset_id(root: Path, dataset: str, *, use_rich: bool) -> None:
    if "/" in dataset:
        return
    try:
        resolved = resolve_dataset_name_interactive(root, dataset, use_rich)
    except SystemExit:
        resolved = None
    if resolved:
        return
    raise SystemExit(
        "Strict bootstrap mode requires a namespace-qualified dataset id (<namespace>/<dataset>) "
        "for pulls when local dataset is missing."
    )


def cmd_diff(
    args,
    *,
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> None:
    fmt = resolve_output_format(args)
    target = args.dataset
    verify = resolve_sync_verify(args)
    if _is_file_mode_target(target):
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        summary = plan_diff_file(local_file, args.remote, remote_path=remote_path, verify=verify)
    elif _is_dataset_dir_target(target):
        dataset_root, dataset = _resolve_dataset_dir_target(Path(target), Path(args.root))
        summary = plan_diff(dataset_root, dataset, args.remote, verify=verify)
    else:
        ds_name = _resolve_dataset_id_for_diff_or_pull(
            Path(args.root),
            getattr(args, "dataset", None),
            use_rich=bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        summary = plan_diff(args.root, ds_name, args.remote, verify=verify)
    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": asdict(summary)})
        return
    _print_verify_notes(summary)
    _print_diff(summary, use_rich=(fmt == "rich"))


def _assert_dataset_only_flags_for_file_mode(args) -> None:
    sync_execution_commands.assert_dataset_only_flags_for_file_mode(args)


def _run_file_sync(args, *, action: str, execute_file: Callable) -> None:
    sync_execution_commands.run_file_sync(
        args,
        action=action,
        execute_file=execute_file,
        deps=_execution_deps(),
    )


def _resolve_pull_dataset_target(args) -> tuple[Path, str] | None:
    return sync_execution_commands.resolve_pull_dataset_target(
        args,
        deps=_execution_deps(),
    )


def _resolve_push_dataset_target(args) -> tuple[Path, str] | None:
    return sync_execution_commands.resolve_push_dataset_target(args, deps=_execution_deps())


def _run_dataset_sync(args, *, action: str, resolve_target: Callable, execute_dataset: Callable) -> None:
    sync_execution_commands.run_dataset_sync(
        args,
        action=action,
        resolve_target=resolve_target,
        execute_dataset=execute_dataset,
        deps=_execution_deps(),
    )


def cmd_pull(args) -> None:
    if _is_file_mode_target(args.dataset):
        _run_file_sync(args, action="pull", execute_file=execute_pull_file)
    else:
        _run_dataset_sync(
            args,
            action="pull",
            resolve_target=_resolve_pull_dataset_target,
            execute_dataset=execute_pull,
        )
    if not args.dry_run:
        print("Pull complete.")


def cmd_push(args) -> None:
    if _is_file_mode_target(args.dataset):
        _run_file_sync(args, action="push", execute_file=execute_push_file)
    else:
        _run_dataset_sync(
            args,
            action="push",
            resolve_target=_resolve_push_dataset_target,
            execute_dataset=execute_push,
        )
    if not args.dry_run:
        print("Push complete.")
