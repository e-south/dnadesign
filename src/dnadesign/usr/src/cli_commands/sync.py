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
from pathlib import Path, PurePosixPath
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
from ..ui import render_diff_rich
from .datasets import resolve_dataset_name_interactive


def _print_diff(summary, *, use_rich: bool | None = None) -> None:
    def fmt_sz(size):
        if size is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.0f}PB"

    if use_rich:
        render_diff_rich(summary)
        return
    pl, pr = summary.primary_local, summary.primary_remote
    print(f"Dataset: {summary.dataset}")

    local_line = f"Local  : {pl.sha256 or '?'}  size={fmt_sz(pl.size)}  rows={pl.rows or '?'}  cols={pl.cols or '?'}"
    print(local_line)

    remote_line = f"Remote : {pr.sha256 or '?'}  size={fmt_sz(pr.size)}  rows={pr.rows or '?'}  cols={pr.cols or '?'}"
    print(remote_line)

    eq = "==" if (pl.sha256 and pr.sha256 and pl.sha256 == pr.sha256) else "≠"
    print(f"Primary sha: {pl.sha256 or '?'} {eq} {pr.sha256 or '?'}")
    print(f"meta.md     mtime: {summary.meta_local_mtime or '-'}  →  {summary.meta_remote_mtime or '-'}")
    delta_evt = max(0, summary.events_remote_lines - summary.events_local_lines)
    print(
        ".events.log lines: "
        f"local={summary.events_local_lines}  "
        f"remote={summary.events_remote_lines}  "
        f"(+{delta_evt} on remote)"
    )
    print(f"_snapshots : remote_count={summary.snapshots.count}  newer_than_local={summary.snapshots.newer_than_local}")
    print("Status     :", "CHANGES" if summary.has_change else "up-to-date")
    print("Verify     :", summary.verify_mode)


def _print_verify_notes(summary) -> None:
    for note in summary.verify_notes:
        print(f"WARNING: {note}")


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


def _print_sync_audit(summary, *, action: str, dry_run: bool, verify_sidecars: bool) -> None:
    has_change = bool(getattr(summary, "has_change", False))
    transfer_state = "DRY-RUN" if dry_run else ("TRANSFERRED" if has_change else "NO-OP")
    dataset_name = str(getattr(summary, "dataset", "<unknown>"))
    verify_mode = str(getattr(summary, "verify_mode", "auto"))
    changes = dict(getattr(summary, "changes", {}) or {})
    events_local = int(getattr(summary, "events_local_lines", 0))
    events_remote = int(getattr(summary, "events_remote_lines", 0))
    snapshots = getattr(summary, "snapshots", None)
    snapshot_count = int(getattr(snapshots, "count", 0)) if snapshots is not None else 0
    snapshots_newer = int(getattr(snapshots, "newer_than_local", 0)) if snapshots is not None else 0
    snapshots_changed = bool(changes.get("snapshots_name_diff")) or snapshots_newer > 0
    print(f"{action.upper()} audit: {transfer_state}")
    print(f"Dataset    : {dataset_name}")
    print(f"Verify     : primary={verify_mode} sidecars={'strict' if verify_sidecars else 'off'}")
    print(f"Primary    : {'changed' if changes.get('primary_sha_diff') else 'unchanged'}")
    print(f"meta.md    : {'changed' if changes.get('meta_mtime_diff') else 'unchanged'}")
    print(f".events.log: local={events_local}  remote={events_remote}")
    print(
        "_snapshots : "
        f"{'changed' if snapshots_changed else 'unchanged'}  "
        f"remote_count={snapshot_count}  newer_than_local={snapshots_newer}"
    )


def _opts_from_args(args) -> SyncOptions:
    return SyncOptions(
        primary_only=bool(args.primary_only),
        skip_snapshots=bool(args.skip_snapshots),
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
        verify=str(getattr(args, "verify", "auto")),
        verify_sidecars=bool(getattr(args, "verify_sidecars", False)),
    )


def _is_file_mode_target(target: str | None) -> bool:
    if not target:
        return False
    try:
        path = Path(target)
    except (TypeError, ValueError):
        return False
    if target.endswith(".parquet"):
        return True
    if path.exists():
        return path.is_file()
    return False


def _is_dataset_dir_target(target: str | None) -> bool:
    if not target:
        return False
    try:
        path = Path(target)
    except (TypeError, ValueError):
        return False
    if not path.exists() or not path.is_dir():
        return False
    return (path / "records.parquet").exists()


def _find_registry_root(path: Path) -> Path | None:
    try:
        cursor = path.resolve()
    except Exception:
        cursor = path
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "registry.yaml").exists():
            return candidate
    return None


def _resolve_dataset_dir_target(dataset_dir: Path, root: Path) -> tuple[Path, str]:
    dataset_dir = dataset_dir.resolve()
    root = root.resolve()
    if not (dataset_dir / "records.parquet").exists():
        raise SystemExit(f"Dataset directory path must include records.parquet: {dataset_dir}")

    try:
        rel = dataset_dir.relative_to(root)
    except ValueError:
        rel = None
    if rel is not None and rel.parts:
        return root, rel.as_posix()

    registry_root = _find_registry_root(dataset_dir)
    if registry_root is not None:
        try:
            rel = dataset_dir.relative_to(registry_root)
        except ValueError:
            rel = None
        if rel is not None and rel.parts:
            return registry_root, rel.as_posix()

    raise SystemExit(
        "Dataset directory path is outside --root and no registry.yaml ancestor was found. "
        "Pass --root <usr_dataset_root> or use a dataset id."
    )


def _resolve_remote_path_for_file(local_file: Path, args) -> str:
    if args.remote_path:
        return args.remote_path
    cfg = get_remote(args.remote)
    if not cfg.repo_root:
        raise SystemExit("FILE mode requires remote.repo_root in remotes.yaml or --remote-path.")

    local_root = args.repo_root or cfg.local_repo_root or os.environ.get("DNADESIGN_REPO_ROOT")
    if not local_root:
        raise SystemExit(
            "FILE mode requires a local repo root. Pass --repo-root, set DNADESIGN_REPO_ROOT, or add local_repo_root in remotes.yaml."  # noqa
        )
    try:
        rel = local_file.resolve().relative_to(Path(local_root).resolve())
    except ValueError as exc:
        raise SystemExit(
            f"Cannot compute path relative to local repo root: {local_file} not under {local_root}"
        ) from exc
    return str(PurePosixPath(cfg.repo_root).joinpath(rel.as_posix()))


def _resolve_dataset_id_for_diff_or_pull(root: Path, dataset: str | None, *, use_rich: bool) -> str | None:
    if dataset is None:
        return resolve_dataset_name_interactive(root, None, use_rich)
    target = str(dataset)
    if "/" in target:
        try:
            return normalize_dataset_id(target)
        except SequencesError as exc:
            raise SystemExit(str(exc)) from None
    return resolve_dataset_name_interactive(root, target, use_rich)


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
    if _is_file_mode_target(target):
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        summary = plan_diff_file(local_file, args.remote, remote_path=remote_path, verify=args.verify)
    elif _is_dataset_dir_target(target):
        dataset_root, dataset = _resolve_dataset_dir_target(Path(target), Path(args.root))
        summary = plan_diff(dataset_root, dataset, args.remote, verify=args.verify)
    else:
        ds_name = _resolve_dataset_id_for_diff_or_pull(
            Path(args.root),
            getattr(args, "dataset", None),
            use_rich=bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        summary = plan_diff(args.root, ds_name, args.remote, verify=args.verify)
    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": asdict(summary)})
        return
    _print_verify_notes(summary)
    _print_diff(summary, use_rich=(fmt == "rich"))


def cmd_pull(args) -> None:
    target = args.dataset
    if _is_file_mode_target(target):
        if args.primary_only or args.skip_snapshots or bool(getattr(args, "verify_sidecars", False)):
            raise SystemExit(
                "--primary-only/--skip-snapshots/--verify-sidecars are dataset-only flags (not valid in FILE mode)."
            )
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        summary = plan_diff_file(local_file, args.remote, remote_path=remote_path, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_pull_file(local_file, args.remote, remote_path, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="pull",
            dry_run=bool(args.dry_run),
            verify_sidecars=False,
        )
    elif _is_dataset_dir_target(target):
        dataset_root, dataset = _resolve_dataset_dir_target(Path(target), Path(args.root))
        summary = plan_diff(dataset_root, dataset, args.remote, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_pull(dataset_root, dataset, args.remote, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="pull",
            dry_run=bool(args.dry_run),
            verify_sidecars=bool(getattr(args, "verify_sidecars", False)),
        )
    else:
        root = Path(args.root)
        if _strict_bootstrap_id_enabled(args):
            _enforce_strict_bootstrap_dataset_id(
                root,
                str(getattr(args, "dataset", "")),
                use_rich=bool(getattr(args, "rich", False)),
            )
        ds_name = _resolve_dataset_id_for_diff_or_pull(
            root,
            getattr(args, "dataset", None),
            use_rich=bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        summary = plan_diff(args.root, ds_name, args.remote, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_pull(args.root, ds_name, args.remote, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="pull",
            dry_run=bool(args.dry_run),
            verify_sidecars=bool(getattr(args, "verify_sidecars", False)),
        )
    if not args.dry_run:
        print("Pull complete.")


def cmd_push(args) -> None:
    target = args.dataset
    if _is_file_mode_target(target):
        if args.primary_only or args.skip_snapshots or bool(getattr(args, "verify_sidecars", False)):
            raise SystemExit(
                "--primary-only/--skip-snapshots/--verify-sidecars are dataset-only flags (not valid in FILE mode)."
            )
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        summary = plan_diff_file(local_file, args.remote, remote_path=remote_path, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_push_file(local_file, args.remote, remote_path, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="push",
            dry_run=bool(args.dry_run),
            verify_sidecars=False,
        )
    elif _is_dataset_dir_target(target):
        dataset_root, dataset = _resolve_dataset_dir_target(Path(target), Path(args.root))
        summary = plan_diff(dataset_root, dataset, args.remote, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_push(dataset_root, dataset, args.remote, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="push",
            dry_run=bool(args.dry_run),
            verify_sidecars=bool(getattr(args, "verify_sidecars", False)),
        )
    else:
        ds_name = resolve_dataset_name_interactive(
            args.root,
            getattr(args, "dataset", None),
            bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        summary = plan_diff(args.root, ds_name, args.remote, verify=args.verify)
        _print_verify_notes(summary)
        _print_diff(summary, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(summary, assume_yes=bool(args.yes))
        summary = execute_push(args.root, ds_name, args.remote, _opts_from_args(args))
        _print_sync_audit(
            summary,
            action="push",
            dry_run=bool(args.dry_run),
            verify_sidecars=bool(getattr(args, "verify_sidecars", False)),
        )
    if not args.dry_run:
        print("Push complete.")
