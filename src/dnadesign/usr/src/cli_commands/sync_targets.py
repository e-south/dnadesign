"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/sync_targets.py

Target resolution helpers for USR sync command modes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Callable


def is_file_mode_target(target: str | None) -> bool:
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


def is_dataset_dir_target(target: str | None) -> bool:
    if not target:
        return False
    try:
        path = Path(target)
    except (TypeError, ValueError):
        return False
    if not path.exists() or not path.is_dir():
        return False
    return (path / "records.parquet").exists()


def find_registry_root(path: Path) -> Path | None:
    try:
        cursor = path.resolve()
    except Exception:
        cursor = path
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "registry.yaml").exists():
            return candidate
    return None


def resolve_dataset_dir_target(dataset_dir: Path, root: Path) -> tuple[Path, str]:
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

    registry_root = find_registry_root(dataset_dir)
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


def resolve_remote_path_for_file(local_file: Path, args, *, get_remote: Callable[[str], object]) -> str:
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


def resolve_dataset_id_for_diff_or_pull(
    root: Path,
    dataset: str | None,
    *,
    use_rich: bool,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    normalize_dataset_id: Callable[[str], str],
    sequences_error_type: type[Exception],
) -> str | None:
    if dataset is None:
        return resolve_dataset_name_interactive(root, None, use_rich)
    target = str(dataset)
    if "/" in target:
        try:
            return normalize_dataset_id(target)
        except sequences_error_type as exc:
            raise SystemExit(str(exc)) from None
    return resolve_dataset_name_interactive(root, target, use_rich)
