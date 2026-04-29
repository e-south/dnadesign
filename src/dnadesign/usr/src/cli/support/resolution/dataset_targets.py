"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/support/resolution/dataset_targets.py

CLI dataset-target resolution helpers shared by the USR entrypoint facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer

from ...commands import datasets as dataset_commands


def normalize_dataset_id(dataset: str) -> str:
    return dataset_commands._normalize_dataset_id(dataset)  # noqa: SLF001


def resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    return dataset_commands.resolve_existing_dataset_id(root, dataset)


def resolve_dataset_name_interactive(root: Path, dataset: str | None, use_rich: bool) -> str | None:
    return dataset_commands.resolve_dataset_name_interactive(root, dataset, use_rich)


def is_explicit_path_target(target: str | None) -> bool:
    text = str(target or "").strip()
    if text in {"", ".", "./", "..", "../"}:
        return True
    if text.startswith("./") or text.startswith("../") or text.startswith("~/"):
        return True
    if Path(text).is_absolute():
        return True
    if text.lower().endswith(".parquet"):
        return True
    if "/" in text or "\\" in text:
        return Path(text).expanduser().exists()
    return False


def exit_missing_path_target(target: str) -> None:
    print(f"ERROR: Path target not found: {target}")
    raise typer.Exit(code=4)


def list_datasets(root: Path):
    return dataset_commands.list_datasets(root)


def resolve_dataset_for_read(
    root: Path,
    dataset_arg: str,
    *,
    resolve_dataset_for_read_impl: Callable[..., object],
    resolve_existing_dataset_id_impl: Callable[[Path, str], str],
    normalize_dataset_id_impl: Callable[[str], str],
    pkg_root: Callable[[], Path],
):
    return resolve_dataset_for_read_impl(
        root,
        dataset_arg,
        resolve_existing_dataset_id=resolve_existing_dataset_id_impl,
        normalize_dataset_id=normalize_dataset_id_impl,
        pkg_root=pkg_root(),
    )


__all__ = [
    "exit_missing_path_target",
    "is_explicit_path_target",
    "list_datasets",
    "normalize_dataset_id",
    "resolve_dataset_for_read",
    "resolve_dataset_name_interactive",
    "resolve_existing_dataset_id",
]
