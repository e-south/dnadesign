"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/datasets/catalog.py

Dataset discovery helpers for USR CLI commands.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...dataset import ARCHIVE_DATASET_PREFIX


def list_datasets(root: Path) -> list[str]:
    root = root.resolve()
    if not root.exists():
        return []
    names: set[str] = set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name == ARCHIVE_DATASET_PREFIX:
            continue
        if (path / "records.parquet").exists():
            names.add(path.name)
            continue
        for child in path.iterdir():
            if child.is_dir() and (child / "records.parquet").exists():
                names.add(f"{path.name}/{child.name}")
    return sorted(names)
