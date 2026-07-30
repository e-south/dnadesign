"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/discovery.py

Bounded filesystem discovery for checked-in OPS metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

_PRUNED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "archived",
        "batch_results",
        "node_modules",
        "outputs",
        "prototypes",
        "runs",
    }
)


def _discover_files(
    *,
    roots: Iterable[Path],
    names: frozenset[str] = frozenset(),
    suffixes: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    if not names and not suffixes:
        return ()

    discovered: set[Path] = set()
    for root in sorted({path.expanduser().resolve() for path in roots}):
        if not root.is_dir():
            continue
        for directory, child_directories, filenames in os.walk(root, topdown=True, followlinks=False):
            child_directories[:] = sorted(
                name
                for name in child_directories
                if name not in _PRUNED_DIRECTORY_NAMES and not (Path(directory) / name).is_symlink()
            )
            for filename in sorted(filenames):
                if filename not in names and not filename.endswith(suffixes):
                    continue
                candidate = Path(directory) / filename
                if candidate.is_file():
                    discovered.add(candidate.resolve())
    return tuple(sorted(discovered))


def discover_named_files(*, roots: Iterable[Path], names: frozenset[str]) -> tuple[Path, ...]:
    """Find exact filenames without entering generated, archived, or cache trees."""

    return _discover_files(roots=roots, names=names)


def discover_suffixed_files(*, roots: Iterable[Path], suffixes: tuple[str, ...]) -> tuple[Path, ...]:
    """Find suffix-matched files without entering generated, archived, or cache trees."""

    return _discover_files(roots=roots, suffixes=suffixes)
