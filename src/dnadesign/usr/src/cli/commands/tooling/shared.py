"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/tooling/shared.py

Shared dependency contract for USR tooling commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ....dataset import Dataset


@dataclass(frozen=True)
class ToolingDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    resolve_path_anywhere: Callable[[Path], Path]
    create_mock_dataset: Callable[..., int]
    add_demo_columns: Callable[..., int]
    dataset_factory: Callable[[Path, str], Dataset]
