"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/workspaces/contracts.py

Immutable workspace contracts for scoped Permuter configuration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.permuter.src.core.config import ScopeConfig


@dataclass(frozen=True)
class PermuterWorkspace:
    scope_id: str
    root: Path
    config_path: Path
    config: ScopeConfig
