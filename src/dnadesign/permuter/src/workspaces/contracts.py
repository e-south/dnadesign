"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/workspaces/contracts.py

Workspace-scope contracts.

Module Author(s): OpenAI Codex
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
