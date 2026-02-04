"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/integrations/meme_suite.py

Lightweight MEME Suite tool resolution for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _find_pixi_root() -> Path | None:
    env_root = os.getenv("PIXI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        if (base / "pixi.toml").exists() or (base / ".pixi").exists():
            return base
    return None


def _find_pixi_tool(tool: str) -> Path | None:
    root = _find_pixi_root()
    if root is None:
        return None
    envs_dir = root / ".pixi" / "envs"
    if not envs_dir.exists():
        return None
    for env_dir in sorted(envs_dir.iterdir()):
        candidate = env_dir / "bin" / tool
        if candidate.exists():
            return candidate
    return None


def resolve_executable(tool: str, *, tool_path: Path | None = None) -> Path | None:
    if tool_path is not None:
        resolved = tool_path.expanduser()
        if resolved.is_dir():
            candidate = resolved / tool
        else:
            candidate = resolved
            if candidate.name != tool:
                raise FileNotFoundError(
                    f"Configured tool_path points to '{candidate.name}', expected '{tool}'. "
                    "Provide a bin directory or the correct executable."
                )
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Configured tool_path does not contain '{tool}': {candidate}")
    env_dir = os.getenv("MEME_BIN")
    if env_dir:
        candidate = Path(env_dir).expanduser() / tool
        if candidate.exists():
            return candidate
    pixi_candidate = _find_pixi_tool(tool)
    if pixi_candidate is not None:
        return pixi_candidate
    found = shutil.which(tool)
    return Path(found) if found else None


def require_executable(tool: str, *, tool_path: Path | None = None) -> Path:
    exe = resolve_executable(tool, tool_path=tool_path)
    if exe is None:
        raise FileNotFoundError(
            f"{tool} executable not found. Install MEME Suite and ensure `{tool}` is on PATH, "
            "or set MEME_BIN to the MEME bin directory (pixi users: `pixi run dense ...`)."
        )
    return exe
