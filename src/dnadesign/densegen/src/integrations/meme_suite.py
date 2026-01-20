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
    found = shutil.which(tool)
    return Path(found) if found else None
