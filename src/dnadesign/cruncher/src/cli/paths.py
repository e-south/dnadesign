"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/paths.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def render_path(path: Path | str | None, *, base: Path | None = None) -> str:
    if path is None:
        return "-"
    p = Path(path).expanduser()
    base_path = (base or Path.cwd()).expanduser().resolve()
    if not p.is_absolute():
        p = (base_path / p).resolve()
    else:
        p = p.resolve()
    try:
        return os.path.relpath(p, base_path)
    except Exception:
        return str(p)


def render_paths(paths: list[Path | str]) -> list[str]:
    return [render_path(p) for p in paths]
