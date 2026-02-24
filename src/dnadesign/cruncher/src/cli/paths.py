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


def _leading_parent_segments(relative_path: str) -> int:
    count = 0
    for part in Path(relative_path).parts:
        if part != "..":
            break
        count += 1
    return count


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
        rendered = os.path.relpath(p, base_path)
    except Exception:
        return str(p)
    if _leading_parent_segments(rendered) >= 3:
        return str(p)
    return rendered


def render_paths(paths: list[Path | str]) -> list[str]:
    return [render_path(p) for p in paths]
