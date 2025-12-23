"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/presets/loader.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .schema import Preset


def _load_dir(d: Path) -> dict[str, Preset]:
    out = {}
    if not d.exists():
        return out
    for p in d.rglob("*.yaml"):
        try:
            obj = Preset(**yaml.safe_load(p.read_text()))
            out[obj.name] = obj
        except Exception:
            continue
    return out


def _find_project_preset_dirs() -> list[Path]:
    """Search for project-level presets deterministically, independent of CWD.
    Returns a list of candidate directories in ascending precedence (earlier = lower).
    """
    dirs: list[Path] = []
    # 1) The package's cluster/presets (works when installed or in editable mode)
    pkg_cluster = Path(__file__).resolve().parents[2] / "presets"  # .../cluster/presets
    if pkg_cluster.exists():
        dirs.append(pkg_cluster)
    # 2) Walk upward from CWD; prefer the *nearest* cluster/presets last
    bases = list(reversed(list(Path.cwd().parents))) + [Path.cwd()]
    seen: set[str] = set()
    for base in bases:
        candidates = []
        candidates.append(base / "cluster" / "presets")
        if base.name == "cluster":
            candidates.append(base / "presets")
        for c in candidates:
            if c.exists():
                key = str(c.resolve())
                if key not in seen:
                    dirs.append(c)
                    seen.add(key)
    return dirs


def load_all() -> dict[str, Preset]:
    """Load presets with clear precedence (later wins):
    built-in defaults → built-in siblings → user dir → package presets → nearest project presets.
    """
    presets_dir = Path(__file__).resolve().parent
    built_in_defaults = presets_dir / "defaults"
    built_in_siblings = presets_dir
    user = Path(os.path.expanduser("~/.dnADESIGN/cluster/presets".lower().replace("dnadesign", "dnadesign")))
    # Assemble precedence
    out: dict[str, Preset] = {}
    out.update(_load_dir(built_in_defaults))
    out.update(_load_dir(built_in_siblings))
    out.update(_load_dir(user))
    for d in _find_project_preset_dirs():
        out.update(_load_dir(d))
    return out
