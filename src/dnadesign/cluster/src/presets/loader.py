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

from ..layout import builtin_cluster_dir, is_builtin_cluster_path
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


def _find_workspace_preset_dirs() -> list[Path]:
    """Search for workspace-level presets deterministically, independent of CWD.
    Returns a list of candidate directories in ascending precedence (earlier = lower).
    """
    dirs: list[Path] = []
    # Walk upward from CWD; prefer the *nearest* workspace cluster/presets last.
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
                if key not in seen and not is_builtin_cluster_path(c):
                    dirs.append(c)
                    seen.add(key)
    return dirs


def load_all() -> dict[str, Preset]:
    """Load presets with clear precedence (later wins):
    built-in defaults → built-in siblings → user dir → package presets → nearest workspace presets.
    """
    presets_dir = Path(__file__).resolve().parent
    built_in_defaults = presets_dir / "defaults"
    built_in_siblings = presets_dir
    user = Path(os.path.expanduser("~/.dnADESIGN/cluster/presets".lower().replace("dnadesign", "dnadesign")))
    package_presets = builtin_cluster_dir() / "presets"
    # Assemble precedence
    out: dict[str, Preset] = {}
    out.update(_load_dir(built_in_defaults))
    out.update(_load_dir(built_in_siblings))
    out.update(_load_dir(user))
    out.update(_load_dir(package_presets))
    for d in _find_workspace_preset_dirs():
        out.update(_load_dir(d))
    return out
