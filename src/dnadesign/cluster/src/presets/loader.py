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
    for p in d.glob("*.yaml"):
        try:
            obj = Preset(**yaml.safe_load(p.read_text()))
            out[obj.name] = obj
        except Exception:
            continue
    return out


def load_all() -> dict[str, Preset]:
    """
    Load presets with clear precedence (later wins):
    built-in defaults → built-in siblings → user dir → project dir.
    """
    presets_dir = Path(__file__).resolve().parent
    built_in_defaults = presets_dir / "defaults"
    built_in_siblings = presets_dir  # allows YAMLs placed next to 'defaults/'
    user = Path(os.path.expanduser("~/.dnadesign/cluster/presets"))
    project = Path.cwd() / "cluster" / "presets"
    out: dict[str, Preset] = {}
    # Start with built-ins (defaults then siblings); siblings can override defaults
    out.update(_load_dir(built_in_defaults))
    out.update(_load_dir(built_in_siblings))
    # User overrides built-ins
    out.update(_load_dir(user))
    # Project overrides user
    out.update(_load_dir(project))
    return out
