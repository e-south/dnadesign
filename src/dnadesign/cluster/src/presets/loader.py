"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/presets/loader.py

Cluster preset loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..layout import builtin_cluster_dir
from .schema import Preset


def _load_dir(d: Path) -> dict[str, Preset]:
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.rglob("*.yaml")):
        obj = Preset(**yaml.safe_load(p.read_text(encoding="utf-8")))
        out[obj.name] = obj
    return out


def load_all() -> dict[str, Preset]:
    """Load built-in presets only; workspace behavior belongs in workspace config.yaml."""
    presets_dir = Path(__file__).resolve().parent
    built_in_defaults = presets_dir / "defaults"
    built_in_siblings = presets_dir
    package_presets = builtin_cluster_dir() / "presets"
    out: dict[str, Preset] = {}
    out.update(_load_dir(built_in_defaults))
    out.update(_load_dir(built_in_siblings))
    out.update(_load_dir(package_presets))
    return out
