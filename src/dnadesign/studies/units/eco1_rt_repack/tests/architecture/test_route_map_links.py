"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_route_map_links.py

Architecture tests for Eco1 RT repack route-map link integrity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_ROUTE_MAP = Path("docs/studies/eco1_rt_repack/routes/README.md")
_PATH_TOKEN_RE = re.compile(r"`([^`]+)`")


def test_route_map_non_generated_relative_paths_resolve() -> None:
    root = repo_root()
    route_path = root / _ROUTE_MAP
    missing = []

    for value in _iter_relative_route_paths(route_path.read_text(encoding="utf-8")):
        resolved = (route_path.parent / value).resolve()
        if not resolved.exists():
            missing.append(value)

    assert missing == []


def _iter_relative_route_paths(text: str) -> list[str]:
    paths = []
    for match in _PATH_TOKEN_RE.finditer(text):
        value = match.group(1)
        if not value.startswith(("../", "./")):
            continue
        if "workspaces/" in value:
            continue
        paths.append(value)
    return paths
