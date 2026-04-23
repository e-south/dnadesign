"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_docs_ontology_contracts.py

Docs contracts for Cruncher workflow-family ontology and released Snapback route
vocabulary.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from dnadesign.cruncher.snapback.released_models import ReleasedFinalGeometrySource, ReleasedRouteFamily
from dnadesign.cruncher.workspaces.families import workflow_family_descriptors

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]


def _read_package(path: str) -> str:
    return (PACKAGE_ROOT / path).read_text(encoding="utf-8")


def _read_repo(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _documented_family_ids(text: str) -> tuple[str, ...]:
    match = re.search(r"Registered family ids:\s*([^\n]+)", text)
    if match is None:
        raise AssertionError("Missing registered workflow-family id line.")
    return tuple(re.findall(r"`([^`]+)`", match.group(1)))


def test_top_level_docs_match_registered_workflow_family_ids() -> None:
    expected = tuple(descriptor.id for descriptor in workflow_family_descriptors())

    assert _documented_family_ids(_read_package("README.md")) == expected
    assert _documented_family_ids(_read_package("docs/README.md")) == expected
    assert _documented_family_ids(_read_package("docs/index.md")) == expected
    assert _documented_family_ids(_read_package("docs/guides/intent_and_lifecycle.md")) == expected


def test_released_snapback_docs_publish_route_and_geometry_literals() -> None:
    combined = "\n".join(
        (
            _read_package("docs/guides/snapback_released_workflow.md"),
            _read_package("docs/reference/cli.md"),
        )
    )

    for route_family in get_args(ReleasedRouteFamily):
        assert route_family in combined
    for final_geometry_source in get_args(ReleasedFinalGeometrySource):
        assert final_geometry_source in combined


def test_snapback_shortening_docs_keep_primary_lane_and_contrast_lane_explicit() -> None:
    status = _read_repo("docs/studies/snapback_shortening_effort/status.md")
    routes = _read_repo("docs/studies/snapback_shortening_effort/routes.md")
    skill = _read_repo(".agents/skills/snapback-hairpin-study/SKILL.md")

    assert "The active execution lane is `released-product Snapback` in `de033`." in status
    assert "### Primary route: released-product Snapback" in routes
    assert "This is the active study lane." in routes
    assert "### Contrast route: YIU boundary check" in routes
    assert "released-product Snapback remains the active shortening lane" in skill
    assert "YIU remains a contrast-only boundary surface" in skill
