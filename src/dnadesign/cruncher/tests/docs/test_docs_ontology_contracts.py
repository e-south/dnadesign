"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_docs_ontology_contracts.py

Docs contracts for Cruncher workflow-family ontology and released Snapback route
vocabulary.

Module Author(s): Eric J. South
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


def test_architecture_reference_keeps_six_peer_workflow_families_explicit() -> None:
    expected = tuple(descriptor.id for descriptor in workflow_family_descriptors())
    architecture = _read_package("docs/reference/architecture.md")

    assert _documented_family_ids(architecture) == expected
    assert "Cruncher is organized as six peer workflow families" in architecture
    assert "**Study workspaces** use `study list|run|summarize|show`" in architecture
    assert "**Portfolio workspaces** use `portfolio run|show`" in architecture
    assert "#### Study lifecycle" in architecture
    assert "#### Portfolio lifecycle" in architecture
    assert "**study list** -> inspect checked-in study specs" in architecture


def test_study_and_portfolio_docs_keep_peer_family_and_source_run_language_explicit() -> None:
    docs_readme = _read_package("docs/README.md")
    docs_index = _read_package("docs/index.md")
    studies = _read_package("docs/guides/studies.md")

    for content in (docs_readme, docs_index):
        assert "explicit source-family runs" in content
        assert "sample-family artifacts" not in content
        assert "sample-family outputs" not in content

    assert "explicit source-family outputs" in docs_index
    assert "Reuse Sample outputs in YIU" not in docs_index
    assert "`study` is a peer Cruncher workflow family." in studies
    assert "#### Current checked-in example posture" in studies
    assert "The currently shipped study specs in Cruncher are sample-backed" in studies
    assert "does not make `study` a hidden `sample` submode" in studies
    assert "cross-workspace aggregation for experimental handoff across explicit source runs" in studies
    assert "they are still a distinct workflow family rather than a `sample` submode" in studies


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
