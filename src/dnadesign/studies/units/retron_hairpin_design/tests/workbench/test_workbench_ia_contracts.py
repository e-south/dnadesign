"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/workbench/test_workbench_ia_contracts.py

Tests for retron hairpin workbench information-architecture contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.retron_hairpin_design.review_outputs.contracts.feature_directions import (
    FEATURE_DIRECTION_BY_ROLE,
)

from ..support.paths import repo_root_from


def test_workbench_readmes_have_current_frontmatter_and_body_dates() -> None:
    root = repo_root_from(__file__)
    paths = [
        root / "docs" / "studies" / "retron_hairpin_design" / "workbench" / "README.md",
        root / "docs" / "studies" / "retron_hairpin_design" / "workbench" / "ontology" / "README.md",
        root / "docs" / "studies" / "retron_hairpin_design" / "workbench" / "design_sets" / "README.md",
        root / "docs" / "studies" / "retron_hairpin_design" / "workbench" / "deliverables" / "README.md",
    ]

    for path in paths:
        frontmatter, body = _frontmatter_and_body(path)
        assert frontmatter["study_id"] == "retron_hairpin_design"
        assert str(frontmatter["last_verified"]) == "2026-07-09"
        assert "**Last verified:** 2026-07-09" in body


def test_feature_role_ontology_matches_genbank_handoff_direction_contract() -> None:
    root = repo_root_from(__file__)
    ontology_path = (
        root / "docs" / "studies" / "retron_hairpin_design" / "workbench" / "ontology" / "feature_roles.yaml"
    )
    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    roles = ontology["roles"]

    assert ontology["contract"] == "retron_msd_feature_role_ontology_v1"
    assert set(ontology["role_order"]) == set(FEATURE_DIRECTION_BY_ROLE)
    assert set(roles) == set(FEATURE_DIRECTION_BY_ROLE)
    assert {
        role: entry["annotation_direction_on_reverse_complement_record"] for role, entry in roles.items()
    } == FEATURE_DIRECTION_BY_ROLE
    assert {entry["primitive_family"] for entry in roles.values()} == {
        "cap",
        "flank",
        "foldback",
        "payload",
        "stem_base",
    }


def _frontmatter_and_body(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path.as_posix()
    _, frontmatter, body = text.split("---", 2)
    payload = yaml.safe_load(frontmatter)
    assert isinstance(payload, dict)
    return payload, body
