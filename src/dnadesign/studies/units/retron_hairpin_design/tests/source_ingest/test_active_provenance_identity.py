"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/source_ingest/test_active_provenance_identity.py

Active MSD structure provenance identity contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())


def test_active_structure_provenance_uses_only_the_hairpin_owned_identity() -> None:
    repo_root = _repo_root()
    provenance_root = repo_root / "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records"
    neutral_identity = "retron_msd_structure_panel_v1"
    retired_identity = "reader_" + "spop_msd_structure_panel_v1"

    assert (provenance_root / neutral_identity).is_dir()
    assert not (provenance_root / retired_identity).exists()

    active_roots = (
        repo_root / ".agents/skills/retron-hairpin-study",
        repo_root / "docs/studies/retron_hairpin_design/workbench/provenance",
        repo_root / "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/subject_bindings",
        repo_root / "src/dnadesign/devtools/security/tracked_text_privacy.py",
    )
    for root in active_roots:
        paths = (root,) if root.is_file() else tuple(root.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml"}:
                assert retired_identity not in path.read_text(encoding="utf-8"), path
