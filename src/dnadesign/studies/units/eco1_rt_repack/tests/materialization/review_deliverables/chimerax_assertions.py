"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/chimerax_assertions.py

ChimeraX artifact assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    resolve_manifest_path,
)


def assert_chimerax_context_scripts(
    *,
    manifest_path: Path,
    deliverables: dict[str, dict[str, Any]],
    forbidden_path_text: str,
) -> None:
    """Assert ChimeraX review scripts are portable and preserve the mask-context contract."""

    chimerax_text = resolve_manifest_path(
        manifest_path,
        deliverables["mask_structure_context_script"]["path"],
    ).read_text(encoding="utf-8")
    assert "eco1_rt_clade9_plurality25_direct_contact5a_v1" in chimerax_text
    assert "set bgColor white" in chimerax_text
    assert "camera ortho" in chimerax_text
    assert '2dlabels text "Ec86 reference"' in chimerax_text
    assert "view orient" in chimerax_text
    assert "# orientation_preset_id: ec86_reference_thumb_down_v1" in chimerax_text
    assert "ProteinMPNN-designable residues" in chimerax_text
    assert "color" in chimerax_text
    assert forbidden_path_text not in chimerax_text

    orientation_text = resolve_manifest_path(
        manifest_path,
        deliverables["mask_structure_context_orientation_template"]["path"],
    ).read_text(encoding="utf-8")
    assert "Manual orientation handoff" in orientation_text
    assert "save mask_structure_context_orientation.cxs" in orientation_text
    assert "exit" not in orientation_text
    assert forbidden_path_text not in orientation_text
