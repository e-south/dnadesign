"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_chimerax_manifest_reuse.py

Eco1 review-deliverable ChimeraX manifest reuse tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_tracks,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_unprovenanced_chimerax_png_is_removed_when_rendering_is_disabled(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    existing_png = tmp_path / "review_deliverables" / "mask_structure_context" / "mask_structure_context.png"
    existing_png.parent.mkdir(parents=True, exist_ok=True)
    existing_png.write_bytes(b"existing-render")

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert deliverables["mask_structure_context_png"]["status"] == "skipped_stale_optional_render_removed"
    assert not existing_png.exists()


def test_current_chimerax_png_stays_visible_when_rendering_is_disabled(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    panel_root = tmp_path / "review_deliverables" / "mask_structure_context"
    existing_png = panel_root / "mask_structure_context.png"

    materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    existing_png.write_bytes(b"current-render")
    script_path = panel_root / "mask_structure_context.cxc"
    reference_path = tmp_path / "foldcheck_review" / "structures" / "ec86kit_protomer1_all_atom_reference.pdb"
    mask_tracks._write_render_manifest(
        render_manifest_path=panel_root / "mask_structure_context_render_manifest.yaml",
        script_path=script_path,
        reference_structure_path=reference_path,
        png_path=existing_png,
    )

    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert deliverables["mask_structure_context_png"]["status"] == "reused_existing_optional_render"
    assert existing_png.exists()
