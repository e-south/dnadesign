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
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_existing_chimerax_png_stays_visible_when_rendering_is_disabled(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    existing_png = tmp_path / "review_deliverables" / "mask_structure_context" / "mask_structure_context.png"
    existing_png.parent.mkdir(parents=True, exist_ok=True)
    existing_png.write_bytes(b"existing-render")

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert deliverables["mask_structure_context_png"]["status"] == "rendered"
    assert "existing ChimeraX PNG" in deliverables["mask_structure_context_png"]["skip_reason"]
