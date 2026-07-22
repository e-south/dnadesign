"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_communication_design_space_map.py

Eco1 RT communication design-space map tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.constraint_map import (  # noqa: E501
    _motif_anchor_segments,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_communication_design_space_map_names_motifs_sources_and_threshold(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["deliverables"] if item["deliverable_id"] == "communication_design_space_map")
    svg_text = (result.manifest_path.parent / row["path"]).read_text(encoding="utf-8")

    assert "NAxxH" in svg_text
    assert "YADD" in svg_text
    assert "VTG" in svg_text
    assert "Wang et al." in svg_text
    assert "Inouye et al." in svg_text
    assert "WT is clade-9 plurality at \N{GREATER-THAN OR EQUAL TO}25%" in svg_text
    assert "Open: distal scaffold &gt;10 \N{LATIN CAPITAL LETTER A WITH RING ABOVE}" in svg_text
    assert (
        "Open: peripheral shell &gt;5 to \N{LESS-THAN OR EQUAL TO}10 \N{LATIN CAPITAL LETTER A WITH RING ABOVE}"
    ) in svg_text
    assert "Fixed motif context windows (study choice)" in svg_text
    assert "Fixed motif neighborhoods (Wang; Simon)" not in svg_text


def test_communication_design_space_map_uses_exact_motif_anchor_spans() -> None:
    segments = dict(
        _motif_anchor_segments(
            [
                {"canonical_position": position, "manual_mask_reason": reason}
                for reason, positions in (
                    ("retron_x_naxxh", range(105, 110)),
                    ("catalytic_yadd", range(195, 199)),
                    ("retron_y_vtg", range(243, 246)),
                )
                for position in positions
            ]
        )
    )

    assert segments == {
        "NAxxH": {105, 106, 107, 108, 109},
        "YADD": {195, 196, 197, 198},
        "VTG": {243, 244, 245},
    }
