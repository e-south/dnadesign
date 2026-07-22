"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_sae_structure_browser.py

Eco1 SAE structure-browser tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_sae_activation_structure_browser_manifest_renders_feature_regions(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert deliverables["biohub_esmc_sae_structure_browser_manifest"]["status"] == "rendered"
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    group_lookup = structure_browser.structure_group_lookup(
        rows,
        selected_section="esmc_feature_review",
        selected_deliverable_id="biohub_esmc_sae_structure_browser_manifest",
    )
    assert "WT/reference SAE activations" in group_lookup
    assert "ProteinMPNN variant SAE activations" in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section="esmc_feature_review",
        selected_deliverable_id="biohub_esmc_sae_structure_browser_manifest",
        selected_group="ProteinMPNN variant SAE activations",
    )
    selected = next(row for label, row in lookup.items() if "F101" in label)

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<sae-feature-dropdown>",
        structure_group_ui="<sae-group-dropdown>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "SAE activation region" in rendered_text
    assert "F101" in rendered_text
    assert "Fixture exact-dictionary feature description" in rendered_text
    assert "Candidate SAE activation" in rendered_text
    assert "Side chains" in rendered_text
    assert "Side chains are shown only for highlighted candidate residues" in rendered_text
    assert "supports model review context, not activity" in rendered_text
