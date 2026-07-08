"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_lazy_loading.py

Eco1 structure-browser lazy-loading tests.

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


def test_structure_browser_rows_can_load_one_manifest_at_a_time(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    all_rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    selected_rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
        selected_deliverable_id="selected_panel_structure_browser_manifest",
    )
    sae_rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
        selected_deliverable_id="biohub_esmc_sae_structure_browser_manifest",
    )

    assert len(all_rows) > len(selected_rows)
    assert selected_rows
    assert sae_rows
    assert {row["_deliverable_id"] for row in selected_rows} == {"selected_panel_structure_browser_manifest"}
    assert {row["_deliverable_id"] for row in sae_rows} == {"biohub_esmc_sae_structure_browser_manifest"}


def test_structure_highlight_rows_filter_sae_by_selected_candidate(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
        selected_deliverable_id="interactive_structure_browser_manifest",
    )
    selected = next(row for row in rows if row.get("candidate_id") == "thread_candidate_alpha")
    highlight_rows = structure_browser.load_structure_highlight_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
        selected_row=selected,
    )

    assert {row["_deliverable_id"] for row in highlight_rows} == {
        "biohub_esmc_sae_structure_browser_manifest",
        "mask_structure_browser_manifest",
    }
    sae_rows = [row for row in highlight_rows if row["_deliverable_id"] == "biohub_esmc_sae_structure_browser_manifest"]
    assert sae_rows
    assert {row.get("source_candidate_id") or row.get("candidate_id") for row in sae_rows} == {"thread_candidate_alpha"}
