"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_selected_panel_structure_browser_runtime.py

Eco1 selected-panel structure-browser tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_PANEL_SELECTION,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.candidate_pool_fixtures import (
    write_candidate_pool,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.foldcheck_fixtures import (
    write_foldcheck_review_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import FakeMo
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.selection_fixtures import (
    write_selection_readiness_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.selection_table_fixtures import (
    panel_row,
    triage_row,
)


def _selected_rows(result: object) -> list[dict[str, object]]:
    manifest_path = result.manifest_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    return [row for row in rows if row.get("_deliverable_id") == "selected_panel_structure_browser_manifest"]


def test_selected_panel_structure_browser_uses_selection_rows(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    selected_rows = _selected_rows(result)
    assert {row["candidate_id"] for row in selected_rows} == {
        "wild_type",
        "thread_candidate_alpha",
        "thread_candidate_beta",
    }
    alpha_row = next(row for row in selected_rows if row["candidate_id"] == "thread_candidate_alpha")
    assert alpha_row["protein_sequence"] == "MKSAGG"
    assert alpha_row["protein_sequence_length"] == 6
    group_lookup = structure_browser.structure_group_lookup(
        selected_rows,
        selected_section=SECTION_PANEL_SELECTION,
        selected_deliverable_id="selected_panel_structure_browser_manifest",
    )
    lookup = structure_browser.structure_browser_lookup(
        selected_rows,
        selected_section=SECTION_PANEL_SELECTION,
        selected_deliverable_id="selected_panel_structure_browser_manifest",
        selected_group=group_lookup["1 Selected hypothesis: alpha"],
    )
    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"],
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        show_sidechains=True,
    )
    rendered_text = str(rendered)
    for text in (
        "Variant summary",
        "Selection rank",
        "eco1-protein-sequence-panel",
        "MKSAGG",
        "selected_hypothesis_01",
        "MSA observed fraction",
        "Near retained DNA/RNA edits",
        "Near-region charge change",
        "Distal scaffold changes",
    ):
        assert text in rendered_text


def test_selected_panel_structure_browser_uses_selection_root_override(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    active_root = tmp_path / "generation_policies_v3"
    active_selection_root = active_root / "selection"
    write_candidate_pool(active_root / "candidate_pool.parquet", include_generation_policies=True)
    write_foldcheck_review_manifest(active_root / "foldcheck_review")
    write_selection_readiness_manifest(active_selection_root)
    beta_panel_row = panel_row(
        slot="selected_hypothesis_01",
        policy_id="distal_scaffold_repack_v1",
        candidate_id="thread_candidate_beta",
        mutation_count=3,
        msa_fraction=0.6,
        na_facing=0,
        chemistry_warnings=0,
    )
    pq.write_table(pa.Table.from_pylist([beta_panel_row]), active_selection_root / "candidate_selection_panel.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                triage_row(
                    candidate_id="thread_candidate_alpha",
                    msa_fraction=0.7,
                    charge_delta=0,
                    mutation_count=2,
                ),
                triage_row(
                    candidate_id="thread_candidate_beta",
                    msa_fraction=0.6,
                    charge_delta=0,
                    mutation_count=3,
                ),
            ]
        ),
        active_selection_root / "candidate_triage_table.parquet",
    )
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        selection_root=active_selection_root,
        render_chimerax_png=False,
    )
    assert {row["candidate_id"] for row in _selected_rows(result)} == {"wild_type", "thread_candidate_beta"}


def test_selected_hypothesis_structure_groups_are_rank_ordered() -> None:
    rows = [
        {
            "_section": SECTION_PANEL_SELECTION,
            "_deliverable_id": "selected_panel_structure_browser_manifest",
            "group": label,
        }
        for label in (
            "3 Selected hypothesis: gamma",
            "0 WT ColabFold baseline",
            "2 Selected hypothesis: beta",
            "1 Selected hypothesis: alpha",
        )
    ]

    groups = structure_browser.structure_group_lookup(
        rows,
        selected_section=SECTION_PANEL_SELECTION,
        selected_deliverable_id="selected_panel_structure_browser_manifest",
    )

    assert list(groups) == [
        "0 WT ColabFold baseline",
        "1 Selected hypothesis: alpha",
        "2 Selected hypothesis: beta",
        "3 Selected hypothesis: gamma",
    ]
