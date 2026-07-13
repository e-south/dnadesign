"""Selection and handoff rendering tests for the Eco1 review notebook."""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_runtime,
    notebook_selection_panel,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_PANEL_SELECTION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
    SELECTION_PLOT_METADATA,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_selection_panel_table_reads_explicit_panel_columns(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    table_row = next(row for row in manifest["deliverables"] if row["deliverable_id"] == "selection_panel_table")
    rendered = notebook_selection_panel.render_selection_panel_table(
        table_row,
        mo=FakeMo(),
        table_path=result.manifest_path.parent / table_row["path"],
    )
    rows = rendered["items"][1]["rows"]

    assert rows[0]["mutations"] == 2
    assert rows[0]["pLDDT"] == 92.4
    assert rows[0]["within-policy mutation-position distance"] == 0.4
    assert rows[0]["within-policy exact-substitution distance"] == 0.7
    assert rows[0]["peripheral DNA/RNA edits"] == 1
    assert rows[0]["peripheral charge change"] == 1
    assert rows[0]["basic gains"] == 2
    assert rows[0]["basic losses"] == 1
    assert rows[0]["acidic gains"] == 0
    assert rows[0]["Wang thumb-track edits"] == 0
    assert rows[0]["residues 255-311 edits"] == 1
    assert len(rendered["items"]) == 2
    assert "Why this row" not in str(rendered)


def test_selection_funnel_and_handoff_readiness_render_from_manifest(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {row["deliverable_id"]: row for row in manifest["deliverables"]}

    funnel = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_funnel_summary"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    funnel_text = str(funnel)
    assert "Selection policy" in funnel_text
    assert "Complete ProteinMPNN sequences" in funnel_text
    assert "Distal, peripheral, and combined groups" in funnel_text
    assert "Eight selected sequences" in funnel_text
    assert "experimental comparisons, not quality tiers" in funnel_text
    assert "first-order" not in funnel_text
    assert "safe mutation" not in funnel_text
    assert "hard_gate_status" not in funnel_text
    assert "wt_like_not_used_for_selection" not in funnel_text
    assert "thread_candidate_alpha" in funnel_text
    assert "ESMC policy" not in funnel_text
    assert "SAE policy" not in funnel_text

    readiness = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_handoff_readiness"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    readiness_text = str(readiness)
    assert "candidate_handoff.yaml is absent" in readiness_text
    assert "candidate_handoff_sequence_csv_materialized" in readiness_text
    assert "construct_subject_created" in readiness_text
    assert "false" in readiness_text.lower()

    sequences = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_handoff_sequences"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    sequence_list_html = sequences["items"][1]
    assert "eco1-handoff-sequence-list" in sequence_list_html
    assert "thread_candidate_alpha" in sequence_list_html
    assert "MKSAGG" in sequence_list_html
    sequence_rows = sequences["items"][2]["rows"]
    assert sequence_rows[0]["candidate_id"] == "thread_candidate_alpha"
    assert len(sequence_rows[0]["protein_sequence"]) == 320
    assert sequence_rows[0]["mapped_protein_sequence"].startswith("MKSAGG")
    assert sequence_rows[0]["sequence_scope"] == "canonical_rt_protein"
    assert sequence_rows[0]["dna_design_status"] == "not_materialized"


def test_panel_selection_deliverables_follow_review_sequence(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))

    visible_panel_rows = notebook_runtime.section_deliverables(
        notebook_runtime.visual_deliverables(manifest["deliverables"]),
        SECTION_PANEL_SELECTION,
    )
    visible_panel_ids = [str(row["deliverable_id"]) for row in visible_panel_rows]
    all_panel_rows = notebook_runtime.section_deliverables(manifest["deliverables"], SECTION_PANEL_SELECTION)
    all_panel_ids = [str(row["deliverable_id"]) for row in all_panel_rows]

    assert visible_panel_ids == [
        *[
            plot_id
            for plot_id in CURRENT_SELECTION_PLOT_IDS
            if SELECTION_PLOT_METADATA[plot_id]["role"] == "manuscript_facing"
        ],
        "selected_panel_structure_browser_manifest",
        "communication_selected_panel",
    ]
    assert "selection_funnel_summary" not in visible_panel_ids
    assert "selection_panel_table" not in visible_panel_ids
    assert "selection_handoff_sequences" not in visible_panel_ids
    assert all_panel_ids.index("selected_panel_structure_browser_manifest") < all_panel_ids.index(
        "selection_panel_table"
    )
    assert all_panel_ids.index("selection_panel_table") < all_panel_ids.index("selection_handoff_sequences")


def test_twist_handoff_manifest_renders_sequence_and_cloning_status(tmp_path: Path) -> None:
    manifest_path = tmp_path / "twist_handoff_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.twist_full_cds_handoff",
                "sequence_status": "quote_and_upload_ready",
                "cloning_status": "blocked_pending_assembly_flanks_and_vendor_portal_complexity_check",
                "sequences": [
                    {
                        "sequence_id": "selected_distal_01",
                        "candidate_id": "candidate_1",
                        "selection_rank": 1,
                        "design_group_id": "distal_scaffold_repack",
                        "within_group_rank": 1,
                        "selection_slot": "selected_distal_scaffold_repack_01",
                        "policy_id": "distal_scaffold_repack_v1",
                        "mutation_tokens": ["A47K"],
                        "length_bp": 963,
                        "genbank_file": "genbank/selected_distal_01.gb",
                        "qc": {"gc_fraction": 0.51, "max_homopolymer_run": 5, "forbidden_site_count": 0},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rendered = notebook_runtime.render_deliverable_artifact(
        {"artifact_kind": "twist_handoff_manifest", "path": manifest_path.name, "title": "Twist handoff"},
        mo=FakeMo(),
        manifest_root=tmp_path,
    )
    rendered_text = str(rendered)
    assert "quote_and_upload_ready" in rendered_text
    assert "blocked_pending_assembly_flanks" in rendered_text
    assert "A47K" in rendered_text
    assert "selected_distal_01.gb" in rendered_text
