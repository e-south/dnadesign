"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_selection_notebook_runtime.py

Eco1 selection-notebook runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
    notebook_runtime,
    notebook_selection_panel,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_selection_panel_table_reads_metrics_from_trace_json(tmp_path: Path) -> None:
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
    assert rows[0]["WT RMSD A"] == 0.82
    assert rows[0]["cryoEM RMSD A"] == 1.23
    assert rows[0]["unobserved MSA changes"] == 1
    assert rows[0]["NA-facing charge change"] == 1
    why = rendered["items"][2]
    assert why["kind"] == "accordion"
    assert "Why this row: thread_candidate_alpha" in why["items"]
    assert "MSA observed fraction: 0.75" in str(why["items"]["Why this row: thread_candidate_alpha"])
    assert "Nucleic-acid-facing mutations: 1" in str(why["items"]["Why this row: thread_candidate_alpha"])


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
    assert "hard_gate_status" in funnel_text
    assert "wt_like_not_used_for_selection" in funnel_text
    assert "thread_candidate_alpha" in funnel_text
    assert "not panel-selection evidence" in funnel_text

    readiness = notebook_runtime.render_deliverable_artifact(
        deliverables["selection_handoff_readiness"],
        mo=FakeMo(),
        manifest_root=result.manifest_path.parent,
    )
    readiness_text = str(readiness)
    assert "candidate_handoff.yaml is absent" in readiness_text
    assert "construct_subject_created" in readiness_text
    assert "false" in readiness_text.lower()


def test_non_image_artifact_fallback_uses_manifest_relative_path(tmp_path: Path) -> None:
    tmp_path.joinpath("review_deliverables").mkdir()
    tmp_path.joinpath("review_deliverables", "notes.txt").write_text("fixture\n", encoding="utf-8")

    rendered = notebook_runtime.render_deliverable_artifact(
        {
            "artifact_kind": "text",
            "path": "notes.txt",
            "status": "materialized",
        },
        mo=FakeMo(),
        manifest_root=tmp_path / "review_deliverables",
    )

    assert rendered == "Artifact file: `notes.txt`"
    assert str(tmp_path) not in rendered
