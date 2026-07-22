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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_rows as structure_rows,
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


def test_structure_manifest_parsing_is_cached_until_file_revision_changes(tmp_path: Path, monkeypatch) -> None:
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    manifest_path = browser_root / "structure_browser_manifest.yaml"
    payload = {
        "reference": {"local_path": "reference.pdb"},
        "structures": [{"candidate_id": "candidate_a", "local_path": "candidate_a.pdb"}],
    }
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    deliverables = [
        {
            "artifact_kind": "structure_browser_manifest",
            "deliverable_id": "browser",
            "path": str(manifest_path.relative_to(tmp_path)),
            "section": "test",
            "status": "rendered",
        }
    ]
    parse_calls = 0
    original_safe_load = structure_rows.yaml.safe_load

    def counting_safe_load(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_safe_load(text)

    structure_rows._load_manifest_mapping.cache_clear()
    monkeypatch.setattr(structure_rows.yaml, "safe_load", counting_safe_load)

    first = structure_rows.load_structure_browser_rows(manifest_root=tmp_path, deliverables=deliverables)
    second = structure_rows.load_structure_browser_rows(manifest_root=tmp_path, deliverables=deliverables)
    payload["structures"].append({"candidate_id": "candidate_b", "local_path": "candidate_b.pdb"})
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    third = structure_rows.load_structure_browser_rows(manifest_root=tmp_path, deliverables=deliverables)

    assert [row["candidate_id"] for row in first] == ["candidate_a"]
    assert second == first
    assert [row["candidate_id"] for row in third] == ["candidate_a", "candidate_b"]
    assert parse_calls == 2
