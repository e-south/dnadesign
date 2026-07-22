"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_highlight_runtime.py

Eco1 structure-highlight routing and validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_rows as structure_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def _capture_highlight_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    deliverable_id: str,
) -> tuple[list[str], list[dict[str, object]]]:
    selected_ids: list[str] = []

    def fake_load_structure_browser_rows(**kwargs: object) -> list[dict[str, object]]:
        selected_ids.append(str(kwargs["selected_deliverable_id"]))
        return [{"_deliverable_id": str(kwargs["selected_deliverable_id"])}]

    monkeypatch.setattr(structure_rows, "load_structure_browser_rows", fake_load_structure_browser_rows)
    rows = structure_rows.load_structure_highlight_rows(
        manifest_root=tmp_path,
        deliverables=[],
        selected_row={"_deliverable_id": deliverable_id, "candidate_id": "thread_candidate_alpha"},
    )
    return selected_ids, rows


def test_selected_panel_highlight_rows_skip_sae_structure_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected_ids, rows = _capture_highlight_sources(
        monkeypatch,
        tmp_path=tmp_path,
        deliverable_id="selected_panel_structure_browser_manifest",
    )
    assert selected_ids == ["mask_structure_browser_manifest"]
    assert rows == [{"_deliverable_id": "mask_structure_browser_manifest"}]


def test_candidate_highlight_rows_load_sae_structure_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    selected_ids, rows = _capture_highlight_sources(
        monkeypatch,
        tmp_path=tmp_path,
        deliverable_id="interactive_structure_browser_manifest",
    )
    assert selected_ids == ["mask_structure_browser_manifest", "biohub_esmc_sae_structure_browser_manifest"]
    assert rows == [
        {"_deliverable_id": "mask_structure_browser_manifest"},
        {"_deliverable_id": "biohub_esmc_sae_structure_browser_manifest"},
    ]


def test_structure_browser_manifest_rejects_missing_declared_pdb(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    full_structure_set_path = tmp_path / "foldcheck_review" / "foldcheck_full_structure_set.yaml"
    payload = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    payload["structures"][0]["local_model_artifact_path"] = "structures/full_fold_set/missing_model.pdb"
    full_structure_set_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="declared structure path is missing"):
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
