"""Contracts for checked-in promoter-study workspace artifacts and docs refs."""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.latentdna.src.contracts.notebook import WorkspaceNotebookControls
from dnadesign.latentdna.src.services.deliverable_service import deliverable_status


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _live_workspace() -> Path:
    return _repo_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth"


def test_live_study_browser_controls_match_workspace_contract() -> None:
    controls_path = _live_workspace() / "outputs" / "notebooks" / "browser" / "controls.json"
    assert controls_path.is_file()

    controls = WorkspaceNotebookControls.model_validate(json.loads(controls_path.read_text(encoding="utf-8")))
    geometry_ids = {row.view_id for row in controls.geometry_switchboard.geometries}

    assert controls.schema_version == "latentdna.workspace_notebook_controls.v2"
    assert controls.workspace_id == "stress_ethanol_cipro_growth"
    assert controls.notebook_id == "browser"
    assert controls.runtime_paths.workspace_relative_path == "../../.."
    assert controls.runtime_paths.output_relative_path == "../.."
    assert controls.runtime_paths.catalog_relative_path == "../../catalog.json"
    assert controls.runtime_paths.health_relative_path == "../health.json"
    assert {"z20_60", "z20_1k_seq", "z20_1k_anchor"} <= geometry_ids
    assert controls.context_audit.decision in {"whole_sequence_primary", "no_context_signal", "not_evaluated"}


def test_live_study_deliverables_resolve_study_docs_refs() -> None:
    workspace = _live_workspace()

    reference_status = deliverable_status(workspace, "reference_alignment_primary_20b")
    assert {entry["relative_ref"] for entry in reference_status.docs_refs} == {
        "deliverables/reference_alignment_primary_20b",
        "reference_sets/promoter_wt_core",
    }

    geometry_status = deliverable_status(workspace, "geometry_switchboard_20b")
    assert {entry["relative_ref"] for entry in geometry_status.docs_refs} == {
        "deliverables/geometry_switchboard_20b",
        "figures/atlas_2x2_intermediate_main",
    }

    context_audit_status = deliverable_status(workspace, "context_audit_primary_20b")
    assert [entry["relative_ref"] for entry in context_audit_status.docs_refs] == [
        "deliverables/context_audit_primary_20b"
    ]

    x2_status = deliverable_status(workspace, "x2_primary_20b")
    assert [entry["relative_ref"] for entry in x2_status.docs_refs] == ["deliverables/x2_primary_20b"]
