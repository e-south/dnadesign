"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/scar_nick/test_workspace_panel_specs.py

Workspace-level scar_nick panel config tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.scar_nick_workflow import validate_scar_nick_spec
from dnadesign.cruncher.scar_nick.load import resolve_workspace_root_for_scar_nick_spec

_WORKSPACE = Path("src/dnadesign/cruncher/workspaces/scar_nick_teto")
_BBSI_SPEC = _WORKSPACE / "configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml"
_PAQCI_SPEC = _WORKSPACE / "configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml"

_CORE_PANEL = {
    "MXMM",
    "WXMM",
    "XWMM",
    "MWXM",
    "MXWM",
    "XMWM",
    "WMMM",
    "MWMM",
    "MMWM",
    "WWMM",
    "WMWM",
    "MWWM",
    "XXMM",
    "XMXM",
}
_STRICT_CATALOG_UNCOVERED = {"WMWM"}


def test_scar_nick_teto_panel_specs_share_one_workspace() -> None:
    specs = [_BBSI_SPEC, _PAQCI_SPEC]

    workspace_roots = {resolve_workspace_root_for_scar_nick_spec(path).resolve() for path in specs}

    assert workspace_roots == {(_WORKSPACE).resolve()}


def test_scar_nick_teto_specs_cover_listed_strict_catalog_hits() -> None:
    reports = [validate_scar_nick_spec(path) for path in [_BBSI_SPEC, _PAQCI_SPEC]]

    assert {report.status for report in reports} == {"satisfied"}
    assert {Path(report.spec_path).parent.parent.parent.resolve() for report in reports} == {_WORKSPACE.resolve()}
    assert all(report.metadata.materialized_candidate_count > 0 for report in reports)

    covered = {
        candidate.profile_s3s2s1s0
        for report in reports
        for candidate in report.candidates
        if candidate.profile_s3s2s1s0 in _CORE_PANEL
    }

    assert covered == _CORE_PANEL - _STRICT_CATALOG_UNCOVERED
    assert _STRICT_CATALOG_UNCOVERED.isdisjoint(covered)
