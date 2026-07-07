"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/notebook_selection_assertions.py

Selection-specific notebook contract assertions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def assert_selection_notebook_contract(combined_text: str) -> None:
    assert "selection_readiness_manifest.yaml" in combined_text
    assert "selection_funnel_summary" in combined_text
    assert "render_selection_funnel_summary" in combined_text
    assert "selection_panel_table" in combined_text
    assert "Six Eco1 RT variants form a protein review panel" in combined_text
    assert "handoff_readiness" in combined_text
    assert "render_handoff_readiness" in combined_text
    assert "candidate_handoff.yaml is absent" in combined_text
