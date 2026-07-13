"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_run_renderer.py

Validates strict run summary rendering contract for selection mode/tie fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.opal.src.cli.formatting.renderers.run import render_run_summary_text


def test_render_run_summary_requires_selection_views() -> None:
    summary = {
        "run_id": "run_001",
        "as_of_round": 0,
        "trained_on": 8,
        "scored": 32,
        "ledger": "outputs/ledger/predictions",
    }

    with pytest.raises(ValueError, match="selection_views"):
        render_run_summary_text(summary)


def test_render_run_summary_uses_explicit_mode_and_tie() -> None:
    summary = {
        "run_id": "run_001",
        "as_of_round": 0,
        "trained_on": 8,
        "scored": 32,
        "selection_views": {
            "ethanol": {
                "top_k_requested": 5,
                "top_k_effective": 5,
                "tie_handling": "dense_rank",
                "objective_mode": "minimize",
            }
        },
        "selection_batch_count": 5,
        "ledger": "outputs/ledger/predictions",
    }

    out = render_run_summary_text(summary)
    assert "ethanol: objective=minimize tie=dense_rank" in out
    assert "selection batch" in out
