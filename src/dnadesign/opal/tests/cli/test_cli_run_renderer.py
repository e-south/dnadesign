"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_run_renderer.py

Validates strict run summary rendering contract for selection mode/tie fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.opal.src.cli.formatting.renderers.run import render_run_summary_human


def test_render_run_summary_requires_selection_mode_and_tie() -> None:
    summary = {
        "run_id": "run_001",
        "as_of_round": 0,
        "trained_on": 8,
        "scored": 32,
        "top_k_requested": 5,
        "top_k_effective": 5,
        "ledger": "outputs/ledger/predictions",
        "top_k_source": "yaml_default",
    }

    with pytest.raises(ValueError, match="tie_handling"):
        render_run_summary_human(summary)


def test_render_run_summary_uses_explicit_mode_and_tie() -> None:
    summary = {
        "run_id": "run_001",
        "as_of_round": 0,
        "trained_on": 8,
        "scored": 32,
        "top_k_requested": 5,
        "top_k_effective": 5,
        "ledger": "outputs/ledger/predictions",
        "top_k_source": "yaml_default",
        "tie_handling": "dense_rank",
        "objective_mode": "minimize",
    }

    out = render_run_summary_human(summary)
    assert "objective=minimize tie=dense_rank" in out
