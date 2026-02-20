"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/plots/test_plot_fold_change_vs_logic_fidelity.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src.plots import fold_change_vs_logic_fidelity as plot_mod
from dnadesign.opal.src.plots._context import PlotContext


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir


def test_fold_change_plot_does_not_request_setpoint_column(tmp_path, monkeypatch):
    def _stub_load_events_with_setpoint(outputs_dir, base_columns, round_selector=None, run_id=None):
        assert "obj__diag__setpoint" not in base_columns
        return pd.DataFrame(
            {
                "as_of_round": [0],
                "run_id": ["r0"],
                "id": ["a"],
                "pred__y_hat_model": [list(np.linspace(0.0, 1.0, 8))],
                "sel__is_selected": [True],
                "pred__score_selected": [0.5],
                "obj__diag__setpoint": [[0.0, 0.0, 0.0, 1.0]],
            }
        )

    monkeypatch.setattr(plot_mod, "load_events_with_setpoint", _stub_load_events_with_setpoint)

    ctx = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path),
        rounds="unspecified",
        run_id=None,
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="fold_change.png",
        dpi=72,
        format="png",
        logger=logging.getLogger("opal.test.fold_change"),
        save_data=False,
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.render(ctx, params={})

    out = ctx.output_dir / ctx.filename
    assert out.exists()


def test_fold_change_plot_saves_namespaced_columns(tmp_path, monkeypatch):
    def _stub_load_events_with_setpoint(outputs_dir, base_columns, round_selector=None, run_id=None):
        assert "obj__effect_scaled" in base_columns
        return pd.DataFrame(
            {
                "as_of_round": [0],
                "run_id": ["r0"],
                "id": ["a"],
                "pred__y_hat_model": [list(np.linspace(0.0, 1.0, 8))],
                "sel__is_selected": [True],
                "pred__score_selected": [0.5],
                "obj__effect_scaled": [0.9],
                "obj__diag__setpoint": [[0.0, 0.0, 0.0, 1.0]],
            }
        )

    monkeypatch.setattr(plot_mod, "load_events_with_setpoint", _stub_load_events_with_setpoint)

    ctx = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path),
        rounds="unspecified",
        run_id=None,
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="fold_change.png",
        dpi=72,
        format="png",
        logger=logging.getLogger("opal.test.fold_change"),
        save_data=True,
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.render(ctx, params={"y_axis": "effect_scaled"})

    csv_path = ctx.output_dir / "fold_change.csv"
    df = pd.read_csv(csv_path)
    assert "obj__logic_fidelity" in df.columns
    assert "obj__effect_scaled" in df.columns
