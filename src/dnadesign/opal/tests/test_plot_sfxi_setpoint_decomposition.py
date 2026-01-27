# ABOUTME: Tests setpoint decomposition plot data loading contract.
# ABOUTME: Ensures derived setpoint column is not requested from predictions.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_plot_sfxi_setpoint_decomposition.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from dnadesign.opal.src.plots import sfxi_setpoint_decomposition as plot_mod
from dnadesign.opal.src.plots._context import PlotContext


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir


def test_setpoint_decomposition_does_not_request_setpoint_column(tmp_path, monkeypatch):
    def _stub_read_runs(path: Path) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "run_id": ["r0"],
                "as_of_round": [0],
                "objective__params": [{"setpoint_vector": [0.0, 0.0, 0.0, 1.0]}],
            }
        )

    def _stub_load_predictions_with_setpoint(
        outputs_dir: Path,
        base_columns,
        round_selector=None,
        run_id=None,
        require_run_id=True,
    ) -> pl.DataFrame:
        assert "obj__diag__setpoint" not in base_columns
        return pl.DataFrame(
            {
                "id": ["a"],
                "pred__y_hat_model": [list(np.linspace(0.0, 1.0, 8))],
                "obj__diag__setpoint": [[0.0, 0.0, 0.0, 1.0]],
            }
        )

    monkeypatch.setattr(plot_mod, "read_runs", _stub_read_runs)
    monkeypatch.setattr(plot_mod, "resolve_single_round", lambda *_, **__: 0)
    monkeypatch.setattr(plot_mod, "resolve_run_id", lambda *_, **__: "r0")
    monkeypatch.setattr(plot_mod, "load_predictions_with_setpoint", _stub_load_predictions_with_setpoint)

    ctx = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path),
        rounds="unspecified",
        run_id=None,
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="sfxi_setpoint_decomposition.png",
        dpi=72,
        format="png",
        logger=logging.getLogger("opal.test.sfxi_setpoint_decomposition"),
        save_data=False,
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.render(ctx, params={"record_id": "a"})

    out_path = ctx.output_dir / ctx.filename
    assert out_path.exists()
