from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dnadesign.opal.src.plots import vector_summary_heatmap as plot_mod
from dnadesign.opal.src.plots._context import PlotContext


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir


def test_vector_summary_explicit_reference_does_not_require_objective_setpoint(tmp_path, monkeypatch) -> None:
    def _stub_load_events(outputs_dir, base_columns, round_selector=None, run_id=None):
        assert "obj__diag__setpoint" not in base_columns
        return pd.DataFrame(
            {
                "as_of_round": [0, 0],
                "run_id": ["r0", "r0"],
                "pred__y_hat_model": [[0.25, 0.75], [0.75, 0.25]],
            }
        )

    def _fail_load_events_with_setpoint(*args, **kwargs):
        raise AssertionError("explicit reference_vector must not require objective setpoint metadata")

    monkeypatch.setattr(plot_mod, "load_events", _stub_load_events)
    monkeypatch.setattr(plot_mod, "load_events_with_setpoint", _fail_load_events_with_setpoint)

    ctx = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path),
        rounds="all",
        run_id=None,
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="vector_summary.png",
        dpi=72,
        format="png",
        logger=logging.getLogger("opal.test.vector_summary"),
        save_data=True,
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.render(
        ctx,
        params={
            "cohort": "all_pool",
            "include_reference_vector": True,
            "reference_vector": [0.0, 1.0],
            "reference_label": "target vec2",
            "channel_labels": ["a", "b"],
        },
    )

    tidy = pd.read_csv(ctx.output_dir / "vector_summary.csv")
    assert set(tidy["row_type"]) == {"reference_vector", "round"}
    assert tidy.loc[tidy["row_type"] == "reference_vector", "cohort"].unique().tolist() == ["target vec2"]
