"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_run_diagnostics_plots.py

Diagnostics plot coverage for run-level metrics and Stage-B sampling health.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from dnadesign.densegen.src.viz.plotting import (
    _plot_required_columns,
    plot_run_failure_pareto,
    plot_run_timeline_funnel,
    plot_stage_a_score_traceability,
    plot_stage_b_library_health,
    plot_stage_b_library_slack,
    plot_stage_b_offered_vs_used,
    plot_stage_b_sampling_pressure,
    plot_tfbs_positional_occupancy,
)


def _metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_group": "library_health",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "library_size": 3,
                "unique_tf_count": 2,
                "unique_tfbs_count": 3,
                "tf_entropy": 0.9,
                "max_tf_dominance": 0.67,
                "score_mean": 7.0,
                "score_median": 7.0,
                "tfbs_length_sum": 15,
                "required_min_length": 10,
                "fixed_bp_min": 0,
                "slack_bp": 10,
                "sequence_length": 20,
                "target_length": 40,
                "achieved_length": 15,
                "never_used_tfbs_fraction": 0.3,
            },
            {
                "metric_group": "offered_vs_used_tf",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_A",
                "offered_count": 2,
                "used_count": 1,
                "used_fraction": 0.5,
            },
            {
                "metric_group": "offered_vs_used_tf",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_B",
                "offered_count": 1,
                "used_count": 1,
                "used_fraction": 1.0,
            },
            {
                "metric_group": "tier_enrichment",
                "run_id": "demo",
                "input_name": "demo_input",
                "tf": "TF_A",
                "tier": 0,
                "pool_tfbs_count": 1,
                "used_tfbs_count": 1,
                "usage_rate": 1.0,
            },
            {
                "metric_group": "tier_enrichment",
                "run_id": "demo",
                "input_name": "demo_input",
                "tf": "TF_A",
                "tier": 1,
                "pool_tfbs_count": 1,
                "used_tfbs_count": 0,
                "usage_rate": 0.0,
            },
            {
                "metric_group": "tier_enrichment",
                "run_id": "demo",
                "input_name": "demo_input",
                "tf": "TF_B",
                "tier": 2,
                "pool_tfbs_count": 1,
                "used_tfbs_count": 1,
                "usage_rate": 1.0,
            },
            {
                "metric_group": "quantile_enrichment",
                "run_id": "demo",
                "input_name": "demo_input",
                "tf": "TF_A",
                "quantile": 1,
                "pool_tfbs_count": 1,
                "used_tfbs_count": 1,
                "pool_fraction": 0.5,
                "used_fraction": 1.0,
                "enrichment": 2.0,
            },
            {
                "metric_group": "quantile_enrichment",
                "run_id": "demo",
                "input_name": "demo_input",
                "tf": "TF_A",
                "quantile": 2,
                "pool_tfbs_count": 1,
                "used_tfbs_count": 0,
                "pool_fraction": 0.5,
                "used_fraction": 0.0,
                "enrichment": 0.0,
            },
            {
                "metric_group": "sampling_pressure",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_A",
                "weight": 1.5,
                "weight_fraction": 0.75,
                "usage_count": 3,
                "failure_count": 1,
            },
            {
                "metric_group": "sampling_pressure",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_B",
                "weight": 0.5,
                "weight_fraction": 0.25,
                "usage_count": 1,
                "failure_count": 0,
            },
        ]
    )


def test_plot_required_columns_for_positional_occupancy_excludes_plain_length() -> None:
    cols = _plot_required_columns(["tfbs_positional_occupancy"], {})
    assert "length" not in cols
    assert "densegen__sequence_length" in cols or "densegen__length" in cols


def _attempts_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attempt_index": 1,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "success",
                "reason": "ok",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 2,
                "created_at": "2026-01-26T00:00:10+00:00",
                "status": "duplicate",
                "reason": "output_duplicate",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 3,
                "created_at": "2026-01-26T00:00:20+00:00",
                "status": "failed",
                "reason": "no_solution",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
        ]
    )


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event": "RESAMPLE_TRIGGERED",
                "created_at": "2026-01-26T00:00:15+00:00",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
            }
        ]
    )


def test_plot_run_timeline_funnel(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "timeline.png"
    plot_run_timeline_funnel(
        pd.DataFrame(),
        out_path,
        attempts_df=_attempts_df(),
        events_df=_events_df(),
        style={},
    )
    assert out_path.exists()


def test_plot_run_failure_pareto(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "pareto.png"
    plot_run_failure_pareto(
        pd.DataFrame(),
        out_path,
        attempts_df=_attempts_df(),
        style={},
    )
    assert out_path.exists()


def test_plot_stage_b_library_health(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "health.png"
    paths = plot_stage_b_library_health(
        pd.DataFrame(),
        out_path,
        run_metrics_df=_metrics_df(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_stage_a_score_traceability(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "trace.png"
    paths = plot_stage_a_score_traceability(
        pd.DataFrame(),
        out_path,
        run_metrics_df=_metrics_df(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_stage_b_offered_vs_used(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "offered.png"
    paths = plot_stage_b_offered_vs_used(
        pd.DataFrame(),
        out_path,
        run_metrics_df=_metrics_df(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_stage_b_library_slack(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "slack.png"
    paths = plot_stage_b_library_slack(
        pd.DataFrame(),
        out_path,
        run_metrics_df=_metrics_df(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_stage_b_sampling_pressure(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "pressure.png"
    paths = plot_stage_b_sampling_pressure(
        pd.DataFrame(),
        out_path,
        run_metrics_df=_metrics_df(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_tfbs_positional_occupancy(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "occupancy.png"
    df = pd.DataFrame(
        [
            {
                "densegen__input_name": "demo_input",
                "densegen__plan": "demo_plan",
                "densegen__used_tfbs_detail": [
                    {"tf": "TF_A", "tfbs": "AAAA", "offset": 0},
                    {"tf": "TF_B", "tfbs": "CCCCCC", "offset": 10},
                ],
                "length": 20,
            }
        ]
    )
    cfg = {
        "generation": {
            "plan": [
                {
                    "name": "demo_plan",
                    "fixed_elements": {"promoter_constraints": [{"upstream_pos": [0, 5], "downstream_pos": [10, 15]}]},
                }
            ]
        }
    }
    paths = plot_tfbs_positional_occupancy(df, out_path, cfg=cfg, style={})
    assert paths
    assert Path(paths[0]).exists()
