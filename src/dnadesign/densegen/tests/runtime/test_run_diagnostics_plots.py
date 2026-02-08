"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_diagnostics_plots.py

Coverage for the canonical DenseGen plot set after the scalability refactor.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from dnadesign.densegen.src.core.artifacts.pool import TFBSPoolArtifact
from dnadesign.densegen.src.viz.plot_run import (
    _aggregate_reason_pareto,
    _build_run_health_detail_figure,
    _build_run_health_figure,
    _build_run_health_outcomes_figure,
    _build_tfbs_usage_breakdown_figure,
    _extract_plan_quotas,
    _progress_axis,
    _rate_series_from_counts,
    plot_run_health,
    plot_tfbs_usage,
)
from dnadesign.densegen.src.viz.plot_stage_a_diversity import _build_stage_a_diversity_figure
from dnadesign.densegen.src.viz.plot_stage_a_strata import _build_stage_a_strata_overview_figure
from dnadesign.densegen.src.viz.plot_stage_b_placement import (
    _allocation_summary_lines,
    _build_tfbs_count_records,
    _category_display_label,
    _render_occupancy,
    _render_tfbs_allocation,
    _sanitize_fixed_label,
    _sanitize_tf_label,
    plot_placement_map,
)
from dnadesign.densegen.src.viz.plotting import _load_dense_arrays, _plot_required_columns, plot_stage_a_summary

PLAN_POOL_LABEL = "plan_pool__demo_plan"


@pytest.fixture(autouse=True)
def _close_figures_after_test() -> None:
    yield
    plt.close("all")


def _composition_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "solution_id": "s1",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_A",
                "tfbs": "AAAA",
                "offset": 0,
                "length": 4,
                "end": 4,
            },
            {
                "solution_id": "s1",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_B",
                "tfbs": "CCCCCC",
                "offset": 8,
                "length": 6,
                "end": 14,
            },
            {
                "solution_id": "s2",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "tf": "TF_A",
                "tfbs": "AAAA",
                "offset": 1,
                "length": 4,
                "end": 5,
            },
        ]
    )


def _dense_arrays_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "s1",
                "sequence": "TTGACACCCCTATAATGGGG",
                "densegen__input_name": PLAN_POOL_LABEL,
                "densegen__plan": "demo_plan",
            },
            {
                "id": "s2",
                "sequence": "TTGACAGGGGTATAATCCCC",
                "densegen__input_name": PLAN_POOL_LABEL,
                "densegen__plan": "demo_plan",
            },
        ]
    )


def _library_members_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_A",
                "tfbs": "AAAA",
            },
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_A",
                "tfbs": "AAAT",
            },
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "tf": "TF_B",
                "tfbs": "CCCCCC",
            },
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "tf": "TF_A",
                "tfbs": "AAAA",
            },
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "tf": "TF_B",
                "tfbs": "CCCCCC",
            },
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "tf": "TF_B",
                "tfbs": "CCCCCA",
            },
        ]
    )


def _attempts_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attempt_index": 1,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "ok",
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
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
            }
        ]
    )


def _cfg() -> dict:
    return {
        "generation": {
            "sequence_length": 20,
            "plan": [
                {
                    "name": "demo_plan",
                    "regulator_constraints": {"groups": []},
                    "fixed_elements": {
                        "promoter_constraints": [
                            {
                                "name": "sigma70",
                                "upstream_pos": [0, 6],
                                "downstream_pos": [10, 16],
                                "upstream": "TTGACA",
                                "downstream": "TATAAT",
                                "spacer_length": [4, 4],
                            }
                        ]
                    },
                }
            ],
        }
    }


def _diversity_block(core_len: int) -> dict:
    return {
        "candidate_pool_size": 2,
        "nnd_unweighted_k1": {
            "top_candidates": {
                "bins": [0, 1, 2],
                "counts": [0, 2, 0],
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
                "frac_le_1": 1.0,
                "n": 2,
                "subsampled": False,
                "k": 1,
            },
            "diversified_candidates": {
                "bins": [0, 1, 2],
                "counts": [0, 2, 0],
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
                "frac_le_1": 1.0,
                "n": 2,
                "subsampled": False,
                "k": 1,
            },
        },
        "nnd_unweighted_median_top": 1.0,
        "nnd_unweighted_median_diversified": 1.0,
        "delta_nnd_unweighted_median": 0.0,
        "core_hamming": {
            "metric": "hamming",
            "nnd_k1": {
                "k": 1,
                "top_candidates": {
                    "bins": [0, 1, 2],
                    "counts": [0, 2, 0],
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
                "diversified_candidates": {
                    "bins": [0, 1, 2],
                    "counts": [0, 2, 0],
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
            },
            "nnd_k5": None,
            "pairwise": {
                "top_candidates": {
                    "bins": [0.0, 1.0, 2.0],
                    "counts": [0, 1, 0],
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "diversified_candidates": {
                    "bins": [0.0, 1.0, 2.0],
                    "counts": [0, 1, 0],
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "max_diversity_upper_bound": {
                    "bins": [0.0, 1.0, 2.0],
                    "counts": [0, 1, 0],
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
            },
        },
        "set_overlap_fraction": 1.0,
        "set_overlap_swaps": 0,
        "core_entropy": {
            "top_candidates": {"values": [0.0] * core_len, "n": 2},
            "diversified_candidates": {"values": [0.0] * core_len, "n": 2},
        },
        "score_quantiles": {
            "top_candidates": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "diversified_candidates": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "top_candidates_global": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "max_diversity_upper_bound": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
        },
    }


def _pool_manifest(tmp_path: Path, *, include_diversity: bool = False) -> TFBSPoolArtifact:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": 2,
                "columns": [
                    "input_name",
                    "tf",
                    "tfbs_sequence",
                    "tfbs_core",
                    "best_hit_score",
                    "tier",
                    "rank_within_regulator",
                    "selection_rank",
                    "nearest_selected_similarity",
                    "selection_score_norm",
                    "nearest_selected_distance_norm",
                ],
                "pool_mode": "tfbs",
                "stage_a_sampling": {
                    "backend": "fimo",
                    "tier_scheme": "pct_0.1_1_9",
                    "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
                    "retention_rule": "top_n_sites_by_best_hit_score",
                    "fimo_thresh": 1.0,
                    "bgfile": None,
                    "background_source": "motif_background",
                    "eligible_score_hist": [
                        {
                            "regulator": "TF_A",
                            "pwm_consensus": "AAA",
                            "pwm_consensus_iupac": "AAA",
                            "pwm_consensus_score": 2.0,
                            "pwm_theoretical_max_score": 2.0,
                            "edges": [0.0, 1.0, 2.0],
                            "counts": [1, 1],
                            "tier0_score": 2.0,
                            "tier1_score": 1.0,
                            "tier2_score": 0.5,
                            "tier_fractions": [0.001, 0.01, 0.09],
                            "tier_fractions_source": "default",
                            "generated": 10,
                            "candidates_with_hit": 8,
                            "eligible_raw": 6,
                            "eligible_unique": 4,
                            "retained": 2,
                            "selection_policy": "mmr",
                            "selection_alpha": 0.9,
                            "selection_similarity": "weighted_hamming_tolerant",
                            "selection_relevance_norm": "minmax_raw_score",
                            "selection_pool_size_final": 20,
                            "selection_pool_rung_fraction_used": 0.001,
                            "selection_pool_min_score_norm_used": None,
                            "selection_pool_capped": False,
                            "selection_pool_cap_value": None,
                            "mining_audit": None,
                            "padding_audit": None,
                        }
                    ],
                },
            }
        ],
    }
    if include_diversity:
        consensus = manifest["inputs"][0]["stage_a_sampling"]["eligible_score_hist"][0]["pwm_consensus"]
        manifest["inputs"][0]["stage_a_sampling"]["eligible_score_hist"][0]["diversity"] = _diversity_block(
            core_len=len(str(consensus))
        )
    path = pools_dir / "pool_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return TFBSPoolArtifact.load(path)


def _pool_manifest_two_inputs(tmp_path: Path) -> TFBSPoolArtifact:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    base_entry = {
        "type": "binding_sites",
        "rows": 2,
        "columns": [
            "input_name",
            "tf",
            "tfbs_sequence",
            "tfbs_core",
            "best_hit_score",
            "tier",
            "rank_within_regulator",
            "selection_rank",
            "nearest_selected_similarity",
            "selection_score_norm",
            "nearest_selected_distance_norm",
        ],
        "pool_mode": "tfbs",
        "stage_a_sampling": {
            "backend": "fimo",
            "tier_scheme": "pct_0.1_1_9",
            "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
            "retention_rule": "top_n_sites_by_best_hit_score",
            "fimo_thresh": 1.0,
            "bgfile": None,
            "background_source": "motif_background",
            "eligible_score_hist": [
                {
                    "regulator": "TF_A",
                    "pwm_consensus": "AAA",
                    "pwm_consensus_iupac": "AAA",
                    "pwm_consensus_score": 2.0,
                    "pwm_theoretical_max_score": 2.0,
                    "edges": [0.0, 1.0, 2.0],
                    "counts": [1, 1],
                    "tier0_score": 2.0,
                    "tier1_score": 1.0,
                    "tier2_score": 0.5,
                    "tier_fractions": [0.001, 0.01, 0.09],
                    "tier_fractions_source": "default",
                    "generated": 10,
                    "candidates_with_hit": 8,
                    "eligible_raw": 6,
                    "eligible_unique": 4,
                    "retained": 2,
                    "selection_policy": "mmr",
                    "selection_alpha": 0.9,
                    "selection_similarity": "weighted_hamming_tolerant",
                    "selection_relevance_norm": "minmax_raw_score",
                    "selection_pool_size_final": 20,
                    "selection_pool_rung_fraction_used": 0.001,
                    "selection_pool_min_score_norm_used": None,
                    "selection_pool_capped": False,
                    "selection_pool_cap_value": None,
                    "mining_audit": None,
                    "padding_audit": None,
                }
            ],
        },
    }
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            dict(
                base_entry,
                name="demo_input",
                pool_path="demo_input__pool.parquet",
            ),
            dict(
                base_entry,
                name="demo_input_b",
                pool_path="demo_input_b__pool.parquet",
            ),
        ],
    }
    for entry in manifest["inputs"]:
        consensus = entry["stage_a_sampling"]["eligible_score_hist"][0]["pwm_consensus"]
        entry["stage_a_sampling"]["eligible_score_hist"][0]["diversity"] = _diversity_block(
            core_len=len(str(consensus))
        )
    path = pools_dir / "pool_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return TFBSPoolArtifact.load(path)


def _background_pool_manifest(tmp_path: Path) -> TFBSPoolArtifact:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "neutral_bg",
                "type": "background_pool",
                "pool_path": "neutral_bg__pool.parquet",
                "rows": 4,
                "columns": ["input_name", "tf", "tfbs", "tfbs_core", "tfbs_id"],
                "pool_mode": "tfbs",
                "stage_a_sampling": None,
            }
        ],
    }
    path = pools_dir / "pool_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return TFBSPoolArtifact.load(path)


def test_plot_required_columns_for_new_plots() -> None:
    cols = _plot_required_columns(["placement_map", "tfbs_usage", "run_health"], {})
    assert cols == []


def test_load_dense_arrays_requires_dense_arrays_table(tmp_path: Path) -> None:
    tables = tmp_path / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "solution_id": "sol_1",
                "sequence": "TTGACATATAAT",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
            },
            {
                "solution_id": None,
                "sequence": "CCCC",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
            },
        ]
    ).to_parquet(tables / "solutions.parquet", index=False)

    with pytest.raises(ValueError, match="dense_arrays.parquet not found"):
        _load_dense_arrays(tmp_path)


def test_plot_run_health(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "run_health.png"
    paths = plot_run_health(
        pd.DataFrame(),
        out_path,
        attempts_df=_attempts_df(),
        events_df=_events_df(),
        cfg={"config": {"generation": {"plan": [{"name": "demo_plan", "quota": 12}]}}},
        style={},
    )
    assert len(paths) == 2
    rel_paths = {str(Path(path).relative_to(tmp_path)) for path in paths}
    assert "run_health/outcomes_over_time.png" in rel_paths
    assert "run_health/run_health.png" in rel_paths
    assert (tmp_path / "run_health" / "outcomes_over_time.png").exists()
    assert (tmp_path / "run_health" / "run_health.png").exists()
    assert not (tmp_path / "run_health" / "run_health_detail.png").exists()
    assert (tmp_path / "run_health" / "summary.csv").exists()


def test_run_health_outcomes_legend_and_waste_subtitle() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = "postprocess_forbidden_kmer"
    attempts.loc[:, "sampling_library_index"] = [1, 2, 3]
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        ax = axes["outcome"]
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert "accepted" in legend_labels
        assert "rejected" in legend_labels
        assert "failed" in legend_labels
        assert "duplicate" not in legend_labels
        assert ax.get_ylabel() == ""
        assert ax._left_title.get_text() == ""
        label_size = ax.xaxis.label.get_size()
        assert all(text.get_size() == pytest.approx(label_size) for text in legend.get_texts())
        assert legend.get_bbox_to_anchor()._bbox.x0 >= 1.0
        connector_lines = [line for line in ax.get_lines() if line.get_color() == "#c7c7c7"]
        assert connector_lines
        assert all(float(line.get_alpha() or 1.0) <= 0.55 for line in connector_lines)
    finally:
        fig.clf()


def test_run_health_outcomes_plot_is_single_panel_event_map() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[2, "status"] = "failed"
    attempts.loc[:, "sampling_library_index"] = [1, 2, 3]
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        assert fig._suptitle is None
        assert set(axes.keys()) == {"outcome"}
        ax = axes["outcome"]
        assert ax.get_title() == "Outcomes over time"
        assert ax._left_title.get_text() == ""
        assert ax.get_ylabel() == ""
        assert all("Rejected/failed reason composition" not in t.get_text() for t in ax.texts)
        assert all("Quota attainment by plan" not in t.get_text() for t in ax.texts)
        legend = ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["accepted", "rejected", "failed"]
        assert any(
            str(line.get_color())
            in {
                "#d0d0d0",
                "#c7c7c7",
                (0.8156862745098039, 0.8156862745098039, 0.8156862745098039, 1.0),
            }
            for line in ax.get_lines()
        )
        collections = ax.collections
        assert collections
        size_values = [float(np.max(coll.get_sizes())) for coll in collections if len(coll.get_sizes()) > 0]
        assert size_values
        assert min(size_values) >= 30.0
        square_collections = [
            coll
            for coll in collections
            if coll.get_paths() and len(coll.get_paths()[0].vertices) == 5 and len(coll.get_edgecolors()) > 0
        ]
        assert square_collections
        assert all(float(np.mean(coll.get_edgecolors()[0][:3])) >= 0.65 for coll in square_collections)
    finally:
        fig.clf()


def test_run_health_outcomes_connectors_draw_with_step_gaps() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": 1,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 2,
                "created_at": "2026-01-26T00:00:10+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "demo_plan",
                "sampling_library_index": 3,
            },
        ]
    )
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        connector_lines = [line for line in axes["outcome"].get_lines() if line.get_color() in {"#d0d0d0", "#c7c7c7"}]
        assert connector_lines
    finally:
        fig.clf()


def test_run_health_outcomes_connectors_follow_actual_run_order() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": 20,
                "created_at": "2026-01-26T00:00:02+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_b",
                "sampling_library_index": 2,
            },
            {
                "attempt_index": 10,
                "created_at": "2026-01-26T00:00:01+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_a",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 40,
                "created_at": "2026-01-26T00:00:04+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_b",
                "sampling_library_index": 4,
            },
            {
                "attempt_index": 30,
                "created_at": "2026-01-26T00:00:03+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_a",
                "sampling_library_index": 3,
            },
        ]
    )
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"plan_a": 12, "plan_b": 12},
    )
    try:
        connector_lines = [line for line in axes["outcome"].get_lines() if line.get_color() == "#c7c7c7"]
        assert connector_lines
        x_data = np.asarray(connector_lines[0].get_xdata(), dtype=float)
        y_data = np.asarray(connector_lines[0].get_ydata(), dtype=float)
        assert np.array_equal(x_data, np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.array_equal(y_data, np.array([0.0, 1.0, 0.0, 1.0]))
    finally:
        fig.clf()


def test_run_health_detail_plot_has_square_panels_without_subtitles() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = 'postprocess_forbidden_kmer {"forbidden_kmer":"ATGC"}'
    attempts = pd.concat(
        [
            attempts,
            pd.DataFrame(
                [
                    {
                        "attempt_index": 4,
                        "created_at": "2026-01-26T00:00:30+00:00",
                        "status": "failed",
                        "reason": 'postprocess_forbidden_kmer {"forbidden_kmer":"GGGG"}',
                        "plan_name": "alt_plan",
                        "sampling_library_index": 4,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fig, axes = _build_run_health_detail_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12, "alt_plan": 4},
    )
    try:
        assert fig._suptitle is None
        assert set(axes.keys()) == {"fail", "plan"}
        assert axes["fail"].get_box_aspect() == pytest.approx(1.0)
        assert axes["plan"].get_box_aspect() == pytest.approx(1.0)
        assert axes["fail"].get_title() == "Reason for failed solve"
        labels = [tick.get_text() for tick in axes["fail"].get_yticklabels()]
        assert any("ATGC" in label or "GGGG" in label for label in labels)
        assert axes["fail"].get_xlim()[1] == pytest.approx(1.0)
        assert fig.legends
        assert fig.legends[0]._ncols == 2
        assert axes["plan"].get_title() == "Quota progress"
        quota_text = next(text for text in axes["plan"].texts if "Quota (" in text.get_text())
        assert quota_text.get_ha() == "left"
        assert quota_text.get_position()[0] <= 0.1
        assert all(bar.get_height() < 0.7 for bar in axes["fail"].patches)
    finally:
        fig.clf()


def test_plot_run_health_no_duplicates_note() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df()
    attempts = attempts[attempts["status"] == "ok"].reset_index(drop=True)
    fig, axes = _build_run_health_figure(attempts, events_df=None, style={}, plan_quotas={"demo_plan": 12})
    dup_texts = [text.get_text() for text in axes["dup"].texts]
    plt.close(fig)
    assert any("No waste observed" in text for text in dup_texts)


def test_progress_axis_uses_discrete_mode_for_small_runs() -> None:
    df = pd.DataFrame({"run_order": np.arange(1, 151, dtype=int)})
    axis = _progress_axis(df, max_points=500, target_bins=160, min_bin_size=10)
    assert axis.mode == "discrete"
    assert axis.bin_size == 1
    assert axis.x.shape[0] == 150
    assert axis.bin_id is None


def test_progress_axis_binned_mode_respects_min_bin_size() -> None:
    df = pd.DataFrame({"run_order": np.arange(1, 1235, dtype=int)})
    axis = _progress_axis(df, max_points=500, target_bins=160, min_bin_size=10)
    assert axis.mode == "binned"
    assert axis.bin_size >= 10
    counts = np.bincount(axis.bin_id)
    assert counts.min() >= 10


def test_rate_series_duplicate_rate_defined_when_waste_zero() -> None:
    counts = pd.DataFrame(
        {
            "ok": [3, 4, 5],
            "rejected": [0, 0, 0],
            "duplicate": [0, 0, 0],
            "failed": [0, 0, 0],
        }
    )
    rates = _rate_series_from_counts(counts)
    assert np.all(np.isfinite(rates["acceptance"]))
    assert np.all(np.isfinite(rates["waste"]))
    assert np.all(np.isfinite(rates["duplicate"]))
    assert np.allclose(rates["duplicate"], 0.0)


def test_reason_pareto_has_unknown_and_other() -> None:
    rows: list[dict[str, object]] = [
        {"status": "failed", "reason": None},
        {"status": "failed", "reason": ""},
        {"status": "rejected", "reason": "postprocess_forbidden_kmer"},
    ]
    for i in range(10):
        rows.append({"status": "failed", "reason": f"custom_reason_{i}"})
    problem = pd.DataFrame(rows)
    pareto = _aggregate_reason_pareto(problem, top_k=5)
    assert "unknown" in pareto.index
    assert "other" in pareto.index


def test_run_health_rejects_unknown_statuses() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df().copy()
    attempts.loc[0, "status"] = "success"
    with pytest.raises(ValueError, match="Unknown attempt status"):
        _build_run_health_figure(attempts, events_df=None, style={}, plan_quotas={"demo_plan": 12})


def test_run_health_uses_created_at_order_not_attempt_index() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": 100,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 200,
                "created_at": "2026-01-26T00:00:01+00:00",
                "status": "rejected",
                "reason": "postprocess_forbidden_kmer",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 300,
                "created_at": "2026-01-26T00:00:02+00:00",
                "status": "failed",
                "reason": "no_solution",
                "plan_name": "demo_plan",
                "sampling_library_index": 1,
            },
        ]
    )
    fig, axes = _build_run_health_figure(attempts, events_df=None, style={}, plan_quotas={"demo_plan": 3})
    try:
        assert axes["outcome"].get_xlim()[1] < 10.0
    finally:
        fig.clf()


def test_run_health_large_runs_omit_binning_or_rolling_annotations() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        {
            "attempt_index": np.arange(1, 701, dtype=int),
            "created_at": pd.date_range("2026-01-26T00:00:00Z", periods=700, freq="1s").astype(str),
            "status": np.where(np.arange(700) % 7 == 0, "rejected", "ok"),
            "reason": np.where(np.arange(700) % 7 == 0, "postprocess_forbidden_kmer", "ok"),
            "plan_name": ["demo_plan"] * 700,
            "sampling_library_index": [1] * 700,
        }
    )
    fig, axes = _build_run_health_figure(attempts, events_df=None, style={}, plan_quotas={"demo_plan": 700})
    try:
        dup_text = [text.get_text().lower() for text in axes["dup"].texts]
        assert not any("binned" in text for text in dup_text)
        assert not any("rolling window" in text for text in dup_text)
        assert axes["outcome"].get_ylabel() != "Fraction of attempts"
    finally:
        fig.clf()


def test_run_health_hides_event_overlays_by_default() -> None:
    matplotlib.use("Agg", force=True)
    fig, axes = _build_run_health_figure(
        _attempts_df(),
        events_df=_events_df(),
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        for key in ("outcome", "dup", "plan"):
            lines = axes[key].get_lines()
            assert not any(str(line.get_color()) == "#6b6b6b" and line.get_linestyle() == "--" for line in lines)
    finally:
        fig.clf()


def test_run_health_default_rates_show_waste_semantics() -> None:
    matplotlib.use("Agg", force=True)
    fig, axes = _build_run_health_figure(
        _attempts_df(),
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        _handles, labels = axes["dup"].get_legend_handles_labels()
        assert "waste rate" in labels
        assert axes["dup"].get_legend() is None
        subtitles = [t.get_text().lower() for t in axes["dup"].texts]
        assert any("waste = rejected + duplicate + failed per solver step." in text for text in subtitles)
        assert any("dashed line shows duplicate share." in text for text in subtitles)
    finally:
        fig.clf()


def test_run_health_uses_solver_step_x_axis_when_available() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[:, "sampling_library_index"] = [1, 2, 3]
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        assert axes["dup"].get_xlabel() == "Solver step"
    finally:
        fig.clf()


def test_extract_plan_quotas_supports_densegen_wrapped_effective_config() -> None:
    cfg = {
        "config": {
            "densegen": {
                "generation": {
                    "plan": [
                        {"name": "demo_plan", "quota": 12},
                        {"name": "alt_plan", "quota": "4"},
                    ]
                }
            }
        }
    }
    quotas = _extract_plan_quotas(cfg)
    assert quotas == {"demo_plan": 12, "alt_plan": 4}


def test_run_health_missing_plan_name_is_normalized() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[0, "plan_name"] = np.nan
    attempts.loc[1, "plan_name"] = ""
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12, "all plans": 12},
    )
    try:
        assert axes["outcome"] is not None
    finally:
        fig.clf()


def test_run_health_reason_breaks_out_forbidden_kmer_token() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = 'postprocess_forbidden_kmer {"forbidden_kmer":"ATGC"}'
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        labels = [tick.get_text() for tick in axes["fail"].get_yticklabels()]
        assert any("ATGC" in label for label in labels)
    finally:
        fig.clf()


def test_run_health_reason_breaks_out_forbidden_kmer_list_tokens() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = 'postprocess_forbidden_kmer {"forbidden_kmers":["ATGC","GGGG"]}'
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        labels = [tick.get_text() for tick in axes["fail"].get_yticklabels()]
        assert any("ATGC" in label and "GGGG" in label for label in labels)
    finally:
        fig.clf()


def test_plot_placement_map(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "placement_map.png"
    paths = plot_placement_map(
        pd.DataFrame(),
        out_path,
        composition_df=_composition_df(),
        dense_arrays_df=_dense_arrays_df(),
        library_members_df=_library_members_df(),
        cfg=_cfg(),
        style={},
    )
    assert len(paths) == 2
    rel = {str(Path(path).relative_to(tmp_path)) for path in paths}
    assert "stage_b/demo_plan/occupancy.png" in rel
    assert "stage_b/demo_plan/tfbs_allocation.png" in rel
    for path in paths:
        path = Path(path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_placement_map_accepts_effective_config(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "placement_map_effective.png"
    paths = plot_placement_map(
        pd.DataFrame(),
        out_path,
        composition_df=_composition_df(),
        dense_arrays_df=_dense_arrays_df(),
        library_members_df=_library_members_df(),
        cfg={"config": _cfg()},
        style={},
    )
    assert len(paths) == 2
    for path in paths:
        path = Path(path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_placement_map_label_sanitizer() -> None:
    assert _sanitize_tf_label("lexA_CTGTATAW") == "lexA"
    assert _sanitize_tf_label("cpxR") == "cpxR"
    assert _sanitize_fixed_label("fixed:sigma70_consensus:-35") == "sigma70_consensus -35"
    assert _sanitize_fixed_label("fixed:sigma70_consensus:-10") == "sigma70_consensus -10"


def test_plot_tfbs_usage(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "tfbs_usage.png"
    paths = plot_tfbs_usage(
        pd.DataFrame(),
        out_path,
        composition_df=_composition_df(),
        style={},
    )
    assert paths
    path = Path(paths[0])
    assert path.exists()
    assert str(path.relative_to(tmp_path)) == "stage_b/demo_plan/tfbs_usage.png"


def test_tfbs_usage_breakdown_figure_has_category_curves() -> None:
    matplotlib.use("Agg", force=True)
    composition = _composition_df().copy()
    composition = pd.concat(
        [
            composition,
            pd.DataFrame(
                [
                    {
                        "solution_id": "s1",
                        "input_name": PLAN_POOL_LABEL,
                        "plan_name": "demo_plan",
                        "library_index": 1,
                        "library_hash": "hash1",
                        "tf": "fixed:sigma70:-35",
                        "tfbs": "TTGACA",
                        "offset": 0,
                        "length": 6,
                        "end": 6,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fig, axes = _build_tfbs_usage_breakdown_figure(
        composition,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        style={},
    )
    try:
        left_lines = axes["usage"].get_lines()
        right_lines = axes["cum"].get_lines()
        assert len(left_lines) >= 2
        assert len(right_lines) >= 2
        legend = axes["usage"].get_legend()
        if legend is not None:
            legend_text = "\n".join(t.get_text() for t in legend.get_texts())
        else:
            assert fig.legends
            legend_text = "\n".join(t.get_text() for t in fig.legends[0].get_texts())
        assert "fixed:sigma70:-35" in legend_text
    finally:
        fig.clf()


def test_placement_allocation_includes_fixed_constraint_records() -> None:
    composition = _composition_df()
    dense_arrays = _dense_arrays_df()
    constraints = _cfg()["generation"]["plan"][0]["fixed_elements"]["promoter_constraints"]
    records = _build_tfbs_count_records(
        composition,
        solutions=dense_arrays,
        constraints=constraints,
    )
    labels = set(records["category_label"].astype(str).tolist())
    assert "TF_A" in labels
    assert any(label.startswith("fixed:sigma70") for label in labels)


def test_allocation_summary_lines_define_denominators() -> None:
    lines = _allocation_summary_lines(
        placements_used=36,
        placements_possible=72,
        unique_used=28,
        unique_available=40,
        top10_share=0.50,
        top50_share=1.00,
    )
    joined = "\n".join(lines)
    assert "TFBS placements used / possible" in joined
    assert "unique TFBS-pairs used / available" in joined
    assert "top10 share (all TFBS-pairs by usage)" in joined
    assert "top50 share (all TFBS-pairs by usage)" in joined


def test_occupancy_legend_is_below_xlabel() -> None:
    matplotlib.use("Agg", force=True)
    seq_len = 20
    occupancy = {
        "TF_A": np.ones(seq_len),
        "TF_B": np.ones(seq_len),
        "fixed:sigma70:-35": np.ones(seq_len),
        "fixed:sigma70:-10": np.ones(seq_len),
    }
    categories = list(occupancy.keys())
    fixed_label_sequences = {
        "fixed:sigma70:-35": "TTGACA",
        "fixed:sigma70:-10": "TATAAT",
    }
    fig, ax = _render_occupancy(
        occupancy,
        categories,
        seq_len=seq_len,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        n_solutions=2,
        alpha=0.22,
        fixed_label_sequences=fixed_label_sequences,
        style={},
    )
    try:
        fig.canvas.draw()
        assert ax.get_title().startswith("Occupancy across sequence positions")
        assert ax.get_title().endswith(".")
        assert fig.legends
        legend = fig.legends[0]
        assert getattr(legend, "_ncols", None) < len(categories)
        assert getattr(legend, "_ncols", 0) <= 3
        assert legend.get_frame_on() is False
        assert min(text.get_fontsize() for text in legend.get_texts()) >= 9.0
        legend_text = "\n".join(text.get_text() for text in legend.get_texts())
        assert "-35 (TTGACA)" in legend_text
        assert "-10 (TATAAT)" in legend_text
        renderer = fig.canvas.get_renderer()
        xlab_bbox = ax.xaxis.get_label().get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        assert legend_bbox.y1 < xlab_bbox.y0
        assert fig.subplotpars.bottom >= 0.28
    finally:
        fig.clf()


def test_category_display_label_background_and_fixed_sequence() -> None:
    fixed_label_sequences = {
        "fixed:sigma70_consensus:-35": "TTGACA",
        "fixed:sigma70_consensus:-10": "TATAAT",
    }
    assert _category_display_label("neutral_bg", fixed_label_sequences=fixed_label_sequences) == "background"
    assert (
        _category_display_label("fixed:sigma70_consensus:-35", fixed_label_sequences=fixed_label_sequences)
        == "-35 (TTGACA)"
    )
    assert (
        _category_display_label("fixed:sigma70_consensus:-10", fixed_label_sequences=fixed_label_sequences)
        == "-10 (TATAAT)"
    )


def test_tfbs_allocation_legend_layout_is_compact_and_wrapped() -> None:
    matplotlib.use("Agg", force=True)
    counts = pd.DataFrame(
        {
            "category_label": ["TF_A", "TF_B", "TF_C", "TF_D", "TF_E", "TF_F"],
            "tfbs": ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT", "TGCA"],
            "count": [12, 10, 8, 6, 4, 2],
        }
    )
    available = counts.copy()
    available["weight"] = 12
    fig, axes = _render_tfbs_allocation(
        counts,
        available=available,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        n_solutions=12,
        top_k_annotation=0,
        fixed_label_sequences={},
        style={},
    )
    try:
        fig.canvas.draw()
        assert fig.legends
        legend = fig.legends[0]
        renderer = fig.canvas.get_renderer()
        xlabel_bbox = (
            axes["cum"].xaxis.get_label().get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        )
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        gap = xlabel_bbox.y0 - legend_bbox.y1
        assert legend_bbox.y1 < xlabel_bbox.y0
        assert gap <= 0.06
        assert getattr(legend, "_ncols", 0) < len(legend.get_texts())
        assert getattr(legend, "_ncols", 0) <= 4
        assert legend.get_frame_on() is False
        assert min(text.get_fontsize() for text in legend.get_texts()) >= 8.6
    finally:
        fig.clf()


def test_plot_stage_a_summary(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [None, 0.5],
        }
    )
    pools = {"demo_input": pool_df}
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    paths = plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )
    assert paths
    assert len(paths) == 3
    assert Path(paths[0]).exists()
    assert Path(paths[1]).exists()
    assert Path(paths[2]).exists()


def test_plot_stage_a_summary_consolidates_inputs(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary.png"
    pool_df_a = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [None, 0.5],
        }
    )
    pool_df_b = pool_df_a.copy()
    pool_df_b["input_name"] = "demo_input_b"
    pools = {"demo_input": pool_df_a, "demo_input_b": pool_df_b}
    manifest = _pool_manifest_two_inputs(tmp_path)
    paths = plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )
    assert len(paths) == 3
    rel = {str(Path(path).relative_to(tmp_path)) for path in paths}
    assert "stage_a/pool_tiers.png" in rel
    assert "stage_a/yield_bias.png" in rel
    assert "stage_a/diversity.png" in rel


def test_plot_stage_a_summary_default_dimensions_are_wider_and_shorter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [None, 0.5],
        }
    )
    pools = {"demo_input": pool_df}
    manifest = _pool_manifest(tmp_path, include_diversity=True)

    size_by_name: dict[str, tuple[float, float]] = {}
    original_savefig = Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        name = Path(str(fname)).name
        if name in {"pool_tiers.png", "yield_bias.png", "diversity.png"}:
            size = tuple(float(v) for v in self.get_size_inches())
            size_by_name[name] = size
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)
    plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )

    assert set(size_by_name.keys()) == {"pool_tiers.png", "yield_bias.png", "diversity.png"}
    assert size_by_name["pool_tiers.png"][0] >= 15.5
    assert size_by_name["yield_bias.png"][0] >= 15.5
    assert size_by_name["diversity.png"][0] >= 15.5
    assert size_by_name["pool_tiers.png"][1] <= 2.5
    assert size_by_name["yield_bias.png"][1] <= 3.0
    assert size_by_name["diversity.png"][1] <= 2.7


def test_plot_stage_a_summary_includes_background_logo(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary_bg.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["neutral_bg"] * 4,
            "tf": ["neutral_bg"] * 4,
            "tfbs": ["AAAA", "AAAT", "AAAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT", "AAAAA", "AAAAT"],
            "tfbs_id": ["a", "b", "c", "d"],
        }
    )
    pools = {"neutral_bg": pool_df}
    manifest = _background_pool_manifest(tmp_path)
    paths = plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )
    assert paths
    assert len(paths) == 1
    assert Path(paths[0]).exists()


def test_plot_stage_a_summary_requires_diversity(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary_missing.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [None, 0.5],
        }
    )
    pools = {"demo_input": pool_df}
    manifest = _pool_manifest(tmp_path, include_diversity=False)
    with pytest.raises(ValueError, match="diversity"):
        plot_stage_a_summary(
            pd.DataFrame(),
            out_path,
            pools=pools,
            pool_manifest=manifest,
            style={},
        )


def test_plot_stage_a_summary_requires_selection_rank(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary_missing_rank.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [None, 0.5],
        }
    )
    pools = {"demo_input": pool_df}
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    with pytest.raises(ValueError, match="selection_rank"):
        plot_stage_a_summary(
            pd.DataFrame(),
            out_path,
            pools=pools,
            pool_manifest=manifest,
            style={},
        )


def test_stage_a_diversity_trajectory_has_nonzero_xlim(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input"],
            "tf": ["TF_A"],
            "tfbs_sequence": ["AAAA"],
            "tfbs_core": ["AAA"],
            "best_hit_score": [2.0],
            "tier": [0],
            "rank_within_regulator": [1],
            "selection_rank": [1],
            "nearest_selected_similarity": [None],
            "selection_score_norm": [1.0],
            "nearest_selected_distance_norm": [None],
        }
    )
    fig, _axes_left, axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        x_min, x_max = axes_right[0].get_xlim()
        assert x_max > x_min
    finally:
        fig.clf()


def test_stage_a_diversity_nn_ticks_every_other(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4],
        }
    )
    fig, axes_left, _axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        ticks = axes_left[0].get_xticks()
        assert list(ticks) == [0.0, 2.0]
    finally:
        fig.clf()


def test_stage_a_diversity_y_tick_size_matches_xlabel(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4],
        }
    )
    fig, axes_left, _axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        fig.canvas.draw()
        x_label_size = axes_left[-1].xaxis.label.get_size()
        y_labels = [label for label in axes_left[0].yaxis.get_ticklabels() if label.get_text()]
        assert y_labels
        assert y_labels[0].get_size() == x_label_size
    finally:
        fig.clf()


def test_stage_a_strata_omits_retained_cutoff_label(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
        }
    )
    fig, axes_left, _ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        header_axes = [ax for ax in fig.axes if ax.get_label() == "header"]
        assert not header_axes
        labels = [text.get_text() for ax in axes_left for text in ax.texts]
        assert "Retained cutoff" not in labels
    finally:
        fig.clf()


def test_stage_a_diversity_axis_labels(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4],
        }
    )
    fig, axes_left, axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        header_axes = [ax for ax in fig.axes if ax.get_label() == "header"]
        assert not header_axes
        assert axes_left[-1].get_xlabel() == "Pairwise Hamming NN"
        assert axes_right[-1].get_xlabel() == "MMR selection step"
    finally:
        fig.clf()


def test_stage_a_diversity_selection_trajectory_legend_bottom_left(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    sampling["eligible_score_hist"][0]["selection_pool_min_score_norm_used"] = 0.85
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
            "selection_rank": [1, 2],
            "nearest_selected_similarity": [0.0, 0.5],
            "selection_score_norm": [1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4],
        }
    )
    fig, _axes_left, axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        legend = axes_right[0].get_legend()
        assert legend is not None
        assert legend._loc == 3
        assert legend.get_texts()
        assert legend.get_texts()[0].get_fontsize() > 8.5
        labels = [text.get_text() for text in legend.get_texts()]
        assert any(label == "Score vs max" for label in labels)
        assert not any("τ=" in label for label in labels)
        twin_axes = [
            ax
            for ax in fig.axes
            if ax not in axes_right and ax not in _axes_left and ax.yaxis.get_ticks_position() == "right"
        ]
        assert twin_axes
        y_ticks = twin_axes[0].yaxis.get_major_ticks()
        assert y_ticks
        assert y_ticks[0].get_pad() <= 1.0
    finally:
        fig.clf()


def test_run_health_dashboard_titles_and_layout() -> None:
    matplotlib.use("Agg", force=True)
    attempts = _attempts_df()
    attempts = pd.concat(
        [
            attempts,
            pd.DataFrame(
                [
                    {
                        "attempt_index": 4,
                        "created_at": "2026-01-26T00:00:30+00:00",
                        "status": "ok",
                        "reason": "ok",
                        "plan_name": "alt_plan",
                        "sampling_library_index": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 3, "alt_plan": 1},
    )
    try:
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == (
            "Run diagnostics: solver-step outcomes, waste prevalence, failure reasons, and quota progress"
        )
        assert axes["outcome"].get_title() == "Solver outcomes across plan rows"
        assert axes["dup"].get_title() == "Waste prevalence over solver sequence"
        assert axes["fail"].get_title() == "Rejected/failed reason composition"
        assert axes["plan"] is not None
        assert axes["plan"].get_title() == "Quota attainment by plan"
        assert axes["plan"].get_ylabel() == "Accepted / quota"
        assert axes["outcome"].figure.get_constrained_layout()
        assert axes["fail"].get_box_aspect() == pytest.approx(1.0)
        assert axes["plan"].get_box_aspect() == pytest.approx(1.0)
        assert any(
            "Solver outcomes by step for each subsampled plan." in text.get_text() for text in axes["outcome"].texts
        )
        assert any("Failure reasons" in text.get_text() for text in axes["fail"].texts)
        assert any(
            "Cumulative accepted libraries relative to each plan quota." in text.get_text()
            for text in axes["plan"].texts
        )

        fig.canvas.draw()
        outcome_ticklabels = [tick.get_text() for tick in axes["outcome"].get_xticklabels() if tick.get_visible()]
        dup_ticklabels = [tick.get_text() for tick in axes["dup"].get_xticklabels() if tick.get_visible()]
        assert any(label for label in outcome_ticklabels)
        assert any(label for label in dup_ticklabels)
        assert axes["outcome"].get_shared_x_axes().joined(axes["outcome"], axes["dup"])

        connectors = [artist for artist in fig.artists if isinstance(artist, ConnectionPatch)]
        assert connectors
        assert all(connector.get_linestyle() == "--" for connector in connectors)
        assert all(connector.get_linewidth() <= 0.7 for connector in connectors)
        assert axes["outcome"].collections
        for coll in axes["outcome"].collections:
            widths = np.asarray(coll.get_linewidths(), dtype=float)
            if widths.size > 0:
                assert np.all(widths > 0.0)
    finally:
        fig.clf()
