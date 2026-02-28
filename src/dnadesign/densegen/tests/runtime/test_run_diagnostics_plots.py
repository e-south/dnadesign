"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_diagnostics_plots.py

Coverage for the canonical DenseGen plot set after the scalability refactor.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from dnadesign.densegen.src.core.artifacts.pool import TFBSPoolArtifact
from dnadesign.densegen.src.viz.plot_run import (
    _aggregate_reason_pareto,
    _build_run_health_compression_ratio_figure,
    _build_run_health_detail_figure,
    _build_run_health_figure,
    _build_run_health_outcomes_figure,
    _build_run_health_tfbs_length_by_regulator_figure,
    _build_tfbs_usage_breakdown_figure,
    _extract_plan_quotas,
    _progress_axis,
    _rate_series_from_counts,
    _render_run_health_summary_table_figure,
    _usage_category_label,
    plot_run_health,
    plot_tfbs_usage,
)
from dnadesign.densegen.src.viz.plot_stage_a_diversity import _build_stage_a_diversity_figure
from dnadesign.densegen.src.viz.plot_stage_a_strata import _build_stage_a_strata_overview_figure
from dnadesign.densegen.src.viz.plot_stage_a_yield import _build_stage_a_yield_bias_figure
from dnadesign.densegen.src.viz.plot_stage_b_placement import (
    _allocation_summary_lines,
    _build_occupancy,
    _build_tfbs_count_records,
    _category_display_label,
    _placement_bounds,
    _promoter_constraints,
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
                "densegen__compression_ratio": 0.68,
            },
            {
                "id": "s2",
                "sequence": "TTGACAGGGGTATAATCCCC",
                "densegen__input_name": PLAN_POOL_LABEL,
                "densegen__plan": "demo_plan",
                "densegen__compression_ratio": 0.74,
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
    assert set(cols) == {
        "densegen__compression_ratio",
        "densegen__input_name",
        "densegen__plan",
        "id",
        "sequence",
    }


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

    with pytest.raises(ValueError, match="records.parquet not found"):
        _load_dense_arrays(tmp_path)


def test_plot_run_health(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "run_health.png"
    paths = plot_run_health(
        _dense_arrays_df(),
        out_path,
        attempts_df=_attempts_df(),
        composition_df=_composition_df(),
        events_df=_events_df(),
        cfg={"config": {"generation": {"plan": [{"name": "demo_plan", "quota": 12}]}}},
        style={},
    )
    assert len(paths) == 5
    rel_paths = {str(Path(path).relative_to(tmp_path)) for path in paths}
    assert "run_health/outcomes_over_time.png" in rel_paths
    assert "run_health/run_health.png" in rel_paths
    assert "run_health/compression_ratio_distribution.png" in rel_paths
    assert "run_health/tfbs_length_by_regulator.png" in rel_paths
    assert "run_health/summary_table.png" in rel_paths
    assert (tmp_path / "run_health" / "outcomes_over_time.png").exists()
    assert (tmp_path / "run_health" / "run_health.png").exists()
    assert (tmp_path / "run_health" / "compression_ratio_distribution.png").exists()
    assert (tmp_path / "run_health" / "tfbs_length_by_regulator.png").exists()
    assert (tmp_path / "run_health" / "summary_table.png").exists()
    assert not (tmp_path / "run_health" / "run_health_detail.png").exists()
    assert (tmp_path / "run_health" / "summary.csv").exists()


def test_plot_run_health_supports_effective_config_sequences(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "run_health_sequences.png"
    plan_a = "sigma70_panel__sig35=a__sig10=A"
    plan_b = "sigma70_panel__sig35=a__sig10=B"

    attempts = _attempts_df().copy()
    attempts["plan_name"] = [plan_a, plan_b, plan_a]

    dense_arrays = _dense_arrays_df().copy()
    dense_arrays["densegen__plan"] = [plan_a if idx % 2 == 0 else plan_b for idx in range(len(dense_arrays))]

    composition = _composition_df().copy()
    composition["plan_name"] = [plan_a if idx % 2 == 0 else plan_b for idx in range(len(composition))]

    paths = plot_run_health(
        dense_arrays,
        out_path,
        attempts_df=attempts,
        composition_df=composition,
        events_df=_events_df(),
        cfg={
            "config": {
                "generation": {
                    "plan": [
                        {"name": plan_a, "sequences": 2},
                        {"name": plan_b, "sequences": 1},
                    ]
                }
            }
        },
        style={},
    )
    assert len(paths) == 5
    assert (tmp_path / "run_health" / "run_health.png").exists()


def test_run_health_compression_ratio_distribution_uses_plan_hue() -> None:
    dense_arrays = pd.DataFrame(
        [
            {"densegen__plan": "plan_a", "densegen__compression_ratio": 0.42},
            {"densegen__plan": "plan_a", "densegen__compression_ratio": 0.55},
            {"densegen__plan": "plan_b", "densegen__compression_ratio": 0.71},
            {"densegen__plan": "plan_b", "densegen__compression_ratio": 0.83},
        ]
    )
    fig, axes = _build_run_health_compression_ratio_figure(dense_arrays, style={})
    try:
        ax = axes["compression"]
        fig_w, fig_h = fig.get_size_inches()
        assert fig_w == pytest.approx(7.2)
        assert fig_h == pytest.approx(4.0)
        assert ax.get_xlabel() == "Compression ratio"
        assert ax.get_ylabel() == "Count"
        assert ax.get_title() == "Compression ratio distribution by plan"
        legend = ax.get_legend()
        assert legend is not None
        assert getattr(legend, "_loc", None) == 6
        legend_bbox = legend.get_bbox_to_anchor()._bbox
        assert legend_bbox.x0 >= 1.0
        assert legend_bbox.y0 == pytest.approx(0.5)
        labels = [text.get_text() for text in legend.get_texts()]
        assert any("plan_a" in label for label in labels)
        assert any("plan_b" in label for label in labels)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    finally:
        fig.clf()


def test_run_health_compression_ratio_distribution_avoids_tight_layout_warning_with_many_plans() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(50):
        plan = f"plan_{idx:02d}"
        rows.append({"densegen__plan": plan, "densegen__compression_ratio": 0.40 + (idx % 7) * 0.03})
        rows.append({"densegen__plan": plan, "densegen__compression_ratio": 0.45 + (idx % 7) * 0.03})
    dense_arrays = pd.DataFrame(rows)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig, _axes = _build_run_health_compression_ratio_figure(dense_arrays, style={})
    try:
        assert not any("tight layout not applied" in str(item.message).lower() for item in captured)
        assert not any("constrained_layout not applied" in str(item.message).lower() for item in captured)
    finally:
        fig.clf()


def test_run_health_compression_ratio_distribution_groups_expanded_plan_names() -> None:
    dense_arrays = pd.DataFrame(
        [
            {"densegen__plan": "ethanol__sig35=f", "densegen__compression_ratio": 0.42},
            {"densegen__plan": "ethanol__sig35=b", "densegen__compression_ratio": 0.51},
            {"densegen__plan": "ethanol__sig35=e", "densegen__compression_ratio": 0.56},
            {"densegen__plan": "ciprofloxacin__sig35=f", "densegen__compression_ratio": 0.63},
            {"densegen__plan": "ciprofloxacin__sig35=b", "densegen__compression_ratio": 0.67},
            {"densegen__plan": "ciprofloxacin__sig35=e", "densegen__compression_ratio": 0.71},
        ]
    )
    fig, axes = _build_run_health_compression_ratio_figure(
        dense_arrays,
        style={"run_health_compression_legend_max": 2},
    )
    try:
        ax = axes["compression"]
        assert ax.get_title() == "Compression ratio distribution by plan group"
        legend = ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert any("ethanol" in label for label in labels)
        assert any("ciprofloxacin" in label for label in labels)
    finally:
        fig.clf()


def test_run_health_tfbs_length_by_regulator_groups_lengths() -> None:
    composition = pd.DataFrame(
        [
            {"tf": "TF_A", "tfbs": "AAAA", "length": 4},
            {"tf": "TF_A", "tfbs": "AAAAT", "length": 5},
            {"tf": "TF_A", "tfbs": "AAAAT", "length": 5},
            {"tf": "TF_B", "tfbs": "CCCC", "length": 4},
            {"tf": "fixed:sigma70:-35", "tfbs": "TTGACA", "length": 6},
        ]
    )
    library_members = pd.DataFrame(
        [
            {"tf": "TF_A", "tfbs": "AAAA"},
            {"tf": "TF_A", "tfbs": "AAAAT"},
            {"tf": "TF_A", "tfbs": "AAAATT"},
            {"tf": "TF_B", "tfbs": "CCCC"},
            {"tf": "TF_B", "tfbs": "CCCCC"},
        ]
    )
    fig, axes = _build_run_health_tfbs_length_by_regulator_figure(
        composition,
        style={},
        library_members_df=library_members,
    )
    try:
        ax = axes["length"]
        assert ax.get_xlabel() == "Regulator"
        assert ax.get_ylabel() == "Count in accepted outputs"
        assert ax.get_title() == "TFBS length counts by regulator across accepted outputs"
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert any("TF_A" in label for label in labels)
        assert any("TF_B" in label for label in labels)
        assert any("\n(n=" in label for label in labels)
        assert not any("fixed:" in label for label in labels)
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert "4 bp" in legend_labels
        assert "5 bp" in legend_labels
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    finally:
        fig.clf()


def test_run_health_tfbs_length_single_regulator_uses_length_axis() -> None:
    composition = pd.DataFrame(
        [
            {"tf": "TF_A", "tfbs": "AAAA", "length": 4},
            {"tf": "TF_A", "tfbs": "AAAAT", "length": 5},
            {"tf": "TF_A", "tfbs": "AAAAT", "length": 5},
        ]
    )
    library_members = pd.DataFrame(
        [
            {"tf": "TF_A", "tfbs": "AAAA"},
            {"tf": "TF_A", "tfbs": "AAAAT"},
            {"tf": "TF_A", "tfbs": "AAAATT"},
        ]
    )
    fig, axes = _build_run_health_tfbs_length_by_regulator_figure(
        composition,
        style={},
        library_members_df=library_members,
    )
    try:
        ax = axes["length"]
        assert ax.get_xlabel() == "TFBS length (bp)"
        assert ax.get_ylabel() == "Count in accepted outputs"
        assert ax.get_title() == "TFBS length distribution across accepted outputs"
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(size[1], rel=0.01)
        assert float(size[0]) <= 4.8
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert any(label == "4" for label in labels)
        assert any(label == "5" for label in labels)
        assert ax.get_legend() is None
        patches = ax.patches
        assert patches
        face_color = patches[0].get_facecolor()
        assert float(face_color[0]) > 0.3
        assert float(face_color[1]) > 0.3
        assert float(face_color[2]) > 0.3
    finally:
        fig.clf()


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
        assert "Accepted" in legend_labels
        assert "Rejected" in legend_labels
        assert "Failed" in legend_labels
        assert "duplicate" not in legend_labels
        assert ax.get_xlabel() == "Plan"
        assert ax.get_ylabel() == "Attempt index"
        assert ax._left_title.get_text() == ""
        label_size = ax.xaxis.label.get_size()
        assert all(text.get_size() == pytest.approx(label_size) for text in legend.get_texts())
        assert legend.get_bbox_to_anchor()._bbox.x0 >= 1.0
        assert len(ax.get_lines()) == 0
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
        assert ax.get_title() == "Attempt outcomes by plan"
        assert ax._left_title.get_text() == ""
        assert ax.get_xlabel() == "Plan"
        assert ax.get_ylabel() == "Attempt index"
        assert all("Rejected/failed reason composition" not in t.get_text() for t in ax.texts)
        assert all("Quota attainment by plan" not in t.get_text() for t in ax.texts)
        legend = ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["Accepted", "Rejected", "Failed"]
        y_labels = [text.get_text().strip() for text in ax.get_yticklabels() if text.get_text().strip()]
        assert y_labels
        y_lim = ax.get_ylim()
        assert float(y_lim[0]) > float(y_lim[1])
        assert len(ax.get_lines()) == 0
        assert ax.get_aspect() == pytest.approx(1.0)
        assert ax.patches
        assert len(ax.patches) == 2
        widths = [float(patch.get_width()) for patch in ax.patches]
        heights = [float(patch.get_height()) for patch in ax.patches]
        assert widths
        assert all(width == pytest.approx(height, rel=1e-6) for width, height in zip(widths, heights))
        assert all(0.8 <= width <= 1.0 for width in widths)
        assert all(float(p.get_linewidth()) == pytest.approx(0.0) for p in ax.patches)
        assert ax.collections
        marker_offsets = [coll.get_offsets() for coll in ax.collections if len(coll.get_offsets()) > 0]
        assert marker_offsets
        first_offsets = np.asarray(marker_offsets[0], dtype=float)
        assert any(
            float(point[0]) == pytest.approx(2.5, abs=1e-6) and float(point[1]) == pytest.approx(0.5, abs=1e-6)
            for point in first_offsets
        )
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
        assert len(axes["outcome"].get_lines()) == 0
    finally:
        fig.clf()


def test_run_health_outcomes_points_follow_actual_run_order() -> None:
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
        style={"run_health_outcomes_attempts_per_row": 2},
        plan_quotas={"plan_a": 12, "plan_b": 12},
    )
    try:
        patches = axes["outcome"].patches
        assert len(patches) == 4
        centers = [
            (float(patch.get_x() + patch.get_width() / 2.0), float(patch.get_y() + patch.get_height() / 2.0))
            for patch in patches
        ]
        y_values = {round(y, 3) for _, y in centers}
        assert y_values == {0.5}
        x_values = sorted(round(x, 3) for x, _ in centers)
        assert x_values == [0.5, 1.5, 2.5, 3.5]
    finally:
        fig.clf()


def test_run_health_outcomes_tiles_attempts_by_plan_row() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": idx + 1,
                "created_at": f"2026-01-26T00:00:{idx:02d}+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_a",
                "sampling_library_index": idx + 1,
            }
            for idx in range(12)
        ]
    )
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={"run_health_outcomes_attempts_per_row": 10},
        plan_quotas={"plan_a": 12},
    )
    try:
        patches = axes["outcome"].patches
        assert len(patches) == 12
        centers = [
            (float(patch.get_x() + patch.get_width() / 2.0), float(patch.get_y() + patch.get_height() / 2.0))
            for patch in patches
        ]
        row_one = sorted(x for x, y in centers if y == pytest.approx(0.5))
        row_two = sorted(x for x, y in centers if y == pytest.approx(1.5))
        assert len(row_one) == 10
        assert len(row_two) == 2
        assert row_one[0] == pytest.approx(0.5)
        assert row_one[-1] == pytest.approx(9.5)
        assert row_two == pytest.approx([0.5, 1.5])
        tick_labels = [tick.get_text().strip() for tick in axes["outcome"].get_yticklabels() if tick.get_text().strip()]
        assert "1" in tick_labels
        assert "11" in tick_labels
    finally:
        fig.clf()


def test_run_health_outcomes_xticks_use_regular_spacing() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": idx + 1,
                "created_at": f"2026-01-26T00:00:{idx:02d}+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_a",
                "sampling_library_index": value,
            }
            for idx, value in enumerate([1, 2, 3, 4, 8, 12, 16, 24, 28, 32])
        ]
    )
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"plan_a": 20},
    )
    try:
        ticks = axes["outcome"].get_xticks()
        assert len(ticks) == 1
        labels = [tick.get_text() for tick in axes["outcome"].get_xticklabels()]
        assert labels
        assert "Plan_a" in labels[0]
    finally:
        fig.clf()


def test_run_health_outcomes_auto_groups_expanded_plans() -> None:
    matplotlib.use("Agg", force=True)
    attempts_rows = []
    quotas: dict[str, int] = {}
    for idx in range(20):
        if idx < 10:
            plan_name = f"sigma70_panel__sig35={idx % 5}__sig10={idx % 2}"
        else:
            plan_name = f"sigma70_topup__sig35={idx % 5}__sig10={idx % 2}"
        attempts_rows.append(
            {
                "attempt_index": idx + 1,
                "created_at": f"2026-01-26T00:00:{idx:02d}+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": plan_name,
                "sampling_library_index": idx + 1,
            }
        )
        quotas[plan_name] = 1
    attempts = pd.DataFrame(attempts_rows)
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={"run_health_outcomes_plan_scope": "auto", "run_health_outcomes_plan_max_labels": 6},
        plan_quotas=quotas,
    )
    try:
        labels = [tick.get_text() for tick in axes["outcome"].get_xticklabels()]
        assert len(labels) == 2
        assert any("Sigma70_panel" in label for label in labels)
        assert any("Sigma70_topup" in label for label in labels)
    finally:
        fig.clf()


def test_run_health_outcomes_defaults_to_parent_plan_ticks() -> None:
    matplotlib.use("Agg", force=True)
    attempts_rows = []
    quotas: dict[str, int] = {}
    for idx in range(12):
        if idx < 6:
            plan_name = f"sigma70_panel__sig35={idx % 3}__sig10={idx % 2}"
        else:
            plan_name = f"sigma70_topup__sig35={idx % 3}__sig10={idx % 2}"
        attempts_rows.append(
            {
                "attempt_index": idx + 1,
                "created_at": f"2026-01-26T00:00:{idx:02d}+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": plan_name,
                "sampling_library_index": idx + 1,
            }
        )
        quotas[plan_name] = quotas.get(plan_name, 0) + 1
    attempts = pd.DataFrame(attempts_rows)
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas=quotas,
    )
    try:
        labels = [tick.get_text() for tick in axes["outcome"].get_xticklabels()]
        assert len(labels) == 2
        assert any("Sigma70_panel" in label for label in labels)
        assert any("Sigma70_topup" in label for label in labels)
    finally:
        fig.clf()


def test_run_health_outcomes_per_plan_scope_keeps_expanded_plan_ticks() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": 1,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "sigma70_panel__sig35=a__sig10=A",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 2,
                "created_at": "2026-01-26T00:00:01+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "sigma70_panel__sig35=a__sig10=B",
                "sampling_library_index": 2,
            },
        ]
    )
    fig, axes = _build_run_health_outcomes_figure(
        attempts,
        events_df=None,
        style={"run_health_outcomes_plan_scope": "per_plan"},
        plan_quotas={
            "sigma70_panel__sig35=a__sig10=A": 1,
            "sigma70_panel__sig35=a__sig10=B": 1,
        },
    )
    try:
        labels = [tick.get_text() for tick in axes["outcome"].get_xticklabels()]
        assert len(labels) == 2
        assert all("Sigma70_panel" in label for label in labels)
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
        assert axes["fail"].get_xlim()[1] >= 5.0
        assert all(float(tick).is_integer() for tick in axes["fail"].get_xticks())
        y0, y1 = axes["fail"].get_ylim()
        low, high = min(y0, y1), max(y0, y1)
        assert low < -0.5
        assert high > (len(axes["fail"].get_yticks()) - 0.5)
        assert not axes["fail"].texts
        assert fig.legends
        assert fig.legends[0]._ncols == 2
        assert axes["plan"].get_title() == "Quota progress"
        quota_text = next(text for text in axes["plan"].texts if "Quota (" in text.get_text())
        assert quota_text.get_ha() == "left"
        assert quota_text.get_position()[0] <= 0.1
        assert not axes["fail"].patches
        assert axes["fail"].collections
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fail_title_bbox = (
            axes["fail"].title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        )
        plan_title_bbox = (
            axes["plan"].title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        )
        assert (1.0 - fail_title_bbox.y1) >= 0.05
        assert (1.0 - plan_title_bbox.y1) >= 0.05
    finally:
        fig.clf()


def test_run_health_detail_reason_labels_start_capitalized() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = "postprocess_forbidden_kmer"

    fig, axes = _build_run_health_detail_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        labels = [tick.get_text() for tick in axes["fail"].get_yticklabels() if tick.get_text()]
        assert labels
        assert all(label[0] == label[0].upper() for label in labels)
    finally:
        fig.clf()


def test_run_health_summary_table_uses_tighter_vertical_save_padding(monkeypatch, tmp_path: Path) -> None:
    import dnadesign.densegen.src.viz.plot_run as plot_run_module

    summary_df = pd.DataFrame(
        [
            {"scope": "run", "name": "attempts", "value": 10, "unit": "count"},
            {"scope": "run", "name": "ok", "value": 7, "unit": "count"},
        ]
    )
    observed: dict[str, object] = {}

    def _fake_save_figure(fig, path, *, style=None):
        observed["path"] = Path(path)
        observed["style"] = dict(style or {})

    monkeypatch.setattr(plot_run_module, "_save_figure", _fake_save_figure)
    _render_run_health_summary_table_figure(summary_df, tmp_path / "summary_table.png", style={})

    assert observed["path"] == tmp_path / "summary_table.png"
    style = observed.get("style")
    assert isinstance(style, dict)
    assert float(style.get("save_pad_inches", 1.0)) <= 0.04


def test_run_health_detail_legend_wraps_for_many_plans() -> None:
    matplotlib.use("Agg", force=True)
    attempts_rows = []
    quotas: dict[str, int] = {}
    for idx in range(12):
        plan_name = f"plan_{idx:02d}"
        attempts_rows.append(
            {
                "attempt_index": idx * 2 + 1,
                "created_at": f"2026-01-26T00:00:{idx:02d}+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": plan_name,
                "sampling_library_index": idx + 1,
            }
        )
        attempts_rows.append(
            {
                "attempt_index": idx * 2 + 2,
                "created_at": f"2026-01-26T00:01:{idx:02d}+00:00",
                "status": "failed",
                "reason": "no_solution",
                "plan_name": plan_name,
                "sampling_library_index": idx + 1,
            }
        )
        quotas[plan_name] = 2
    attempts = pd.DataFrame(attempts_rows)
    fig, _axes = _build_run_health_detail_figure(
        attempts,
        events_df=None,
        style={"run_health_plan_scope": "per_plan"},
        plan_quotas=quotas,
    )
    try:
        assert fig.legends
        assert fig.legends[0]._ncols <= 4
    finally:
        fig.clf()


def test_run_health_quota_progress_legend_encodes_plan_color_and_marker() -> None:
    matplotlib.use("Agg", force=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_index": 1,
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_a",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 2,
                "created_at": "2026-01-26T00:00:10+00:00",
                "status": "ok",
                "reason": "ok",
                "plan_name": "plan_b",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 3,
                "created_at": "2026-01-26T00:00:20+00:00",
                "status": "failed",
                "reason": "no_solution",
                "plan_name": "plan_a",
                "sampling_library_index": 1,
            },
            {
                "attempt_index": 4,
                "created_at": "2026-01-26T00:00:30+00:00",
                "status": "failed",
                "reason": "no_solution",
                "plan_name": "plan_b",
                "sampling_library_index": 1,
            },
        ]
    )
    fig, axes = _build_run_health_detail_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"plan_a": 1, "plan_b": 2},
    )
    try:
        assert fig.legends
        legend = fig.legends[0]
        handles = [handle for handle in getattr(legend, "legend_handles", []) if isinstance(handle, Line2D)]
        assert len(handles) == 2
        marker_symbols = [str(handle.get_marker()) for handle in handles]
        assert all(marker not in {"None", "", " "} for marker in marker_symbols)
        assert len(set(marker_symbols)) == 2
        collections = axes["plan"].collections
        assert len(collections) == 2
        marker_x = sorted(float(np.asarray(coll.get_offsets())[-1][0]) for coll in collections)
        assert marker_x == [3.0, 4.0]
    finally:
        fig.clf()


def test_stage_a_strata_tier_marker_font_is_larger(tmp_path: Path) -> None:
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
        percent_labels = [text for text in axes_left[0].texts if "%" in text.get_text()]
        assert percent_labels
        assert min(text.get_fontsize() for text in percent_labels) >= 7.8
    finally:
        fig.clf()


def test_stage_a_yield_stepwise_x_ticks_are_slightly_smaller(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = manifest.entry_for("demo_input").stage_a_sampling
    assert sampling is not None
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAT"],
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
    fig, axes_left, _axes_right, _cbar_ax = _build_stage_a_yield_bias_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        fig.canvas.draw()
        x_tick_labels = [tick for tick in axes_left[-1].get_xticklabels() if tick.get_text()]
        assert x_tick_labels
        assert x_tick_labels[0].get_fontsize() <= 10.9
    finally:
        fig.clf()


def test_stage_a_yield_right_column_shows_x_labels_only_on_bottom_row(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = copy.deepcopy(manifest.entry_for("demo_input").stage_a_sampling)
    assert sampling is not None
    assert sampling["eligible_score_hist"]
    second = copy.deepcopy(sampling["eligible_score_hist"][0])
    second["regulator"] = "TF_B"
    sampling["eligible_score_hist"].append(second)
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 4,
            "tf": ["TF_A", "TF_A", "TF_B", "TF_B"],
            "tfbs_sequence": ["AAAA", "AAAT", "CCCC", "CCCA"],
            "tfbs_core": ["AAAA", "AAAT", "CCCC", "CCCA"],
            "best_hit_score": [2.0, 1.5, 2.1, 1.6],
            "tier": [0, 1, 0, 1],
            "rank_within_regulator": [1, 2, 1, 2],
            "selection_rank": [1, 2, 1, 2],
            "nearest_selected_similarity": [0.0, 0.5, 0.0, 0.5],
            "selection_score_norm": [1.0, 0.5, 1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4, 0.2, 0.4],
        }
    )
    fig, _axes_left, axes_right, _cbar_ax = _build_stage_a_yield_bias_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        assert axes_right[0].get_xlabel() == ""
        assert axes_right[-1].get_xlabel() == "Core position"
        top_tick_text = [tick.get_text() for tick in axes_right[0].get_xticklabels() if tick.get_visible()]
        assert all(text == "" for text in top_tick_text)
    finally:
        fig.clf()


def test_stage_a_diversity_score_vs_max_ylabel_is_closer_to_axis(tmp_path: Path) -> None:
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
    fig, _axes_left, axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        right_bbox = axes_right[0].get_position()
        score_label = next(text for text in fig.texts if text.get_text() == "Score vs max")
        assert (score_label.get_position()[0] - right_bbox.x1) <= 0.045
    finally:
        fig.clf()


def test_stage_a_diversity_shows_x_labels_only_on_bottom_row(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    manifest = _pool_manifest(tmp_path, include_diversity=True)
    sampling = copy.deepcopy(manifest.entry_for("demo_input").stage_a_sampling)
    assert sampling is not None
    assert sampling["eligible_score_hist"]
    second = copy.deepcopy(sampling["eligible_score_hist"][0])
    second["regulator"] = "TF_B"
    sampling["eligible_score_hist"].append(second)
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 4,
            "tf": ["TF_A", "TF_A", "TF_B", "TF_B"],
            "tfbs_sequence": ["AAAA", "AAAT", "CCCC", "CCCA"],
            "tfbs_core": ["AAAA", "AAAT", "CCCC", "CCCA"],
            "best_hit_score": [2.0, 1.5, 2.1, 1.6],
            "tier": [0, 1, 0, 1],
            "rank_within_regulator": [1, 2, 1, 2],
            "selection_rank": [1, 2, 1, 2],
            "nearest_selected_similarity": [0.0, 0.5, 0.0, 0.5],
            "selection_score_norm": [1.0, 0.5, 1.0, 0.5],
            "nearest_selected_distance_norm": [0.2, 0.4, 0.2, 0.4],
        }
    )
    fig, axes_left, axes_right = _build_stage_a_diversity_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        assert axes_left[0].get_xlabel() == ""
        assert axes_right[0].get_xlabel() == ""
        assert axes_left[-1].get_xlabel() == "Pairwise Hamming NN"
        assert axes_right[-1].get_xlabel() == "MMR selection step"
        top_left_ticks = [tick.get_text() for tick in axes_left[0].get_xticklabels() if tick.get_visible()]
        top_right_ticks = [tick.get_text() for tick in axes_right[0].get_xticklabels() if tick.get_visible()]
        assert all(text == "" for text in top_left_ticks)
        assert all(text == "" for text in top_right_ticks)
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


def test_extract_plan_quotas_supports_effective_config_sequences() -> None:
    cfg = {
        "config": {
            "generation": {
                "plan": [
                    {"name": "sigma70_panel__sig35=a__sig10=A", "sequences": 1},
                    {"name": "sigma70_panel__sig35=a__sig10=B", "sequences": "2"},
                ]
            }
        }
    }
    quotas = _extract_plan_quotas(cfg)
    assert quotas == {
        "sigma70_panel__sig35=a__sig10=A": 1,
        "sigma70_panel__sig35=a__sig10=B": 2,
    }


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


def test_run_health_reason_breaks_out_forbidden_kmer_from_detail_json() -> None:
    attempts = _attempts_df().copy()
    attempts.loc[1, "status"] = "rejected"
    attempts.loc[1, "reason"] = "postprocess_forbidden_kmer"
    attempts.loc[1, "detail_json"] = '{"kmer":"ATGC","position":9}'
    fig, axes = _build_run_health_figure(
        attempts,
        events_df=None,
        style={},
        plan_quotas={"demo_plan": 12},
    )
    try:
        labels = [tick.get_text() for tick in axes["fail"].get_yticklabels()]
        assert any("forbidden kmer" in label.lower() and "ATGC" in label for label in labels)
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
    assert len(paths) == 1
    rel = {str(Path(path).relative_to(tmp_path)) for path in paths}
    assert "stage_b/demo_plan/occupancy.png" in rel
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
    assert len(paths) == 1
    for path in paths:
        path = Path(path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_placement_map_label_sanitizer() -> None:
    assert _sanitize_tf_label("lexA_CTGTATAW") == "lexA"
    assert _sanitize_tf_label("cpxR") == "cpxR"
    assert _sanitize_fixed_label("fixed:sigma70_consensus:-35") == "σ70 upstream site (-35)"
    assert _sanitize_fixed_label("fixed:sigma70_consensus:-10") == "σ70 downstream site (-10)"


def test_placement_bounds_uses_final_offset_when_present() -> None:
    row = pd.Series({"offset": 6, "offset_raw": 0, "pad_left": 6, "length": 4})
    assert _placement_bounds(row, seq_len=60) == (6, 10)


def test_placement_bounds_rejects_inconsistent_offset_metadata() -> None:
    row = pd.Series({"offset": 7, "offset_raw": 0, "pad_left": 6, "length": 4})
    with pytest.raises(ValueError, match="offset metadata mismatch"):
        _placement_bounds(row, seq_len=60)


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


def test_tfbs_usage_title_is_human_readable_and_legend_has_no_frame() -> None:
    matplotlib.use("Agg", force=True)
    fig, axes = _build_tfbs_usage_breakdown_figure(
        _composition_df(),
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        style={},
        pools=None,
        library_members_df=_library_members_df(),
    )
    try:
        title = axes["usage"].get_title()
        assert "plan_pool__" not in title
        assert "for input" not in title.lower()
        assert "and plan" not in title.lower()
        assert "distribution" in title.lower()
        assert "heatmap" in axes["cum"].get_title().lower()
        if fig.legends:
            assert fig.legends[0].get_frame_on() is False
            assert min(text.get_fontsize() for text in fig.legends[0].get_texts()) >= 11.0
        summary_blocks = [text for text in axes["usage"].texts if "Placements in outputs" in str(text.get_text())]
        assert summary_blocks
        summary_lines = [line for line in str(summary_blocks[0].get_text()).splitlines() if line.strip()]
        assert summary_lines
        assert all(line[0].isupper() for line in summary_lines)
        summary_font_size = float(summary_blocks[0].get_fontsize())
        assert summary_font_size >= 10.8
        assert summary_font_size <= 11.6
        for text in axes["usage"].texts:
            assert text.get_bbox_patch() is None
    finally:
        fig.clf()


def test_plot_tfbs_usage_saves_with_tight_bbox(monkeypatch, tmp_path: Path) -> None:
    import dnadesign.densegen.src.viz.plot_run_panels as plot_run_panels_module

    class _FakeFigure:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, dict[str, object]]] = []

        def savefig(self, path: Path, **kwargs) -> None:
            self.calls.append((Path(path), dict(kwargs)))

    fake_fig = _FakeFigure()
    monkeypatch.setattr(
        plot_run_panels_module,
        "_build_tfbs_usage_breakdown_figure",
        lambda *_args, **_kwargs: (fake_fig, {}),
    )
    monkeypatch.setattr(plot_run_panels_module.plt, "close", lambda *_args, **_kwargs: None)

    paths = plot_run_panels_module.plot_tfbs_usage(
        pd.DataFrame(),
        tmp_path / "tfbs_usage.png",
        composition_df=_composition_df(),
        style={},
    )
    assert len(paths) == 1
    assert len(fake_fig.calls) == 1
    _path, kwargs = fake_fig.calls[0]
    assert kwargs.get("bbox_inches") == "tight"
    assert float(kwargs.get("pad_inches", 0.0)) > 0.0
    assert kwargs.get("facecolor") == "white"


def test_plot_run_health_saves_with_tight_bbox(monkeypatch, tmp_path: Path) -> None:
    import dnadesign.densegen.src.viz.plot_run as plot_run_module

    class _FakeFigure:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, dict[str, object]]] = []

        def savefig(self, path: Path, **kwargs) -> None:
            self.calls.append((Path(path), dict(kwargs)))

    fake_outcomes = _FakeFigure()
    fake_detail = _FakeFigure()
    fake_compression = _FakeFigure()
    fake_tfbs = _FakeFigure()
    monkeypatch.setattr(
        plot_run_module,
        "_build_run_health_outcomes_figure",
        lambda *_args, **_kwargs: (fake_outcomes, {}),
    )
    monkeypatch.setattr(
        plot_run_module,
        "_build_run_health_detail_figure",
        lambda *_args, **_kwargs: (fake_detail, {}),
    )
    monkeypatch.setattr(
        plot_run_module,
        "_build_run_health_compression_ratio_figure",
        lambda *_args, **_kwargs: (fake_compression, {}),
    )
    monkeypatch.setattr(
        plot_run_module,
        "_build_run_health_tfbs_length_by_regulator_figure",
        lambda *_args, **_kwargs: (fake_tfbs, {}),
    )
    monkeypatch.setattr(plot_run_module.plt, "close", lambda *_args, **_kwargs: None)

    paths = plot_run_module.plot_run_health(
        _dense_arrays_df(),
        tmp_path / "run_health.png",
        attempts_df=_attempts_df(),
        composition_df=_composition_df(),
        events_df=_events_df(),
        cfg={"config": {"generation": {"plan": [{"name": "demo_plan", "quota": 12}]}}},
        style={},
    )
    assert len(paths) == 5
    for fake_figure in (fake_outcomes, fake_detail, fake_compression, fake_tfbs):
        assert len(fake_figure.calls) == 1
        _path, kwargs = fake_figure.calls[0]
        assert kwargs.get("bbox_inches") == "tight"
        assert float(kwargs.get("pad_inches", 0.0)) > 0.0
        assert kwargs.get("facecolor") == "white"


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
        assert axes["usage"].lines
        assert axes["usage"].get_yscale() == "linear"
        assert axes["cum"].images
        expected_regulators = composition["tf"].map(_usage_category_label).nunique()
        heatmap = np.asarray(axes["cum"].images[0].get_array(), dtype=float)
        assert heatmap.shape[0] == expected_regulators
        assert axes["usage"].get_xlabel() == "Global TFBS rank (descending count)"
        assert axes["cum"].get_xlabel() == "TFBS rank within regulator"
        assert axes["cum"].get_ylabel() == ""
        usage_box = axes["usage"].get_position()
        cum_box = axes["cum"].get_position()
        assert abs(float(usage_box.y0) - float(cum_box.y0)) < 0.03
        assert float(abs(usage_box.x0 - cum_box.x0)) > 0.2
        assert axes["usage"].get_box_aspect() == pytest.approx(1.0, rel=0.01)
        assert axes["cum"].get_box_aspect() == pytest.approx(1.0, rel=0.01)
        y_labels = [tick.get_text().strip() for tick in axes["cum"].get_yticklabels() if tick.get_text().strip()]
        assert y_labels
        assert all(label[0].isupper() for label in y_labels if label[0].isalpha())
        legend = axes["usage"].get_legend()
        if legend is not None:
            legend_text = "\n".join(t.get_text() for t in legend.get_texts())
        else:
            assert fig.legends
            legend_text = "\n".join(t.get_text() for t in fig.legends[0].get_texts())
        assert "Fixed:sigma70:-35" in legend_text
        assert len(fig.axes) >= 3
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


def test_grouped_plan_occupancy_keeps_fixed_components() -> None:
    composition = _composition_df().copy()
    dense_arrays = _dense_arrays_df().copy()
    composition["plan_name"] = "demo_plan"
    dense_arrays["densegen__plan"] = "demo_plan"
    cfg_effective = {
        "config": {
            "generation": {
                "sequence_length": 20,
                "plan": [
                    {
                        "name": "demo_plan__sig35=f__sig10=H",
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream_pos": [0, 6],
                                    "downstream_pos": [10, 16],
                                    "upstream": "TTGACA",
                                    "downstream": "TATAAT",
                                    "spacer_length": [4, 4],
                                }
                            ]
                        },
                    },
                    {
                        "name": "demo_plan__sig35=b__sig10=H",
                        "fixed_elements": {
                            "promoter_constraints": [
                                {
                                    "name": "sigma70_core",
                                    "upstream_pos": [0, 6],
                                    "downstream_pos": [10, 16],
                                    "upstream": "CTGACA",
                                    "downstream": "TATAAT",
                                    "spacer_length": [4, 4],
                                }
                            ]
                        },
                    },
                ],
            }
        }
    }
    constraints, aggregate_fixed_components = _promoter_constraints(cfg_effective, "demo_plan")
    assert aggregate_fixed_components is True
    occupancy, categories, missing_counts = _build_occupancy(
        composition,
        solutions=dense_arrays,
        seq_len=20,
        constraints=constraints,
        max_categories=12,
        aggregate_fixed_components=aggregate_fixed_components,
    )
    assert set(missing_counts.keys()).issubset({"promoter"})
    assert "fixed:promoter:-35" in categories
    assert "fixed:promoter:-10" in categories
    assert float(occupancy["fixed:promoter:-35"].sum()) > 0.0
    assert float(occupancy["fixed:promoter:-10"].sum()) > 0.0


def test_build_occupancy_respects_upstream_position_window_for_inferred_fixed_components() -> None:
    seq_chars = list("C" * 60)

    def _write_motif(start: int, motif: str) -> None:
        seq_chars[start : start + len(motif)] = list(motif)

    _write_motif(4, "TTGACA")
    _write_motif(12, "TTGACA")
    _write_motif(26, "TATAAT")
    _write_motif(34, "TATAAT")
    sequence = "".join(seq_chars)

    solutions = pd.DataFrame([{"id": "s1", "sequence": sequence}])
    sub = pd.DataFrame(columns=["tf", "offset", "length", "end", "offset_raw", "pad_left"])
    constraints = [
        {
            "name": "sigma70",
            "upstream": "TTGACA",
            "downstream": "TATAAT",
            "spacer_min": 16,
            "spacer_max": 16,
            "upstream_pos": [10, 25],
            "downstream_pos": None,
        }
    ]

    occupancy, categories, _missing_counts = _build_occupancy(
        sub,
        solutions=solutions,
        seq_len=60,
        constraints=constraints,
        max_categories=12,
        aggregate_fixed_components=False,
    )
    label = "fixed:sigma70:-35"
    assert label in categories
    nonzero_positions = np.flatnonzero(np.asarray(occupancy[label], dtype=float) > 0.0)
    assert nonzero_positions.size > 0
    assert int(nonzero_positions.min()) == 12
    assert int(nonzero_positions.max()) == 17


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
        assert not ax.get_title().endswith(".")
        assert ax.title.get_size() >= 16.0
        assert ax.xaxis.label.get_size() >= 15.0
        assert ax.yaxis.label.get_size() >= 15.0
        tick_sizes = [
            tick.label1.get_fontsize()
            for tick in list(ax.xaxis.get_major_ticks()) + list(ax.yaxis.get_major_ticks())
            if tick.label1.get_text() != ""
        ]
        assert tick_sizes
        assert min(tick_sizes) >= 14.0
        assert len(ax.patches) > 0
        assert fig.legends
        legend = fig.legends[0]
        assert 1 <= int(getattr(legend, "_ncols", 0)) <= len(categories)
        assert legend.get_frame_on() is False
        assert min(text.get_fontsize() for text in legend.get_texts()) >= 12.0
        legend_text = "\n".join(text.get_text() for text in legend.get_texts())
        assert "σ70 upstream site (-35)" in legend_text
        assert "σ70 downstream site (-10)" in legend_text
        renderer = fig.canvas.get_renderer()
        xlab_bbox = ax.xaxis.get_label().get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        assert legend_bbox.y1 < xlab_bbox.y0
        assert fig.subplotpars.bottom >= 0.28
        x_min, x_max = ax.get_xlim()
        assert x_min < 0.0
        assert x_max > float(seq_len)
        nonzero = [patch for patch in ax.patches if float(patch.get_height()) > 0.0]
        assert nonzero
        assert all(float(patch.get_edgecolor()[3]) == pytest.approx(1.0, abs=1e-6) for patch in nonzero)
        pairs = []
        for patch in nonzero:
            face = patch.get_facecolor()
            edge = patch.get_edgecolor()
            face_luma = (0.2126 * float(face[0])) + (0.7152 * float(face[1])) + (0.0722 * float(face[2]))
            edge_luma = (0.2126 * float(edge[0])) + (0.7152 * float(edge[1])) + (0.0722 * float(edge[2]))
            if face_luma > 0.02:
                pairs.append((edge_luma, face_luma))
        assert pairs
        assert all(edge_luma <= face_luma for edge_luma, face_luma in pairs)
        assert any(edge_luma < face_luma for edge_luma, face_luma in pairs)
    finally:
        fig.clf()


def test_occupancy_low_count_categories_render_on_top() -> None:
    matplotlib.use("Agg", force=True)
    seq_len = 12
    occupancy = {
        "TF_HIGH": np.full(seq_len, 4.0),
        "TF_LOW": np.full(seq_len, 1.0),
    }
    categories = list(occupancy.keys())

    fig, ax = _render_occupancy(
        occupancy,
        categories,
        seq_len=seq_len,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        n_solutions=3,
        alpha=0.22,
        fixed_label_sequences={},
        style={},
    )
    try:
        high_z = max(float(patch.get_zorder()) for patch in ax.patches if abs(float(patch.get_height()) - 4.0) <= 1e-6)
        low_z = max(float(patch.get_zorder()) for patch in ax.patches if abs(float(patch.get_height()) - 1.0) <= 1e-6)
        assert low_z > high_z
    finally:
        fig.clf()


def test_occupancy_legend_wraps_when_entries_are_too_wide() -> None:
    matplotlib.use("Agg", force=True)
    seq_len = 20
    categories = [f"TF_{idx}_LONG_LABEL_FOR_OCCUPANCY_LEGEND" for idx in range(8)]
    occupancy = {label: np.ones(seq_len) for label in categories}

    fig, _ax = _render_occupancy(
        occupancy,
        categories,
        seq_len=seq_len,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        n_solutions=2,
        alpha=0.22,
        fixed_label_sequences={},
        style={},
    )
    try:
        fig.canvas.draw()
        assert fig.legends
        legend = fig.legends[0]
        assert getattr(legend, "_ncols", 0) < len(categories)
    finally:
        fig.clf()


def test_occupancy_legend_orders_background_then_sigma_sites() -> None:
    matplotlib.use("Agg", force=True)
    seq_len = 20
    occupancy = {
        "TF_A": np.ones(seq_len),
        "background": np.ones(seq_len),
        "fixed:sigma70:-35": np.ones(seq_len),
        "fixed:sigma70:-10": np.ones(seq_len),
    }
    categories = list(occupancy.keys())
    fixed_label_sequences = {
        "fixed:sigma70:-35": "TTGACA",
        "fixed:sigma70:-10": "TATAAT",
    }

    fig, _ax = _render_occupancy(
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
        assert fig.legends
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert labels[0] == "Background"
        assert labels[1].startswith("σ70 upstream site (-35)")
        assert labels[2].startswith("σ70 downstream site (-10)")
    finally:
        fig.clf()


def test_occupancy_legend_capitalizes_non_sigma_fixed_labels() -> None:
    matplotlib.use("Agg", force=True)
    seq_len = 20
    occupancy = {
        "background": np.ones(seq_len),
        "fixed:promoter:-35": np.ones(seq_len),
        "fixed:promoter:-10": np.ones(seq_len),
    }
    categories = list(occupancy.keys())
    fig, _ax = _render_occupancy(
        occupancy,
        categories,
        seq_len=seq_len,
        input_name=PLAN_POOL_LABEL,
        plan_name="demo_plan",
        n_solutions=2,
        alpha=0.22,
        fixed_label_sequences={},
        style={},
    )
    try:
        fig.canvas.draw()
        assert fig.legends
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert "Background" in labels
        assert any(label.startswith("Promoter upstream site (-35)") for label in labels)
        assert any(label.startswith("Promoter downstream site (-10)") for label in labels)
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
        == "σ70 upstream site (-35) (TTGACA)"
    )
    assert (
        _category_display_label("fixed:sigma70_consensus:-10", fixed_label_sequences=fixed_label_sequences)
        == "σ70 downstream site (-10) (TATAAT)"
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
        assert axes["cum"].get_ylim()[1] > 1.0
        assert axes["rank"].get_ylim()[1] > float(counts["count"].max())
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


def test_plot_stage_a_summary_background_logo_filename_avoids_double_background(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary_bg_name.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["background"] * 2,
            "tf": ["background"] * 2,
            "tfbs": ["AAAA", "AAAT"],
            "tfbs_core": ["AAAA", "AAAT"],
            "tfbs_id": ["a", "b"],
        }
    )
    pools = {"background": pool_df}
    pools_dir = tmp_path / "pools_bg_name"
    pools_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pools_dir / "pool_manifest.json"
    manifest_payload = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "background",
                "type": "background_pool",
                "pool_path": "background__pool.parquet",
                "rows": 2,
                "columns": ["input_name", "tf", "tfbs", "tfbs_core", "tfbs_id"],
                "pool_mode": "tfbs",
                "stage_a_sampling": None,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    manifest = TFBSPoolArtifact.load(manifest_path)
    paths = plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )
    assert paths
    assert Path(paths[0]).name == "background_logo.png"


def test_plot_stage_a_summary_background_logo_uses_compact_size_and_nonredundant_title(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary_bg_style.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["background"] * 4,
            "tf": ["background"] * 4,
            "tfbs": [
                "A" * 60,
                "C" * 60,
                "G" * 60,
                "T" * 60,
            ],
            "tfbs_core": [
                "A" * 60,
                "C" * 60,
                "G" * 60,
                "T" * 60,
            ],
            "tfbs_id": ["a", "b", "c", "d"],
        }
    )
    pools = {"background": pool_df}
    pools_dir = tmp_path / "pools_bg_style"
    pools_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pools_dir / "pool_manifest.json"
    manifest_payload = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "background",
                "type": "background_pool",
                "pool_path": "background__pool.parquet",
                "rows": 4,
                "columns": ["input_name", "tf", "tfbs", "tfbs_core", "tfbs_id"],
                "pool_mode": "tfbs",
                "stage_a_sampling": None,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    manifest = TFBSPoolArtifact.load(manifest_path)

    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def _capture_savefig(self, fname, *args, **kwargs):
        if Path(str(fname)).name == "background_logo.png":
            captured["size"] = tuple(float(v) for v in self.get_size_inches())
            suptitle = getattr(self, "_suptitle", None)
            captured["title"] = str(getattr(suptitle, "get_text", lambda: "")())
            if self.axes and suptitle is not None:
                self.canvas.draw()
                renderer = self.canvas.get_renderer()
                suptitle_bbox = suptitle.get_window_extent(renderer=renderer).transformed(self.transFigure.inverted())
                first_axis_bbox = self.axes[0].get_position()
                captured["title_gap"] = float(suptitle_bbox.y0 - first_axis_bbox.y1)
        return original_savefig(self, fname, *args, **kwargs)

    with patch.object(Figure, "savefig", _capture_savefig):
        paths = plot_stage_a_summary(
            pd.DataFrame(),
            out_path,
            pools=pools,
            pool_manifest=manifest,
            style={},
        )

    assert paths
    assert Path(paths[0]).name == "background_logo.png"
    assert captured.get("title") == "Background sequence logo"
    width, height = captured["size"]  # type: ignore[misc]
    assert width <= 11.0
    assert height <= 3.0
    assert float(captured.get("title_gap", 1.0)) <= 0.17


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
