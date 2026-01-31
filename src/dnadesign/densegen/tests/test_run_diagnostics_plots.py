"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_run_diagnostics_plots.py

Coverage for the canonical DenseGen plot set after the scalability refactor.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from dnadesign.densegen.src.core.artifacts.pool import TFBSPoolArtifact
from dnadesign.densegen.src.viz.plot_stage_a_diversity import _build_stage_a_diversity_figure
from dnadesign.densegen.src.viz.plotting import (
    _plot_required_columns,
    plot_placement_map,
    plot_run_health,
    plot_stage_a_summary,
    plot_stage_b_summary,
    plot_tfbs_usage,
)


def _composition_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "solution_id": "s1",
                "input_name": "demo_input",
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
                "input_name": "demo_input",
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
                "input_name": "demo_input",
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


def _library_builds_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "library_size": 2,
                "sequence_length": 20,
                "fixed_bp": 6,
                "min_required_bp": 6,
                "slack_bp": 8,
                "infeasible": False,
            },
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "library_size": 2,
                "sequence_length": 20,
                "fixed_bp": 6,
                "min_required_bp": 8,
                "slack_bp": 6,
                "infeasible": False,
            },
        ]
    )


def _library_members_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "position": 0,
                "tf": "TF_A",
                "tfbs": "AAAA",
            },
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "position": 1,
                "tf": "TF_B",
                "tfbs": "CCCCCC",
            },
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "position": 0,
                "tf": "TF_A",
                "tfbs": "AAAA",
            },
            {
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 2,
                "library_hash": "hash2",
                "position": 1,
                "tf": "TF_B",
                "tfbs": "GGGG",
            },
        ]
    )


def _cfg() -> dict:
    return {
        "generation": {
            "sequence_length": 20,
            "plan": [
                {
                    "name": "demo_plan",
                    "fixed_elements": {
                        "promoter_constraints": [
                            {
                                "upstream_pos": [0, 6],
                                "downstream_pos": [10, 16],
                                "upstream": "TTGACA",
                                "downstream": "TATAAT",
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


def test_plot_required_columns_for_new_plots() -> None:
    cols = _plot_required_columns(["placement_map", "tfbs_usage", "run_health"], {})
    assert cols == []


def test_plot_run_health(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "run_health.png"
    plot_run_health(
        pd.DataFrame(),
        out_path,
        attempts_df=_attempts_df(),
        events_df=_events_df(),
        style={},
    )
    assert out_path.exists()


def test_plot_placement_map(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "placement_map.png"
    paths = plot_placement_map(
        pd.DataFrame(),
        out_path,
        composition_df=_composition_df(),
        cfg=_cfg(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_placement_map_accepts_effective_config(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "placement_map_effective.png"
    paths = plot_placement_map(
        pd.DataFrame(),
        out_path,
        composition_df=_composition_df(),
        cfg={"config": _cfg()},
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


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
    assert Path(paths[0]).exists()


def test_plot_stage_b_summary(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_b_summary.png"
    paths = plot_stage_b_summary(
        pd.DataFrame(),
        out_path,
        library_builds_df=_library_builds_df(),
        library_members_df=_library_members_df(),
        composition_df=_composition_df(),
        cfg=_cfg(),
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()


def test_plot_stage_b_summary_requires_metrics(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_b_summary_missing.png"
    builds = _library_builds_df().copy()
    builds["slack_bp"] = [None, None]
    with pytest.raises(ValueError, match="slack"):
        plot_stage_b_summary(
            pd.DataFrame(),
            out_path,
            library_builds_df=builds,
            library_members_df=_library_members_df(),
            composition_df=_composition_df(),
            cfg=_cfg(),
            style={},
        )


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
