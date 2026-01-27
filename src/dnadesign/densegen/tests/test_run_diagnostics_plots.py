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

from dnadesign.densegen.src.core.artifacts.pool import TFBSPoolArtifact
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


def _pool_manifest(tmp_path: Path) -> TFBSPoolArtifact:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.3",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": 2,
                "columns": ["input_name", "tf", "tfbs_sequence", "best_hit_score", "tier", "rank_within_regulator"],
                "pool_mode": "tfbs",
                "stage_a_sampling": {
                    "backend": "fimo",
                    "tier_scheme": "pct_1_9_90",
                    "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
                    "retention_rule": "top_n_sites_by_best_hit_score",
                    "fimo_thresh": 1.0,
                    "bgfile": None,
                    "background_source": "motif_background",
                    "eligible_score_hist": [
                        {
                            "regulator": "TF_A",
                            "edges": [0.0, 1.0, 2.0],
                            "counts": [1, 1],
                            "tier0_score": 2.0,
                            "tier1_score": 1.0,
                            "generated": 10,
                            "candidates_with_hit": 8,
                            "eligible": 6,
                            "unique_eligible": 4,
                            "retained": 2,
                        }
                    ],
                },
            }
        ],
    }
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


def test_plot_stage_a_summary(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    out_path = tmp_path / "stage_a_summary.png"
    pool_df = pd.DataFrame(
        {
            "input_name": ["demo_input", "demo_input"],
            "tf": ["TF_A", "TF_A"],
            "tfbs_sequence": ["AAAA", "AAAAT"],
            "best_hit_score": [2.0, 1.5],
            "tier": [0, 1],
            "rank_within_regulator": [1, 2],
        }
    )
    pools = {"demo_input": pool_df}
    manifest = _pool_manifest(tmp_path)
    paths = plot_stage_a_summary(
        pd.DataFrame(),
        out_path,
        pools=pools,
        pool_manifest=manifest,
        style={},
    )
    assert paths
    assert Path(paths[0]).exists()
