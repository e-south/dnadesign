"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_plot_manifest.py

Plot manifest coverage for plot generation outputs.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.viz.plotting import run_plots_from_config


def _diversity_block(core_len: int) -> dict:
    bins = [0, 1, 2]
    counts = [0, 2, 0]
    return {
        "candidate_pool_size": 2,
        "shortlist_target": 10,
        "core_hamming": {
            "metric": "hamming",
            "nnd_k1": {
                "k": 1,
                "baseline": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
                "actual": {
                    "bins": bins,
                    "counts": counts,
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
                "baseline": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "actual": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "upper_bound": {
                    "bins": bins,
                    "counts": counts,
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
            "baseline": {"values": [0.0] * core_len, "n": 2},
            "actual": {"values": [0.0] * core_len, "n": 2},
        },
        "score_quantiles": {
            "baseline": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "actual": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "baseline_global": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "upper_bound": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
        },
    }


def _write_config(path: Path, *, plots_default: list[str]) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.7"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 10
                quota: 1
                plan:
                  - name: demo_plan
                    quota: 1
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              format: png
              default: PLACEHOLDER_DEFAULT
            """
        )
        .strip()
        .replace("PLACEHOLDER_DEFAULT", json.dumps(plots_default))
        + "\n"
    )


def _write_pool_manifest(run_root: Path) -> None:
    pools_dir = run_root / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["tfA", "tfA", "tfB"],
            "tfbs": ["AAAA", "AAAAT", "AAAAAA"],
            "best_hit_score": [7.0, 9.0, 5.5],
            "tier": [1, 0, 2],
            "rank_within_regulator": [2, 1, 1],
            "motif_id": ["m1", "m1", "m2"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
    manifest = {
        "schema_version": "1.5",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": int(len(df)),
                "columns": list(df.columns),
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
                            "regulator": "tfA",
                            "pwm_consensus": "AAAA",
                            "edges": [4.0, 6.0, 8.0, 10.0],
                            "counts": [0, 1, 1],
                            "tier0_score": 9.0,
                            "tier1_score": 7.0,
                            "tier2_score": 6.0,
                            "tier_fractions": [0.001, 0.01, 0.09],
                            "tier_fractions_source": "default",
                            "generated": 10,
                            "candidates_with_hit": 9,
                            "eligible_raw": 8,
                            "eligible_unique": 3,
                            "retained": 2,
                            "selection_policy": "mmr",
                            "selection_alpha": 0.9,
                            "selection_similarity": "weighted_hamming_tolerant",
                            "selection_shortlist_k": 50,
                            "selection_shortlist_min": 10,
                            "selection_shortlist_factor": 5,
                            "selection_shortlist_max": None,
                            "selection_shortlist_target": 250,
                            "selection_shortlist_target_met": True,
                            "selection_tier_fraction_used": 0.001,
                            "selection_tier_limit": 50,
                            "selection_pool_source": "shortlist_k",
                            "diversity": _diversity_block(core_len=4),
                            "mining_audit": None,
                            "padding_audit": None,
                        },
                        {
                            "regulator": "tfB",
                            "pwm_consensus": "AAAAAA",
                            "edges": [4.0, 6.0, 8.0],
                            "counts": [1, 0],
                            "tier0_score": 5.5,
                            "tier1_score": None,
                            "tier2_score": None,
                            "tier_fractions": [0.001, 0.01, 0.09],
                            "tier_fractions_source": "default",
                            "generated": 5,
                            "candidates_with_hit": 4,
                            "eligible_raw": 3,
                            "eligible_unique": 2,
                            "retained": 1,
                            "selection_policy": "mmr",
                            "selection_alpha": 0.9,
                            "selection_similarity": "weighted_hamming_tolerant",
                            "selection_shortlist_k": 10,
                            "selection_shortlist_min": 5,
                            "selection_shortlist_factor": 5,
                            "selection_shortlist_max": None,
                            "selection_shortlist_target": 25,
                            "selection_shortlist_target_met": True,
                            "selection_tier_fraction_used": 0.001,
                            "selection_tier_limit": 10,
                            "selection_pool_source": "shortlist_k",
                            "diversity": _diversity_block(core_len=6),
                            "mining_audit": None,
                            "padding_audit": None,
                        },
                    ],
                },
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def test_plot_manifest_written(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    names = {item["name"] for item in payload.get("plots", [])}
    assert "stage_a_summary" in names
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_a_summary__demo_input.png" in paths
    assert "stage_a_summary__demo_input__diversity.png" in paths


def test_stage_a_plots_without_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    plots_dir = run_root / "outputs" / "plots"
    overview_plot = plots_dir / "stage_a_summary__demo_input.png"
    assert overview_plot.exists()
    diversity_plot = plots_dir / "stage_a_summary__demo_input__diversity.png"
    assert diversity_plot.exists()
