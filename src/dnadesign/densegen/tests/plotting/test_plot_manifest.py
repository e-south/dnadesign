"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_manifest.py

Plot manifest coverage for plot generation outputs.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.densegen.src.adapters.outputs.parquet import _build_schema
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.config.base import ConfigError
from dnadesign.densegen.src.viz import plotting as plotting_module
from dnadesign.densegen.src.viz.plotting import run_plots_from_config


def _diversity_block(core_len: int) -> dict:
    bins = [0, 1, 2]
    counts = [0, 2, 0]
    return {
        "candidate_pool_size": 2,
        "nnd_unweighted_k1": {
            "top_candidates": {
                "bins": bins,
                "counts": counts,
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
                "frac_le_1": 1.0,
                "n": 2,
                "subsampled": False,
                "k": 1,
            },
            "diversified_candidates": {
                "bins": bins,
                "counts": counts,
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
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
                "diversified_candidates": {
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
                "top_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "diversified_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "max_diversity_upper_bound": {
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


def _write_config(
    path: Path,
    *,
    plots_default: list[str],
    plots_options: dict[str, dict[str, object]] | None = None,
) -> None:
    options = plots_options or {}
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
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
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
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
              options: PLACEHOLDER_OPTIONS
            """
        )
        .strip()
        .replace("PLACEHOLDER_DEFAULT", json.dumps(plots_default))
        .replace("PLACEHOLDER_OPTIONS", json.dumps(options))
        + "\n"
    )


def _write_usr_source_config(
    path: Path,
    *,
    usr_root: Path,
    dataset: str,
    plots_default: list[str],
) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  root: "{usr_root}"
                  dataset: {dataset}
              generation:
                sequence_length: 30
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    fixed_elements:
                      promoter_constraints:
                        - upstream: TTGACA
                          downstream: TATAAT
                          spacer_length: [16, 16]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: usr
              out_dir: outputs/plots
              format: png
              default: {json.dumps(plots_default)}
              options: {{}}
              video:
                enabled: false
                mode: all_plans_round_robin_single_video
                sampling:
                  stride: 5
                  max_source_rows: 100
                  max_snapshots: 10
                playback:
                  target_duration_sec: 3
                  fps: 8
            """
        ).strip()
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
            "tfbs_core": ["AAAA", "AAAT", "AAAAAA"],
            "best_hit_score": [7.0, 9.0, 5.5],
            "tier": [1, 0, 2],
            "rank_within_regulator": [2, 1, 1],
            "selection_rank": [2, 1, 1],
            "nearest_selected_similarity": [0.5, 0.0, 0.0],
            "selection_score_norm": [0.25, 1.0, 1.0],
            "nearest_selected_distance_norm": [0.5, None, None],
            "motif_id": ["m1", "m1", "m2"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
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
                            "pwm_consensus_iupac": "AAAA",
                            "pwm_consensus_score": 10.0,
                            "pwm_theoretical_max_score": 10.0,
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
                            "selection_relevance_norm": "minmax_raw_score",
                            "selection_pool_size_final": 50,
                            "selection_pool_rung_fraction_used": 0.001,
                            "selection_pool_min_score_norm_used": None,
                            "selection_pool_capped": False,
                            "selection_pool_cap_value": None,
                            "diversity": _diversity_block(core_len=4),
                            "mining_audit": None,
                            "padding_audit": None,
                        },
                        {
                            "regulator": "tfB",
                            "pwm_consensus": "AAAAAA",
                            "pwm_consensus_iupac": "AAAAAA",
                            "pwm_consensus_score": 6.0,
                            "pwm_theoretical_max_score": 6.0,
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
                            "selection_relevance_norm": "minmax_raw_score",
                            "selection_pool_size_final": 10,
                            "selection_pool_rung_fraction_used": 0.001,
                            "selection_pool_min_score_norm_used": None,
                            "selection_pool_capped": False,
                            "selection_pool_cap_value": None,
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


def _write_pool_manifest_without_sampling(run_root: Path) -> None:
    pools_dir = run_root / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["TF_A", "TF_B", "TF_C"],
            "tfbs": ["AAAA", "CCCC", "GGGG"],
            "tfbs_core": ["AAAA", "CCCC", "GGGG"],
            "best_hit_score": [1.0, 1.0, 1.0],
            "tier": [0, 0, 0],
            "rank_within_regulator": [1, 1, 1],
            "selection_rank": [1, 1, 1],
            "nearest_selected_similarity": [0.0, 0.0, 0.0],
            "selection_score_norm": [1.0, 1.0, 1.0],
            "nearest_selected_distance_norm": [None, None, None],
            "motif_id": ["m1", "m2", "m3"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
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
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
                "stage_a_sampling": None,
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_stage_a_companion_records(run_root: Path) -> None:
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    expected_schema = _build_schema("densegen", pa)
    legacy_used_tfbs_detail_type = pa.list_(
        pa.struct(
            [
                pa.field("part_kind", pa.string()),
                pa.field("role", pa.string()),
                pa.field("constraint_name", pa.string()),
                pa.field("sequence", pa.string()),
                pa.field("variant_id", pa.string()),
                pa.field("spacer_length", pa.int64()),
                pa.field("placement_index", pa.int64()),
                pa.field("tf", pa.string()),
                pa.field("tfbs", pa.string()),
                pa.field("motif_id", pa.string()),
                pa.field("tfbs_id", pa.string()),
                pa.field("orientation", pa.string()),
                pa.field("offset", pa.int64()),
                pa.field("offset_raw", pa.int64()),
                pa.field("length", pa.int64()),
                pa.field("end", pa.int64()),
                pa.field("pad_left", pa.int64()),
                pa.field("site_id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("stage_a_best_hit_score", pa.float64()),
                pa.field("stage_a_rank_within_regulator", pa.int64()),
                pa.field("stage_a_tier", pa.int64()),
                pa.field("stage_a_fimo_start", pa.int64()),
                pa.field("stage_a_fimo_stop", pa.int64()),
                pa.field("stage_a_fimo_strand", pa.string()),
                pa.field("stage_a_selection_rank", pa.int64()),
                pa.field("stage_a_selection_score_norm", pa.float64()),
                pa.field("stage_a_tfbs_core", pa.string()),
            ]
        )
    )
    legacy_schema = pa.schema(
        [
            pa.field(field.name, legacy_used_tfbs_detail_type) if field.name == "densegen__used_tfbs_detail" else field
            for field in expected_schema
        ]
    )
    rows = [
        {
            "id": "row-1",
            "sequence": "ACGTACGTAA",
            "bio_type": "dna",
            "alphabet": "dna_4",
            "source": "demo",
            "densegen__schema_version": "2.9",
            "densegen__created_at": "2026-04-13T00:00:00Z",
            "densegen__run_id": "demo",
            "densegen__length": 10,
            "densegen__plan": "demo_plan",
            "densegen__input_name": "demo_input",
            "densegen__input_mode": "binding_sites",
            "densegen__input_pwm_ids": [],
            "densegen__used_tfbs": ["AAAA", "AAAAAA"],
            "densegen__used_tfbs_detail": [
                {
                    "part_kind": "tfbs",
                    "sequence": "AAAA",
                    "tf": "tfA",
                    "tfbs": "AAAA",
                    "orientation": "fwd",
                    "offset": 0,
                    "offset_raw": 0,
                    "length": 4,
                    "end": 4,
                    "pad_left": 0,
                    "source": "demo_input",
                    "stage_a_tfbs_core": "AAAA",
                    "stage_a_selection_rank": 1,
                },
                {
                    "part_kind": "tfbs",
                    "sequence": "AAAAAA",
                    "tf": "tfB",
                    "tfbs": "AAAAAA",
                    "orientation": "fwd",
                    "offset": 4,
                    "offset_raw": 4,
                    "length": 6,
                    "end": 10,
                    "pad_left": 0,
                    "source": "demo_input",
                    "stage_a_tfbs_core": "AAAAAA",
                    "stage_a_selection_rank": 1,
                },
            ],
            "densegen__used_tf_counts": [{"tf": "tfA", "count": 1}, {"tf": "tfB", "count": 1}],
            "densegen__library_unique_tf_count": 2,
            "densegen__library_unique_tfbs_count": 2,
            "densegen__covers_all_tfs_in_solution": True,
            "densegen__required_regulators": ["tfA", "tfB"],
            "densegen__min_count_by_regulator": [{"tf": "tfA", "min_count": 1}, {"tf": "tfB", "min_count": 1}],
            "densegen__compression_ratio": 1.0,
            "densegen__sampling_library_hash": "hash1",
            "densegen__sampling_library_index": 0,
            "densegen__pad_used": False,
            "densegen__pad_bases": 0,
            "densegen__pad_end": None,
            "densegen__pad_literal": None,
            "densegen__sequence_validation": {"validation_passed": True, "violations": []},
            "densegen__gc_total": 0.5,
            "densegen__gc_core": 0.5,
        },
        {
            "id": "row-2",
            "sequence": "TTTTCCCCAA",
            "bio_type": "dna",
            "alphabet": "dna_4",
            "source": "demo",
            "densegen__schema_version": "2.9",
            "densegen__created_at": "2026-04-13T00:00:00Z",
            "densegen__run_id": "demo",
            "densegen__length": 10,
            "densegen__plan": "demo_plan",
            "densegen__input_name": "demo_input",
            "densegen__input_mode": "binding_sites",
            "densegen__input_pwm_ids": [],
            "densegen__used_tfbs": ["AAAAT"],
            "densegen__used_tfbs_detail": [
                {
                    "part_kind": "tfbs",
                    "sequence": "AAAAT",
                    "tf": "tfA",
                    "tfbs": "AAAAT",
                    "orientation": "fwd",
                    "offset": 0,
                    "offset_raw": 0,
                    "length": 5,
                    "end": 5,
                    "pad_left": 0,
                    "source": "demo_input",
                    "stage_a_tfbs_core": "AAAT",
                    "stage_a_selection_rank": 2,
                }
            ],
            "densegen__used_tf_counts": [{"tf": "tfA", "count": 1}],
            "densegen__library_unique_tf_count": 1,
            "densegen__library_unique_tfbs_count": 1,
            "densegen__covers_all_tfs_in_solution": False,
            "densegen__required_regulators": ["tfA", "tfB"],
            "densegen__min_count_by_regulator": [{"tf": "tfA", "min_count": 1}, {"tf": "tfB", "min_count": 1}],
            "densegen__compression_ratio": 1.0,
            "densegen__sampling_library_hash": "hash1",
            "densegen__sampling_library_index": 1,
            "densegen__pad_used": False,
            "densegen__pad_bases": 0,
            "densegen__pad_end": None,
            "densegen__pad_literal": None,
            "densegen__sequence_validation": {"validation_passed": True, "violations": []},
            "densegen__gc_total": 0.5,
            "densegen__gc_core": 0.5,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=legacy_schema), tables_dir / "records.parquet")


def test_plot_manifest_written_for_concrete_stage_a_plots(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["stage_a_sampling_yield", "stage_a_pool_diversity"],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    names = {item["name"] for item in payload.get("plots", [])}
    assert {"stage_a_sampling_yield", "stage_a_pool_diversity"} <= names
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_a/stage_a_sampling_yield.png" in paths
    assert "stage_a/stage_a_pool_diversity.png" in paths
    stage_a_entries = [
        item
        for item in payload.get("plots", [])
        if item.get("name") in {"stage_a_sampling_yield", "stage_a_pool_diversity"}
    ]
    assert stage_a_entries
    assert all(str(item.get("plan_name") or "") == "stage_a" for item in stage_a_entries)
    for item in payload.get("plots", []):
        assert "plot_id" in item
        assert "group" in item
        assert "family" in item
        assert "variant" in item
        assert str(item.get("title") or "").strip()
        assert str(item.get("caption") or "").strip()
        assert str(item.get("alt_text") or "").strip()
    assert (run_root / "outputs" / "plots" / "current_inventory.json").exists()
    assert (run_root / "outputs" / "plots" / "artifact_ledger.json").exists()


def test_plot_run_purges_legacy_inventory_entries_during_regeneration(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["stage_a_sampling_yield"],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n", encoding="utf-8")
    _write_pool_manifest(run_root)

    plots_dir = run_root / "outputs" / "plots"
    legacy_plot_path = plots_dir / "dataset" / "dataset_source_inventory.png"
    legacy_plot_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_plot_path.write_text("legacy", encoding="utf-8")

    legacy_entry = {
        "plot_id": "dataset_source_inventory",
        "name": "dataset_source_inventory",
        "path": "dataset/dataset_source_inventory.png",
    }
    (plots_dir / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [legacy_entry],
            }
        ),
        encoding="utf-8",
    )
    (plots_dir / "artifact_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.artifact_ledger.v1",
                "plots": [legacy_entry],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    current_inventory = json.loads((plots_dir / "current_inventory.json").read_text(encoding="utf-8"))
    current_ids = {str(item.get("plot_id") or item.get("name") or "") for item in current_inventory.get("plots", [])}
    assert "dataset_source_inventory" not in current_ids
    assert "stage_a_sampling_yield" in current_ids

    plot_manifest = json.loads((plots_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    manifest_ids = {str(item.get("plot_id") or item.get("name") or "") for item in plot_manifest.get("plots", [])}
    assert "dataset_source_inventory" not in manifest_ids
    assert "stage_a_sampling_yield" in manifest_ids


def test_stage_b_summary_plots_are_manifested_with_current_summary_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=[
            "plan_regulator_deployment_heatmap",
            "retained_pool_coverage_by_regulator",
            "upstream_motif_supply_and_pwm_strength",
        ],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)
    _write_stage_a_companion_records(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    payload = json.loads((run_root / "outputs" / "plots" / "plot_manifest.json").read_text())
    entries_by_name = {item["name"]: item for item in payload.get("plots", [])}

    assert entries_by_name["plan_regulator_deployment_heatmap"]["path"] == (
        "stage_b_summary/plan_regulator_deployment_heatmap.png"
    )
    assert entries_by_name["retained_pool_coverage_by_regulator"]["path"] == (
        "stage_b_summary/retained_pool_coverage_by_regulator.png"
    )
    assert entries_by_name["upstream_motif_supply_and_pwm_strength"]["path"] == (
        "stage_b_summary/upstream_motif_supply_and_pwm_strength.png"
    )
    assert entries_by_name["plan_regulator_deployment_heatmap"]["plot_id"] == "plan_regulator_deployment_heatmap"
    assert entries_by_name["retained_pool_coverage_by_regulator"]["plot_id"] == ("retained_pool_coverage_by_regulator")
    assert entries_by_name["upstream_motif_supply_and_pwm_strength"]["plot_id"] == (
        "upstream_motif_supply_and_pwm_strength"
    )


def test_concrete_stage_a_plots_without_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["stage_a_sampling_yield", "stage_a_pool_diversity"],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    plots_dir = run_root / "outputs" / "plots"
    assert (plots_dir / "stage_a" / "stage_a_sampling_yield.png").exists()
    assert (plots_dir / "stage_a" / "stage_a_pool_diversity.png").exists()


def test_plot_run_removes_legacy_flat_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_sampling_yield"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    legacy = run_root / "outputs" / "plots" / "stage_a_summary__pool_tiers.png"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy")
    assert legacy.exists()

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    assert not legacy.exists()
    assert (run_root / "outputs" / "plots" / "stage_a" / "stage_a_sampling_yield.png").exists()


def test_plot_runner_rejects_unknown_plot_before_loading_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_sampling_yield"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)
    calls = {"load_records": 0}

    def _fail_if_called(*_args, **_kwargs):
        calls["load_records"] += 1
        raise AssertionError("records loader should not run for invalid --only plot names")

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        _fail_if_called,
    )

    with pytest.raises(ValueError, match="Unknown plot name requested: definitely_missing"):
        run_plots_from_config(loaded.root, cfg_path, only="definitely_missing")
    assert calls["load_records"] == 0


def test_compression_ratio_plot_requests_projected_output_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["compression_ratio_by_plan"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)
    captured: dict[str, list[str] | None] = {"columns": None}

    def _fake_load_records_from_config(*_args, **kwargs):
        cols = kwargs.get("columns")
        captured["columns"] = list(cols) if cols is not None else None
        return (
            pd.DataFrame(
                {
                    "densegen__compression_ratio": [1.0],
                    "densegen__plan": ["demo_plan"],
                }
            ),
            "parquet:outputs/tables/records.parquet",
        )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        _fake_load_records_from_config,
    )

    def _fake_compression_ratio_by_plan(df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        target = out_path.parent / "run_health" / "compression_ratio_by_plan.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        assert "densegen__compression_ratio" in df.columns
        assert "densegen__plan" in df.columns
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["compression_ratio_by_plan"],
        "fn",
        _fake_compression_ratio_by_plan,
    )
    run_plots_from_config(loaded.root, cfg_path, only="compression_ratio_by_plan")
    assert captured["columns"] is not None
    assert "densegen__compression_ratio" in (captured["columns"] or [])
    assert "densegen__plan" in (captured["columns"] or [])


def test_dataset_plot_uses_selected_output_source_rows_and_writes_dataset_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["source_cohort_concentration"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)
    captured: dict[str, list[str] | None] = {"columns": None}

    def _fake_load_records_from_config(*_args, **kwargs):
        cols = kwargs.get("columns")
        captured["columns"] = list(cols) if cols is not None else None
        return (
            pd.DataFrame(
                {
                    "source": ["plan_pool__demo_plan", "plan_pool__demo_plan", "plan_pool__other_plan"],
                    "densegen__plan": ["demo_plan", None, "other_plan"],
                    "densegen__input_name": ["plan_pool__demo_plan", None, "plan_pool__other_plan"],
                }
            ),
            "parquet:outputs/tables/records.parquet",
        )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        _fake_load_records_from_config,
    )

    def _fake_dataset_plot(df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        assert set(df.columns) >= {"source", "densegen__plan", "densegen__input_name"}
        target = out_path.parent / "dataset" / "source_cohort_concentration.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["source_cohort_concentration"], "fn", _fake_dataset_plot)

    run_plots_from_config(loaded.root, cfg_path, only="source_cohort_concentration")

    assert captured["columns"] is not None
    assert set(captured["columns"] or []) == {"source", "densegen__plan", "densegen__input_name"}

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    entries = [item for item in payload.get("plots", []) if item.get("name") == "source_cohort_concentration"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["path"] == "dataset/source_cohort_concentration.png"
    assert entry["group"] == "dataset"
    assert entry["family"] == "provenance"
    assert entry["plan_name"] == "unscoped"
    assert entry["plot_id"] == "source_cohort_concentration"
    assert entry["title"] == "Source cohort concentration"
    assert "final DenseGen records concentrate across source cohorts" in str(entry["caption"])


def test_dataset_plot_rerun_preserves_sibling_dataset_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["source_cohort_concentration", "source_plan_input_heatmap"],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "source": ["plan_pool__demo_plan", "plan_pool__other_plan"],
                    "densegen__plan": ["demo_plan", "other_plan"],
                    "densegen__input_name": ["plan_pool__demo_plan", "plan_pool__other_plan"],
                }
            ),
            "parquet:outputs/tables/records.parquet",
        ),
    )

    def _fake_dataset_inventory_plot(_df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        target = out_path.parent / "dataset" / "source_cohort_concentration.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("inventory")
        return [target]

    def _fake_dataset_heatmap_plot(_df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        target = out_path.parent / "dataset" / "source_plan_input_heatmap.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("heatmap")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["source_cohort_concentration"],
        "fn",
        _fake_dataset_inventory_plot,
    )
    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["source_plan_input_heatmap"],
        "fn",
        _fake_dataset_heatmap_plot,
    )

    run_plots_from_config(loaded.root, cfg_path, only="source_cohort_concentration")
    run_plots_from_config(loaded.root, cfg_path, only="source_plan_input_heatmap")

    inventory_path = run_root / "outputs" / "plots" / "dataset" / "source_cohort_concentration.png"
    heatmap_path = run_root / "outputs" / "plots" / "dataset" / "source_plan_input_heatmap.png"
    assert inventory_path.exists()
    assert heatmap_path.exists()

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    names = sorted(item.get("name") for item in payload.get("plots", []))
    assert names.count("source_cohort_concentration") == 1
    assert names.count("source_plan_input_heatmap") == 1


def test_placement_occupancy_map_uses_selected_output_source_for_solutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  root: ../usr_root
                  dataset: densegen_demo
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: usr
              out_dir: outputs/plots
              format: png
              default: ["placement_occupancy_map"]
            """
        ).strip()
        + "\n"
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\nTF_A,AAAA\n")
    loaded = load_config(cfg_path)

    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "sequence": ["ACGTACGTAA"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
        }
    )
    composition_df = pd.DataFrame(
        {
            "solution_id": ["sol-1"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
            "offset": [1],
            "length": [4],
        }
    )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:densegen_demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: composition_df.copy(),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"sequence_length": 10}}},
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_dense_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("placement_occupancy_map must use selected output source rows, not records.parquet")
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_libraries",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"library_index": [1], "library_hash": ["hash1"]}),
            pd.DataFrame(
                {
                    "input_name": ["demo_input"],
                    "plan_name": ["demo_plan"],
                    "library_index": [1],
                    "library_hash": ["hash1"],
                    "tf": ["TF_A"],
                    "tfbs": ["AAAA"],
                }
            ),
        ),
    )

    def _fake_placement_occupancy_map(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        cfg: dict,
        **_kwargs,
    ) -> list[Path]:
        assert list(dense_arrays_df["id"]) == ["sol-1"]
        assert list(composition_df["solution_id"]) == ["sol-1"]
        assert int(cfg["densegen"]["generation"]["sequence_length"]) == 10
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["placement_occupancy_map"],
        "fn",
        _fake_placement_occupancy_map,
    )

    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_b/demo_plan/demo_input/occupancy.png" in paths


def test_tfbs_concentration_profile_does_not_eager_load_stage_a_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["tfbs_concentration_profile"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "tf": ["TF_A"], "tfbs": ["AAAA"]}
    ).to_parquet(tables_dir / "composition.parquet", index=False)
    libs_dir = run_root / "outputs" / "libraries"
    libs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "library_index": [1], "library_hash": ["h1"]}
    ).to_parquet(libs_dir / "library_builds.parquet", index=False)
    pd.DataFrame(
        {
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "library_index": [1],
            "library_hash": ["h1"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
        }
    ).to_parquet(libs_dir / "library_members.parquet", index=False)

    loaded = load_config(cfg_path)
    expected_error = AssertionError("stage-a pools should not be loaded for tfbs_concentration_profile")
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_stage_a_pools",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(expected_error),
    )

    def _fake_tfbs_concentration_profile(
        _df: pd.DataFrame, out_path: Path, *, pools=None, composition_df=None, **_kwargs
    ) -> list[Path]:
        assert pools is None
        assert isinstance(composition_df, pd.DataFrame)
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "tfbs_concentration_profile.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["tfbs_concentration_profile"],
        "fn",
        _fake_tfbs_concentration_profile,
    )
    run_plots_from_config(loaded.root, cfg_path, only="tfbs_concentration_profile")


def test_placement_occupancy_map_recovers_composition_from_output_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["placement_occupancy_map"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "sequence": ["ACGTACGTAA"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
            "densegen__used_tfbs_detail": [
                [
                    {
                        "part_kind": "tfbs",
                        "regulator": "TF_A",
                        "sequence": "AAAA",
                        "offset": 2,
                        "offset_raw": 2,
                        "length": 4,
                        "end": 6,
                        "orientation": "fwd",
                        "source": "demo_source",
                        "motif_id": "motif-1",
                        "tfbs_id": "tfbs-1",
                        "site_id": "site-1",
                    }
                ]
            ],
        }
    )

    loaded = load_config(cfg_path)
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:densegen_demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"sequence_length": 10}}},
    )

    def _fake_placement_occupancy_map(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        cfg: dict,
        **_kwargs,
    ) -> list[Path]:
        assert list(dense_arrays_df["id"]) == ["sol-1"]
        assert list(composition_df["solution_id"]) == ["sol-1"]
        assert list(composition_df["tf"]) == ["TF_A"]
        assert list(composition_df["tfbs"]) == ["AAAA"]
        assert list(composition_df["offset"]) == [2]
        assert int(cfg["densegen"]["generation"]["sequence_length"]) == 10
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["placement_occupancy_map"],
        "fn",
        _fake_placement_occupancy_map,
    )

    run_plots_from_config(loaded.root, cfg_path, only="placement_occupancy_map")


def test_tfbs_concentration_profile_recovers_composition_from_output_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["tfbs_concentration_profile"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
            "densegen__used_tfbs_detail": [
                [
                    {
                        "part_kind": "tfbs",
                        "regulator": "TF_A",
                        "sequence": "AAAA",
                        "offset": 2,
                        "offset_raw": 2,
                        "length": 4,
                        "end": 6,
                        "orientation": "fwd",
                        "source": "demo_source",
                        "motif_id": "motif-1",
                        "tfbs_id": "tfbs-1",
                        "site_id": "site-1",
                    }
                ]
            ],
        }
    )

    loaded = load_config(cfg_path)
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:densegen_demo"),
    )

    def _fake_tfbs_concentration_profile(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        pools=None,
        composition_df: pd.DataFrame | None = None,
        **_kwargs,
    ) -> list[Path]:
        assert pools is None
        assert composition_df is not None
        assert list(composition_df["input_name"]) == ["demo_input"]
        assert list(composition_df["plan_name"]) == ["demo_plan"]
        assert list(composition_df["tf"]) == ["TF_A"]
        assert list(composition_df["tfbs"]) == ["AAAA"]
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "tfbs_concentration_profile.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["tfbs_concentration_profile"],
        "fn",
        _fake_tfbs_concentration_profile,
    )

    run_plots_from_config(loaded.root, cfg_path, only="tfbs_concentration_profile")


def test_load_composition_reads_part_files_when_final_file_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "solution_id": ["sol-1"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "placement_index": [0],
            "regulator": ["TF_A"],
            "sequence": ["AAAA"],
        }
    ).to_parquet(tables_dir / "composition_part-a.parquet", index=False)
    pd.DataFrame(
        {
            "solution_id": ["sol-2"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "placement_index": [0],
            "regulator": ["TF_B"],
            "sequence": ["CCCC"],
        }
    ).to_parquet(tables_dir / "composition_part-b.parquet", index=False)

    composition_df = plotting_module._load_composition(run_root, columns=["solution_id", "tf", "tfbs"])
    assert set(composition_df["solution_id"].astype(str)) == {"sol-1", "sol-2"}
    assert set(composition_df["tf"].astype(str)) == {"TF_A", "TF_B"}
    assert set(composition_df["tfbs"].astype(str)) == {"AAAA", "CCCC"}
    assert not (tables_dir / "composition.parquet").exists()
    assert len(list(tables_dir.glob("composition_part-*.parquet"))) == 2


def test_load_attempts_reads_part_files_when_final_file_missing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "status": ["ok"],
            "reason": [None],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:00:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(tables_dir / "attempts_part-a.parquet", index=False)
    pd.DataFrame(
        {
            "status": ["rejected"],
            "reason": ["duplicate"],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:01:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(tables_dir / "attempts_part-b.parquet", index=False)

    attempts_df = plotting_module._load_attempts(
        run_root, columns=["status", "reason", "plan_name", "created_at", "detail_json"]
    )
    assert sorted(attempts_df["status"].astype(str).tolist()) == ["ok", "rejected"]
    assert not (tables_dir / "attempts.parquet").exists()
    assert len(list(tables_dir.glob("attempts_part-*.parquet"))) == 2


def test_load_attempts_reads_part_files_without_mutating_read_only_tables_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    part_a = tables_dir / "attempts_part-a.parquet"
    part_b = tables_dir / "attempts_part-b.parquet"
    pd.DataFrame(
        {
            "status": ["ok"],
            "reason": [None],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:00:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(part_a, index=False)
    pd.DataFrame(
        {
            "status": ["rejected"],
            "reason": ["duplicate"],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:01:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(part_b, index=False)
    os.chmod(tables_dir, 0o555)
    try:
        attempts_df = plotting_module._load_attempts(
            run_root, columns=["status", "reason", "plan_name", "created_at", "detail_json"]
        )
    finally:
        os.chmod(tables_dir, 0o755)

    assert sorted(attempts_df["status"].astype(str).tolist()) == ["ok", "rejected"]
    assert not (tables_dir / "attempts.parquet").exists()


def test_load_attempts_dedupes_mixed_final_and_part_artifacts_by_attempt_id(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "attempt_id": ["a1"],
            "status": ["ok"],
            "reason": [None],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:00:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(tables_dir / "attempts.parquet", index=False)
    pd.DataFrame(
        {
            "attempt_id": ["a1"],
            "status": ["rejected"],
            "reason": ["duplicate"],
            "plan_name": ["demo_plan"],
            "created_at": ["2026-03-01T00:01:00+00:00"],
            "detail_json": [None],
        }
    ).to_parquet(tables_dir / "attempts_part-a.parquet", index=False)

    attempts_df = plotting_module._load_attempts(
        run_root,
        columns=["status", "reason", "plan_name", "created_at", "detail_json"],
    )

    assert len(attempts_df) == 1
    assert attempts_df.iloc[0]["status"] == "rejected"
    assert (tables_dir / "attempts.parquet").exists()
    assert (tables_dir / "attempts_part-a.parquet").exists()


def test_load_composition_reads_pending_part_files_with_final_file(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "solution_id": ["sol-0"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "placement_index": [0],
            "regulator": ["TF_A"],
            "sequence": ["AAAA"],
        }
    ).to_parquet(tables_dir / "composition.parquet", index=False)
    pd.DataFrame(
        {
            "solution_id": ["sol-1"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "placement_index": [0],
            "regulator": ["TF_B"],
            "sequence": ["CCCC"],
        }
    ).to_parquet(tables_dir / "composition_part-a.parquet", index=False)

    composition_df = plotting_module._load_composition(run_root, columns=["solution_id", "tf", "tfbs"])
    assert set(composition_df["solution_id"].astype(str)) == {"sol-0", "sol-1"}
    assert set(composition_df["tf"].astype(str)) == {"TF_A", "TF_B"}
    assert (tables_dir / "composition.parquet").exists()
    assert len(list(tables_dir.glob("composition_part-*.parquet"))) == 1


def test_run_plots_writes_manifest_for_partial_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["placement_occupancy_map", "tfbs_concentration_profile"],
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "sequence": ["ACGTACGTAA"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
            "densegen__used_tfbs_detail": [
                [
                    {
                        "part_kind": "tfbs",
                        "regulator": "TF_A",
                        "sequence": "AAAA",
                        "offset": 2,
                        "offset_raw": 2,
                        "length": 4,
                        "end": 6,
                        "orientation": "fwd",
                        "source": "demo_source",
                        "motif_id": "motif-1",
                        "tfbs_id": "tfbs-1",
                        "site_id": "site-1",
                    }
                ]
            ],
        }
    )

    loaded = load_config(cfg_path)
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:densegen_demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"sequence_length": 10}}},
    )

    def _fake_placement_occupancy_map(_df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    def _fake_tfbs_concentration_profile(*_args, **_kwargs) -> list[Path]:
        raise ValueError("synthetic failure")

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["placement_occupancy_map"],
        "fn",
        _fake_placement_occupancy_map,
    )
    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["tfbs_concentration_profile"],
        "fn",
        _fake_tfbs_concentration_profile,
    )

    with pytest.raises(RuntimeError, match="1 plot\\(s\\) failed"):
        run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_b/demo_plan/demo_input/occupancy.png" in paths


def test_stage_a_sampling_yield_reads_projected_pool_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_sampling_yield"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    original = plotting_module.pd.read_parquet
    captured: dict[str, list[str] | None] = {"columns": None}

    def _spy_read_parquet(path: Path, *args, **kwargs):
        if Path(path).name.endswith("__pool.parquet"):
            cols = kwargs.get("columns")
            captured["columns"] = list(cols) if cols is not None else None
        return original(path, *args, **kwargs)

    monkeypatch.setattr(plotting_module.pd, "read_parquet", _spy_read_parquet)
    run_plots_from_config(loaded.root, cfg_path, only="stage_a_sampling_yield")
    assert captured["columns"] is not None


def test_stage_b_plot_options_reject_unknown_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["placement_occupancy_map"],
        plots_options={"placement_occupancy_map": {"scope": "auto", "max_plans": 2, "unknown_key": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(cfg_path)


def test_tfbs_concentration_profile_plot_options_reject_unknown_keys(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["tfbs_concentration_profile"],
        plots_options={"tfbs_concentration_profile": {"scope": "auto", "max_plans": 2, "unknown_key": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "tf": ["TF_A"], "tfbs": ["AAAA"]}
    ).to_parquet(tables_dir / "composition.parquet", index=False)

    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(cfg_path)


def test_stage_b_scope_auto_groups_and_drills_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["placement_occupancy_map"],
        plots_options={"placement_occupancy_map": {"scope": "auto", "max_plans": 2, "drilldown_plans": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    loaded = load_config(cfg_path)

    dense_arrays_df = pd.DataFrame(
        {
            "id": [f"sol-{idx}" for idx in range(1, 7)],
            "sequence": ["ACGTACGTAA"] * 6,
            "densegen__input_name": [
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_b__sig10_B",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
            ],
            "densegen__plan": [
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=b__sig10=B",
                "sigma70_topup__sig35=f__sig10=H",
                "sigma70_topup__sig35=f__sig10=H",
            ],
        }
    )
    composition_df = pd.DataFrame(
        {
            "solution_id": [f"sol-{idx}" for idx in range(1, 7)],
            "input_name": [
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_b__sig10_B",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
            ],
            "plan_name": [
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=b__sig10=B",
                "sigma70_topup__sig35=f__sig10=H",
                "sigma70_topup__sig35=f__sig10=H",
            ],
            "tf": ["TF_A"] * 6,
            "tfbs": ["AAAA"] * 6,
            "offset": [1, 1, 1, 1, 1, 1],
            "length": [4, 4, 4, 4, 4, 4],
        }
    )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: composition_df.copy(),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {
            "densegen": {
                "generation": {
                    "sequence_length": 10,
                    "plan": [{"name": "sigma70_panel", "quota": 1}, {"name": "sigma70_topup", "quota": 1}],
                }
            }
        },
    )

    seen_plan_sets: list[tuple[str, ...]] = []
    seen_input_sets: list[tuple[str, ...]] = []

    def _fake_placement_occupancy_map(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        **_kwargs,
    ) -> list[Path]:
        dense_plans = tuple(sorted(set(dense_arrays_df["densegen__plan"].astype(str))))
        comp_plans = tuple(sorted(set(composition_df["plan_name"].astype(str))))
        dense_inputs = tuple(sorted(set(dense_arrays_df["densegen__input_name"].astype(str))))
        comp_inputs = tuple(sorted(set(composition_df["input_name"].astype(str))))
        assert dense_plans == comp_plans
        assert dense_inputs == comp_inputs
        seen_plan_sets.append(dense_plans)
        seen_input_sets.append(dense_inputs)
        target = out_path.parent / "stage_b" / dense_plans[0] / dense_inputs[0] / f"occupancy_{len(seen_plan_sets)}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(
        plotting_module.AVAILABLE_PLOTS["placement_occupancy_map"],
        "fn",
        _fake_placement_occupancy_map,
    )

    run_plots_from_config(loaded.root, cfg_path, only="placement_occupancy_map")

    assert len(seen_plan_sets) == 2
    assert seen_plan_sets[0] == ("sigma70_panel", "sigma70_topup")
    assert seen_plan_sets[1] == ("sigma70_panel__sig35=a__sig10=A",)
    assert seen_input_sets[0] == ("plan_pool__sigma70_panel", "plan_pool__sigma70_topup")
    assert seen_input_sets[1] == ("plan_pool__sigma70_panel__sig35_a__sig10_A",)
