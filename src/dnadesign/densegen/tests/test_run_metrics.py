"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_run_metrics.py

Run-metrics extraction for diagnostics plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.run_metrics import build_run_metrics


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
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
                sequence_length: 20
                quota: 1
                plan:
                  - name: demo_plan
                    quota: 1
                    required_regulators: [TF_A, TF_B]
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    return cfg_path


def _write_pool_manifest(tmp_path: Path) -> None:
    pools_dir = tmp_path / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["TF_A", "TF_A", "TF_B"],
            "tfbs": ["AAAA", "AAAAT", "CCCCCC"],
            "best_hit_score": [10.0, 8.0, 5.0],
            "tier": [0, 1, 2],
            "rank_within_regulator": [1, 2, 1],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
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
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_libraries(tmp_path: Path) -> None:
    libraries_dir = tmp_path / "outputs" / "libraries"
    libraries_dir.mkdir(parents=True, exist_ok=True)
    builds = pd.DataFrame(
        [
            {
                "created_at": "2026-01-26T00:00:00+00:00",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_id": "hash1",
                "library_hash": "hash1",
                "pool_strategy": "subsample",
                "library_sampling_strategy": "coverage_weighted",
                "library_size": 3,
                "target_length": 40,
                "achieved_length": 15,
                "relaxed_cap": False,
                "final_cap": None,
                "iterative_max_libraries": 1,
                "iterative_min_new_solutions": 1,
            }
        ]
    )
    members = pd.DataFrame(
        [
            {
                "library_id": "hash1",
                "library_hash": "hash1",
                "library_index": 1,
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "position": 0,
                "tf": "TF_A",
                "tfbs": "AAAA",
                "tfbs_id": None,
                "motif_id": None,
                "site_id": None,
                "source": None,
            },
            {
                "library_id": "hash1",
                "library_hash": "hash1",
                "library_index": 1,
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "position": 1,
                "tf": "TF_A",
                "tfbs": "AAAAT",
                "tfbs_id": None,
                "motif_id": None,
                "site_id": None,
                "source": None,
            },
            {
                "library_id": "hash1",
                "library_hash": "hash1",
                "library_index": 1,
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "position": 2,
                "tf": "TF_B",
                "tfbs": "CCCCCC",
                "tfbs_id": None,
                "motif_id": None,
                "site_id": None,
                "source": None,
            },
        ]
    )
    builds.to_parquet(libraries_dir / "library_builds.parquet", index=False)
    members.to_parquet(libraries_dir / "library_members.parquet", index=False)


def _write_attempts(tmp_path: Path) -> None:
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    attempts = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-26T00:00:00+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "AAA",
                "sequence_hash": "hash",
                "solution_id": "s1",
                "used_tf_counts_json": "{}",
                "used_tf_list": [],
                "sampling_library_index": 1,
                "sampling_library_hash": "hash1",
                "solver_status": "ok",
                "solver_objective": 1.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": "0.1.0",
                "dense_arrays_version_source": "installed",
                "library_tfbs": ["AAAA", "AAAAT", "CCCCCC"],
                "library_tfs": ["TF_A", "TF_A", "TF_B"],
                "library_site_ids": ["", "", ""],
                "library_sources": ["", "", ""],
            }
        ]
    )
    attempts.to_parquet(tables_dir / "attempts.parquet", index=False)


def _write_composition(tmp_path: Path) -> None:
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    composition = pd.DataFrame(
        [
            {
                "solution_id": "s1",
                "attempt_id": "a1",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "placement_index": 0,
                "tf": "TF_A",
                "tfbs": "AAAA",
                "motif_id": None,
                "tfbs_id": None,
                "orientation": "+",
                "offset": 0,
                "length": 4,
                "end": 4,
                "pad_left": 0,
                "site_id": None,
                "source": None,
            },
            {
                "solution_id": "s1",
                "attempt_id": "a1",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_index": 1,
                "library_hash": "hash1",
                "placement_index": 1,
                "tf": "TF_B",
                "tfbs": "CCCCCC",
                "motif_id": None,
                "tfbs_id": None,
                "orientation": "+",
                "offset": 10,
                "length": 6,
                "end": 16,
                "pad_left": 0,
                "site_id": None,
                "source": None,
            },
        ]
    )
    composition.to_parquet(tables_dir / "composition.parquet", index=False)


def _write_dense_arrays(tmp_path: Path) -> None:
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        [
            {
                "densegen__input_name": "demo_input",
                "densegen__plan": "demo_plan",
                "densegen__sampling_library_index": 1,
                "densegen__sampling_library_hash": "hash1",
                "densegen__used_tfbs_detail": [
                    {"tf": "TF_A", "tfbs": "AAAA", "offset": 0},
                    {"tf": "TF_B", "tfbs": "CCCCCC", "offset": 10},
                ],
                "length": 20,
            }
        ]
    )
    rows.to_parquet(tables_dir / "dense_arrays.parquet", index=False)


def _write_sampling_pressure_events(tmp_path: Path) -> None:
    meta_dir = tmp_path / "outputs" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event": "LIBRARY_SAMPLING_PRESSURE",
            "created_at": "2026-01-26T00:00:00+00:00",
            "input_name": "demo_input",
            "plan_name": "demo_plan",
            "library_index": 1,
            "library_hash": "hash1",
            "sampling_strategy": "coverage_weighted",
            "weight_by_tf": {"TF_A": 1.5, "TF_B": 0.5},
            "weight_fraction_by_tf": {"TF_A": 0.75, "TF_B": 0.25},
            "usage_count_by_tf": {"TF_A": 3, "TF_B": 1},
            "failure_count_by_tf": {"TF_A": 1, "TF_B": 0},
        }
    ]
    (meta_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_build_run_metrics_library_health(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    _write_pool_manifest(tmp_path)
    _write_libraries(tmp_path)
    _write_attempts(tmp_path)
    _write_composition(tmp_path)

    loaded = load_config(cfg_path)
    metrics = build_run_metrics(cfg=loaded.root.densegen, run_root=tmp_path)

    health = metrics[metrics["metric_group"] == "library_health"]
    assert len(health) == 1
    row = health.iloc[0]
    assert row["library_size"] == 3
    assert row["unique_tf_count"] == 2
    assert row["unique_tfbs_count"] == 3
    assert math.isclose(row["max_tf_dominance"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(row["score_mean"], (10.0 + 8.0 + 5.0) / 3, rel_tol=1e-6)
    assert row["score_median"] == 8.0
    assert row["tfbs_length_sum"] == 15
    assert row["required_min_length"] == 10
    assert row["fixed_bp_min"] == 0
    assert row["slack_bp"] == 10
    assert math.isclose(row["never_used_tfbs_fraction"], 1 - (2 / 3), rel_tol=1e-6)

    offered = metrics[metrics["metric_group"] == "offered_vs_used_tf"]
    assert set(offered["tf"].tolist()) == {"TF_A", "TF_B"}
    offered_a = offered[offered["tf"] == "TF_A"].iloc[0]
    assert offered_a["offered_count"] == 2
    assert offered_a["used_count"] == 1

    tiers = metrics[metrics["metric_group"] == "tier_enrichment"]
    assert set(tiers["tier"].tolist()) == {0, 1, 2}
    t0 = tiers[(tiers["tf"] == "TF_A") & (tiers["tier"] == 0)].iloc[0]
    assert t0["usage_rate"] == 1.0


def test_build_run_metrics_sampling_pressure(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    _write_pool_manifest(tmp_path)
    _write_libraries(tmp_path)
    _write_attempts(tmp_path)
    _write_composition(tmp_path)
    _write_sampling_pressure_events(tmp_path)

    loaded = load_config(cfg_path)
    metrics = build_run_metrics(cfg=loaded.root.densegen, run_root=tmp_path)

    pressure = metrics[metrics["metric_group"] == "sampling_pressure"]
    assert set(pressure["tf"].tolist()) == {"TF_A", "TF_B"}
    row = pressure[pressure["tf"] == "TF_A"].iloc[0]
    assert row["weight"] == 1.5
    assert row["weight_fraction"] == 0.75
    assert row["usage_count"] == 3


def test_build_run_metrics_traceability_dense_arrays(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    _write_pool_manifest(tmp_path)
    _write_libraries(tmp_path)
    _write_attempts(tmp_path)
    _write_dense_arrays(tmp_path)

    loaded = load_config(cfg_path)
    metrics = build_run_metrics(cfg=loaded.root.densegen, run_root=tmp_path)

    tiers = metrics[metrics["metric_group"] == "tier_enrichment"]
    assert not tiers.empty
