"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_sample_multiplicity_smoke.py

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.artifacts.layout import (
    elites_hits_path,
    elites_objective_scores_path,
    elites_occurrences_path,
    manifest_path,
    random_baseline_hits_path,
    random_baseline_objective_scores_path,
    random_baseline_occurrences_path,
)
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path

runner = CliRunner()


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.97, 0.01, 0.01, 0.01]] * width,
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def _find_runs(runs_root: Path) -> list[Path]:
    runs: list[Path] = []
    if not runs_root.exists():
        return runs
    for manifest_file in runs_root.rglob("run_manifest.json"):
        run_dir = manifest_file.parent
        if run_dir.name in {"run", "meta"}:
            run_dir = run_dir.parent
        if run_dir.name == "previous":
            continue
        runs.append(run_dir)
    return runs


def test_sample_cli_smoke_single_tf_multiplicity(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entries = {
        "regulondb:RBM1": CatalogEntry(
            source="regulondb",
            motif_id="RBM1",
            tf_name="lexA",
            kind="PFM",
            has_matrix=True,
            matrix_source="alignment",
        ),
    }
    CatalogIndex(entries=entries).save(catalog_root)
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        width=1,
    )

    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix"},
            "sample": {
                "seed": 3,
                "sequence_length": 4,
                "budget": {"tune": 0, "draws": 4},
                "optimizer": {
                    "kind": "gibbs_anneal",
                    "chains": 1,
                    "cooling": {"kind": "fixed", "beta": 1.0},
                },
                "objective": {
                    "bidirectional": False,
                    "score_scale": "normalized-llr",
                    "combine": "min",
                    "multiplicity": {
                        "enabled": True,
                        "copies": 2,
                        "distinctness": {"mode": "interval", "min_gap": 0, "strand_rule": "collapse_same_locus"},
                        "aggregation": {"selector": "top_k_distinct", "scalar": "weakest_selected"},
                    },
                },
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                    "postprocess": {"trim_uncovered_internal": False},
                },
                "moves": {"profile": "balanced"},
                "output": {
                    "save_trace": False,
                    "save_sequences": True,
                    "save_random_baseline": True,
                    "random_baseline_n": 16,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
            },
            "analysis": {
                "run_selector": "latest",
                "fimo_compare": {"enabled": False},
                "trajectory_video": {"enabled": False},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {
                    "lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"},
                },
            }
        )
    )

    result = runner.invoke(app, ["sample", "-c", str(config_path)])
    assert result.exit_code == 0

    sample_runs = _find_runs(tmp_path / "runs")
    assert sample_runs
    run_dir = sample_runs[0]
    manifest = json.loads(manifest_path(run_dir).read_text())

    assert manifest["objective"]["representative_hit_contract"] is False
    assert manifest["objective"]["occurrence_artifacts"] is True
    assert manifest["objective"]["objective_kinds"] == ["k_distinct_weakest"]

    assert elites_objective_scores_path(run_dir).exists()
    assert elites_occurrences_path(run_dir).exists()
    assert random_baseline_objective_scores_path(run_dir).exists()
    assert random_baseline_occurrences_path(run_dir).exists()
    assert not elites_hits_path(run_dir).exists()
    assert not random_baseline_hits_path(run_dir).exists()

    elite_scores_df = pd.read_parquet(elites_objective_scores_path(run_dir), engine="pyarrow")
    elite_occurrences_df = pd.read_parquet(elites_occurrences_path(run_dir), engine="pyarrow")
    baseline_scores_df = pd.read_parquet(random_baseline_objective_scores_path(run_dir), engine="pyarrow")

    assert not elite_scores_df.empty
    assert {"objective_id", "requested_copies", "selected_copies", "scalar_score"}.issubset(elite_scores_df.columns)
    assert not elite_occurrences_df.empty
    assert {"objective_id", "occurrence_rank", "start", "end"}.issubset(elite_occurrences_df.columns)
    assert not baseline_scores_df.empty

    result = runner.invoke(app, ["analyze", "--summary", "-c", str(config_path)])
    assert result.exit_code == 0
    plots_dir = run_dir / "plots"
    assert (plots_dir / "elite_score_space_context.pdf").exists()
    assert (plots_dir / "chain_trajectory_sweep.pdf").exists()
    assert (plots_dir / "elites_nn_distance.pdf").exists()
    assert (plots_dir / "elites_showcase.pdf").exists()
