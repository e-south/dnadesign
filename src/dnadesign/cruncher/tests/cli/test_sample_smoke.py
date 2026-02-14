"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_sample_smoke.py

CLI smoke test for a minimal two-TF sampling run (matrix PWMs).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.artifacts.layout import sequences_path
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path

runner = CliRunner()


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.7, 0.1, 0.1, 0.1]] * 3,
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


def test_sample_cli_smoke_matrix(tmp_path: Path) -> None:
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
        "regulondb:RBM2": CatalogEntry(
            source="regulondb",
            motif_id="RBM2",
            tf_name="cpxR",
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
    )
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM2.json",
        source="regulondb",
        motif_id="RBM2",
        tf_name="cpxR",
    )

    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix"},
            "sample": {
                "seed": 3,
                "sequence_length": 6,
                "budget": {"tune": 1, "draws": 2},
                "optimizer": {
                    "kind": "gibbs_anneal",
                    "chains": 2,
                    "cooling": {"kind": "linear", "beta_start": 0.1, "beta_end": 1.0},
                },
                "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                },
                "moves": {"profile": "balanced"},
                "output": {
                    "save_trace": False,
                    "save_sequences": True,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
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
                    "cpxR": {"source": "regulondb", "motif_id": "RBM2", "sha256": "good"},
                },
            }
        )
    )

    result = runner.invoke(app, ["sample", "-c", str(config_path)])
    assert result.exit_code == 0
    sample_runs = _find_runs(tmp_path / "runs")
    assert sample_runs
    logos_dir = tmp_path / "runs" / "plots" / "logos" / "catalog" / "demo"
    logos_dir.mkdir(parents=True, exist_ok=True)
    logo_path = logos_dir / "lexA_logo.png"
    logo_path.write_text("logo")
    assert logo_path.exists()
    result = runner.invoke(app, ["sample", "-c", str(config_path)])
    assert result.exit_code != 0
    assert "--force-overwrite" in result.output
    result = runner.invoke(app, ["sample", "--force-overwrite", "-c", str(config_path)])
    assert result.exit_code == 0
    assert logo_path.exists()
    seq_path = sequences_path(sample_runs[0])
    seq_df = pd.read_parquet(seq_path)
    assert "min_norm" in seq_df.columns
    assert "min_per_tf_norm" not in seq_df.columns
    assert "chain" in seq_df.columns
    assert "chain_1based" in seq_df.columns
    assert "beta" in seq_df.columns
    assert "sweep_idx" in seq_df.columns
