"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_export_sequences_service.py

Validate sequence export artifacts and strict contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.cruncher.app.export_sequences_service import export_sequences_for_run
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_path,
    ensure_run_dirs,
    manifest_path,
    run_export_sequences_manifest_path,
    run_export_sequences_table_path,
)


def _uniform_pwm(length: int) -> list[list[float]]:
    return [[0.25, 0.25, 0.25, 0.25] for _ in range(length)]


def _write_run_fixture(tmp_path: Path, *, duplicate_hits: bool = False) -> Path:
    run_dir = tmp_path / "outputs" / "sample_export"
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_dirs(run_dir, meta=True, artifacts=True, live=False)

    sequence = "CTGCATATATTTTACAG"
    elites = pd.DataFrame([{"id": "elite-1", "rank": 1, "sequence": sequence}])
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    elites.to_parquet(elites_path(run_dir), engine="fastparquet")

    hits_rows = [
        {
            "elite_id": "elite-1",
            "tf": "lexA",
            "rank": 1,
            "chain": 0,
            "draw_idx": 0,
            "best_start": 0,
            "best_core_offset": 0,
            "best_strand": "+",
            "best_window_seq": "CTGCATATATTTTACA",
            "best_core_seq": "CTGCATATATTTTACA",
            "best_score_raw": 12.0,
            "best_score_scaled": 1.2,
            "best_score_norm": 0.92,
            "tiebreak_rule": "max_leftmost_plus",
            "pwm_ref": "demo:lexA_oops",
            "pwm_hash": "sha256_lexA",
            "pwm_width": 16,
            "core_width": 16,
            "core_def_hash": "core_lexA",
        },
        {
            "elite_id": "elite-1",
            "tf": "cpxR",
            "rank": 1,
            "chain": 0,
            "draw_idx": 0,
            "best_start": 3,
            "best_core_offset": 3,
            "best_strand": "-",
            "best_window_seq": "CATATATTTTA",
            "best_core_seq": "CATATATTTTA",
            "best_score_raw": 10.0,
            "best_score_scaled": 1.0,
            "best_score_norm": 0.85,
            "tiebreak_rule": "max_leftmost_minus",
            "pwm_ref": "demo:cpxR_oops",
            "pwm_hash": "sha256_cpxR",
            "pwm_width": 11,
            "core_width": 11,
            "core_def_hash": "core_cpxR",
        },
    ]
    if duplicate_hits:
        hits_rows.append(dict(hits_rows[0]))
    hits_df = pd.DataFrame(hits_rows)
    elites_hits_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    hits_df.to_parquet(elites_hits_path(run_dir), engine="fastparquet")

    config_used = {
        "cruncher": {
            "pwms_info": {
                "lexA": {"pwm_matrix": _uniform_pwm(16)},
                "cpxR": {"pwm_matrix": _uniform_pwm(11)},
            },
            "active_regulator_set": {"index": 1, "tfs": ["lexA", "cpxR"]},
        }
    }
    config_used_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    config_used_path(run_dir).write_text(yaml.safe_dump(config_used))

    manifest_payload = {
        "stage": "sample",
        "run_dir": str(run_dir.resolve()),
        "created_at": "2026-02-14T00:00:00+00:00",
        "artifacts": [],
        "motif_store": {"catalog_root": str((tmp_path / ".cruncher").resolve()), "pwm_source": "matrix"},
    }
    manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(run_dir).write_text(json.dumps(manifest_payload, indent=2))
    return run_dir


def test_export_sequences_for_run_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path)

    result = export_sequences_for_run(run_dir, run_name="sample_export")

    consensus_path = run_export_sequences_table_path(run_dir, table_name="consensus_sites", fmt="parquet")
    windows_path = run_export_sequences_table_path(run_dir, table_name="elite_tf_windows", fmt="parquet")
    pairwise_path = run_export_sequences_table_path(run_dir, table_name="elite_pairwise_windows", fmt="parquet")
    combos_path = run_export_sequences_table_path(run_dir, table_name="elite_combinations", fmt="parquet")
    manifest_out = run_export_sequences_manifest_path(run_dir)

    assert result.manifest_path == manifest_out
    assert consensus_path.exists()
    assert windows_path.exists()
    assert pairwise_path.exists()
    assert combos_path.exists()
    assert manifest_out.exists()

    consensus_df = pd.read_parquet(consensus_path, engine="fastparquet")
    windows_df = pd.read_parquet(windows_path, engine="fastparquet")
    pairwise_df = pd.read_parquet(pairwise_path, engine="fastparquet")
    combos_df = pd.read_parquet(combos_path, engine="fastparquet")
    export_manifest = json.loads(manifest_out.read_text())
    run_manifest = json.loads(manifest_path(run_dir).read_text())

    assert len(consensus_df) == 2
    assert len(windows_df) == 2
    assert len(pairwise_df) == 1
    assert len(combos_df) == 1
    assert export_manifest["kind"] == "sequence_export_v1"
    assert export_manifest["row_counts"]["elite_pairwise_windows"] == 1
    assert export_manifest["row_counts"]["elite_combinations"] == 1

    artifact_paths = {entry["path"] for entry in run_manifest.get("artifacts", [])}
    assert "export/sequences/table__consensus_sites.parquet" in artifact_paths
    assert "export/sequences/table__elite_tf_windows.parquet" in artifact_paths
    assert "export/sequences/table__elite_pairwise_windows.parquet" in artifact_paths
    assert "export/sequences/table__elite_combinations.parquet" in artifact_paths
    assert "export/sequences/export_manifest.json" in artifact_paths


def test_export_sequences_for_run_rejects_duplicate_tf_hits(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, duplicate_hits=True)

    with pytest.raises(ValueError, match="duplicate elite/tf rows"):
        export_sequences_for_run(run_dir, run_name="sample_export")
