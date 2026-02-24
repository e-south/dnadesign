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
    run_export_dir,
    run_export_sequences_manifest_path,
    run_export_sequences_table_path,
)


def _uniform_pwm(length: int) -> list[list[float]]:
    return [[0.25, 0.25, 0.25, 0.25] for _ in range(length)]


def _write_run_fixture(
    tmp_path: Path,
    *,
    duplicate_hits: bool = False,
    include_third_tf: bool = False,
) -> Path:
    run_dir = tmp_path / "outputs" / "sample_export"
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_dirs(run_dir, meta=True, artifacts=True, live=False)

    sequence = "CTGCATATATTTTACAG"
    elites = pd.DataFrame([{"id": "elite-1", "rank": 1, "sequence": sequence, "combined_score_final": 0.9}])
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
    if include_third_tf:
        hits_rows.append(
            {
                "elite_id": "elite-1",
                "tf": "baeR",
                "rank": 1,
                "chain": 0,
                "draw_idx": 0,
                "best_start": 10,
                "best_core_offset": 10,
                "best_strand": "+",
                "best_window_seq": "TTACA",
                "best_core_seq": "TTACA",
                "best_score_raw": 8.5,
                "best_score_scaled": 0.9,
                "best_score_norm": 0.8,
                "tiebreak_rule": "max_leftmost_plus",
                "pwm_ref": "demo:baeR_oops",
                "pwm_hash": "sha256_baeR",
                "pwm_width": 5,
                "core_width": 5,
                "core_def_hash": "core_baeR",
            }
        )
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
    if include_third_tf:
        config_used["cruncher"]["pwms_info"]["baeR"] = {"pwm_matrix": _uniform_pwm(5)}
        config_used["cruncher"]["active_regulator_set"]["tfs"] = ["lexA", "cpxR", "baeR"]
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
    stale_export = run_export_dir(run_dir) / "table__monospecific_elite_windows.csv"
    stale_export.parent.mkdir(parents=True, exist_ok=True)
    stale_export.write_text("stale\n")

    result = export_sequences_for_run(run_dir, run_name="sample_export")

    consensus_path = run_export_sequences_table_path(run_dir, table_name="consensus_sites", fmt="csv")
    elites_path_csv = run_export_dir(run_dir) / "table__elites.csv"
    manifest_out = run_export_sequences_manifest_path(run_dir)

    assert result.manifest_path == manifest_out
    assert result.output_dir == run_export_dir(run_dir)
    assert consensus_path.exists()
    assert elites_path_csv.exists()
    assert manifest_out.exists()
    assert not stale_export.exists()

    consensus_df = pd.read_csv(consensus_path)
    elites_csv_df = pd.read_csv(elites_path_csv)
    export_manifest = json.loads(manifest_out.read_text())
    run_manifest = json.loads(manifest_path(run_dir).read_text())

    assert len(consensus_df) == 2
    assert len(elites_csv_df) == 1
    assert "sequence_length" in elites_csv_df.columns
    assert "window_members_json" in elites_csv_df.columns
    assert int(elites_csv_df.loc[0, "sequence_length"]) == len("CTGCATATATTTTACAG")
    members = json.loads(str(elites_csv_df.loc[0, "window_members_json"]))
    assert isinstance(members, list)
    assert len(members) == 2
    assert {
        "regulator_id",
        "offset_start",
        "offset_end",
        "window_kmer",
        "core_kmer",
        "strand",
        "score_name",
        "score",
    }.issubset(set(members[0].keys()))
    assert export_manifest["kind"] == "sequence_export_v3"
    assert export_manifest["table_format"] == "csv"
    assert export_manifest["files"]["elites"] == "export/table__elites.csv"
    assert export_manifest["files"]["consensus_sites"] == "export/table__consensus_sites.csv"
    assert export_manifest["row_counts"]["elites"] == 1
    assert export_manifest["row_counts"]["consensus_sites"] == 2

    artifact_paths = {entry["path"] for entry in run_manifest.get("artifacts", [])}
    assert "export/table__consensus_sites.csv" in artifact_paths
    assert "export/table__elites.csv" in artifact_paths
    assert "export/export_manifest.json" in artifact_paths


def test_export_sequences_for_run_rejects_duplicate_tf_hits(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, duplicate_hits=True)

    with pytest.raises(ValueError, match="duplicate elite/tf rows"):
        export_sequences_for_run(run_dir, run_name="sample_export")


def test_export_sequences_for_run_includes_rich_window_members_metadata(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, include_third_tf=True)

    export_sequences_for_run(run_dir, run_name="sample_export", table_format="parquet")

    elites_path_csv = run_export_dir(run_dir) / "table__elites.csv"
    consensus_path = run_export_sequences_table_path(run_dir, table_name="consensus_sites", fmt="parquet")

    elites_df = pd.read_csv(elites_path_csv)
    consensus_df = pd.read_parquet(consensus_path, engine="fastparquet")

    assert len(elites_df) == 1
    assert len(consensus_df) == 3
    members = json.loads(str(elites_df.loc[0, "window_members_json"]))
    assert len(members) == 3
    assert sorted(item["regulator_id"] for item in members) == ["baeR", "cpxR", "lexA"]
    assert all(item["score_name"] == "best_score_norm" for item in members)


def test_export_sequences_for_run_requires_combined_score_final(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path)
    elites_df = pd.read_parquet(elites_path(run_dir), engine="fastparquet")
    elites_df = elites_df.drop(columns=["combined_score_final"], errors="ignore")
    elites_df.to_parquet(elites_path(run_dir), engine="fastparquet")

    with pytest.raises(ValueError, match="combined_score_final"):
        export_sequences_for_run(run_dir, run_name="sample_export")
