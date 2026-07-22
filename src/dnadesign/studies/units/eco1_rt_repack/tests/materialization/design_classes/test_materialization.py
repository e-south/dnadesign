"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/design_classes/test_materialization.py

Design-class expansion materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes import (
    materialize_design_class_candidate_pool,
    materialize_design_class_requests,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.cli import main
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import require_ec86kit_source_artifacts
from dnadesign.thread.candidates import write_candidate_table


def test_design_class_requests_materialize_named_mask_policies(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    result = materialize_design_class_requests(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    by_id = {row["design_class_id"]: row for row in manifest["design_classes"]}
    assert by_id["eco1_rt_clade9_plurality25_contact5a_v1"]["role"] == "baseline_existing"
    assert by_id["eco1_rt_clade9_plurality25_contact6a_v1"]["non_fixed_mapped_position_count"] == 103
    assert by_id["eco1_rt_clade9_plurality25_contact8a_v1"]["non_fixed_mapped_position_count"] == 51
    assert by_id["eco1_rt_clade9_plurality25_contact10a_v1"]["non_fixed_mapped_position_count"] == 32
    assert by_id["eco1_rt_clade9_plurality50_contact5a_v1"]["non_fixed_mapped_position_count"] == 139
    assert by_id["eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1"]["non_fixed_mapped_position_count"] == 118

    class_root = Path(by_id["eco1_rt_clade9_plurality25_contact6a_v1"]["class_root"])
    mask_set = yaml.safe_load((class_root / "mask_set.yaml").read_text(encoding="utf-8"))
    thread_plan = yaml.safe_load((class_root / "thread_plan.yaml").read_text(encoding="utf-8"))
    request_manifest = yaml.safe_load(
        (class_root / "proteinmpnn_request/request_manifest.yaml").read_text(encoding="utf-8")
    )
    assert mask_set["mask_policy_id"] == "eco1_rt_clade9_plurality25_contact6a_v1"
    assert mask_set["summary"]["selected_contact_threshold_angstrom"] == 6.0
    assert thread_plan["mask_policy_id"] == "eco1_rt_clade9_plurality25_contact6a_v1"
    assert thread_plan["batch_id"] == "eco1_rt_clade9_p25_6a_n96_20260701"
    assert request_manifest["mask_policy_id"] == "eco1_rt_clade9_plurality25_contact6a_v1"
    assert request_manifest["mutable_position_count"] == 103


def test_candidate_pool_deduplicates_sequence_hashes_across_classes(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    result = materialize_design_class_requests(repo_root=Path.cwd(), output_root=tmp_path)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    by_id = {row["design_class_id"]: row for row in manifest["design_classes"]}

    baseline_root = tmp_path / "baseline"
    class_root = Path(by_id["eco1_rt_clade9_plurality25_contact6a_v1"]["class_root"])
    _write_candidate_table(
        baseline_root / "candidate_table.parquet",
        [
            _candidate("thread_candidate_a", "sha256:aaa", rank=1),
            _candidate("thread_candidate_b", "sha256:bbb", rank=2),
        ],
    )
    _write_candidate_table(
        class_root / "candidate_table.parquet",
        [
            _candidate("thread_candidate_b_dup", "sha256:bbb", rank=1),
            _candidate("thread_candidate_c", "sha256:ccc", rank=2),
        ],
    )

    pool = materialize_design_class_candidate_pool(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        baseline_candidate_table_path=baseline_root / "candidate_table.parquet",
    )
    rows = pq.read_table(pool.candidate_pool_path).to_pylist()
    pool_manifest = yaml.safe_load(pool.manifest_path.read_text(encoding="utf-8"))

    assert [row["sequence_hash"] for row in rows] == ["sha256:aaa", "sha256:bbb", "sha256:ccc"]
    assert pool_manifest["generated_candidate_table_count"] == 1
    assert pool_manifest["pending_design_class_ids"] == [
        "eco1_rt_clade9_plurality25_contact8a_v1",
        "eco1_rt_clade9_plurality25_contact10a_v1",
        "eco1_rt_clade9_plurality50_contact5a_v1",
        "eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1",
    ]
    duplicate = next(row for row in rows if row["sequence_hash"] == "sha256:bbb")
    assert duplicate["design_class_id"] == "eco1_rt_clade9_plurality25_contact5a_v1"
    assert duplicate["duplicate_design_class_ids"] == [
        "eco1_rt_clade9_plurality25_contact5a_v1",
        "eco1_rt_clade9_plurality25_contact6a_v1",
    ]
    assert duplicate["duplicate_candidate_ids"] == ["thread_candidate_b", "thread_candidate_b_dup"]


def test_foldcheck_request_cli_reports_baseline_only_pool_without_traceback(tmp_path: Path, capsys) -> None:
    require_ec86kit_source_artifacts()
    materialize_design_class_requests(repo_root=Path.cwd(), output_root=tmp_path)
    materialize_design_class_candidate_pool(repo_root=Path.cwd(), output_root=tmp_path)

    exit_code = main(["--repo-root", str(Path.cwd()), "--output-root", str(tmp_path), "foldcheck-request"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "expanded fold-check request requires at least one generated design-class candidate table" in captured.err
    assert "Traceback" not in captured.err


def _write_candidate_table(path: Path, rows: list[dict[str, object]]) -> None:
    write_candidate_table(path, rows, request_hash="sha256:test")


def _candidate(candidate_id: str, sequence_hash: str, *, rank: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_sample_id": f"{candidate_id}_sample",
        "backend_run_id": "backend",
        "request_hash": "sha256:test",
        "sequence_hash": sequence_hash,
        "sequence": "ACDE",
        "score": float(rank),
        "global_score": float(rank),
        "seq_recovery": 0.5,
        "seed": 101,
        "temperature": 0.1,
        "sample_index": rank,
        "duplicate_sample_count": 1,
        "mutation_count": 1,
        "mutable_mutation_count": 1,
        "protected_mutation_count": 0,
        "outside_mutable_positions": [],
        "canonical_mutations": ["A1C"],
        "status": "accepted",
        "rank": rank,
    }
