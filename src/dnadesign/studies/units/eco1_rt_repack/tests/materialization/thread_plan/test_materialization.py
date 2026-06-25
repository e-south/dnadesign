"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/thread_plan/test_materialization.py

Thread-plan materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import (
    validate_thread_plan_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request import (
    materialize_proteinmpnn_request,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import materialize_thread_plan
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_thread_plan_materializer_writes_explicit_non_fallback_request(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    mask_result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)

    plan = _load_yaml(result.thread_plan_path)
    mask_set = _load_yaml(mask_result.mask_set_path)
    expected_mutable = [row["canonical_position"] for row in mask_set["residues"] if row["non_fixed"]]
    expected_fixed = [row["canonical_position"] for row in mask_set["residues"] if row["protected"]]
    expected_missing = [row["canonical_position"] for row in mask_set["residues"] if row["non_fixed_missing_backbone"]]

    assert plan["schema_id"] == "thread.thread_plan"
    assert plan["status"] == "materialized"
    assert plan["profile_id"] == "eco1_rt_v1"
    assert plan["mask_policy_id"] == "eco1_rt_clade9_plurality25_direct_contact5a_v1"
    assert plan["backend_kind"] == "proteinmpnn"
    assert plan["fallback_policy"] == "explicit_no_fallback"
    assert plan["seed_set"] == [101, 202, 303]
    assert plan["temperature_schedule"] == [0.1, 0.3]
    assert plan["batch_id"] == "eco1_rt_p25_5a_n96_20260624"
    assert plan["num_seq_per_target"] == 16
    assert plan["batch_size"] == 1
    assert plan["expected_sample_count"] == 96
    assert plan["fixed_position_source"]["path"] == str(mask_result.mask_set_path)
    assert plan["fixed_positions"] == expected_fixed
    assert plan["mutable_positions"] == expected_mutable
    assert plan["excluded_non_fixed_missing_backbone_positions"] == expected_missing
    assert set(expected_missing).isdisjoint(plan["mutable_positions"])
    assert plan["backend_request_manifest"]["fixed_positions"] == expected_fixed
    assert plan["backend_request_manifest"]["mutable_positions"] == expected_mutable
    assert plan["backend_request_manifest"]["excluded_positions"] == expected_missing
    assert plan["request_hash"].startswith("sha256:")

    issues = validate_thread_plan_content(
        result.thread_plan_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )
    assert issues == []


def test_thread_plan_validator_rejects_missing_backbone_as_mutable(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    plan = _load_yaml(result.thread_plan_path)
    plan["mutable_positions"].append(1)
    plan["backend_request_manifest"]["mutable_positions"].append(1)
    result.thread_plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    issues = validate_thread_plan_content(
        result.thread_plan_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )

    assert "eco1_rt.sampling.thread_plan_mutable_position_mismatch" in {issue.check_id for issue in issues}


def test_phase2_with_thread_plan_reaches_sample_table_gate(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)

    report = validate_checked_in_contracts(
        repo_root=repo_root(), phase="phase2_real_backend_ingest", output_root=tmp_path
    )

    check_ids = {issue.check_id for issue in report.issues}
    assert report.passed is False
    assert "eco1_rt.sampling.thread_plan_not_materialized" not in check_ids
    assert "eco1_rt.sampling.proteinmpnn_request_not_materialized" not in check_ids
    assert "eco1_rt.sampling.sample_table_not_materialized" in check_ids


def test_phase2_without_thread_plan_fails_at_thread_plan_gate(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)

    report = validate_checked_in_contracts(
        repo_root=repo_root(), phase="phase2_real_backend_ingest", output_root=tmp_path
    )

    check_ids = {issue.check_id for issue in report.issues}
    assert report.passed is False
    assert "eco1_rt.sampling.thread_plan_not_materialized" in check_ids
    assert "eco1_rt.sampling.proteinmpnn_request_not_materialized" in check_ids
    assert "eco1_rt.sampling.sample_table_not_materialized" in check_ids


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
