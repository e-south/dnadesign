"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/thread_plan/validation.py

Thread-plan field validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.constants import (
    EXPLICIT_NO_FALLBACK,
    MASK_POLICY_ID,
    THREAD_PLAN_ARTIFACT_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.io import (
    request_hash,
    sha256_file,
)


def validate_metadata(issues: list[ContractIssue], *, plan: Mapping[str, Any], path: Path) -> None:
    expected = {
        "schema_id": "thread.thread_plan",
        "schema_version": 1,
        "artifact_id": THREAD_PLAN_ARTIFACT_ID,
        "status": "materialized",
        "mask_policy_id": MASK_POLICY_ID,
        "fallback_policy": EXPLICIT_NO_FALLBACK,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.thread_plan_metadata_mismatch",
                    message=f"thread_plan.yaml field {key!r} must equal {value!r}",
                    path=str(path),
                )
            )
    for field in ("created_by", "created_at", "upstream_artifact_hashes", "request_hash"):
        if field not in plan:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.thread_plan_missing_lifecycle_field",
                    message=f"thread_plan.yaml must declare {field!r}",
                    path=str(path),
                )
            )


def validate_upstream_hashes(
    issues: list[ContractIssue],
    *,
    plan: Mapping[str, Any],
    path: Path,
    expected_paths: Mapping[str, Path],
) -> None:
    hashes = plan.get("upstream_artifact_hashes")
    if not isinstance(hashes, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_missing_upstream_hashes",
                message="thread_plan.yaml must declare upstream_artifact_hashes",
                path=str(path),
            )
        )
        return
    for key, artifact_path in expected_paths.items():
        expected = "sha256:" + sha256_file(artifact_path)
        if hashes.get(key) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.thread_plan_upstream_hash_mismatch",
                    message=f"thread_plan.yaml upstream hash {key!r} must match current artifact",
                    path=str(path),
                )
            )


def validate_request_fields(
    issues: list[ContractIssue],
    *,
    plan: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: Path,
) -> None:
    scalar_fields = (
        "profile_id",
        "backend_kind",
        "seed_set",
        "temperature_schedule",
        "batch_id",
        "num_seq_per_target",
        "batch_size",
        "expected_sample_count",
        "fixed_positions",
        "excluded_non_fixed_missing_backbone_positions",
    )
    for field in scalar_fields:
        if plan.get(field) != expected[field]:
            issues.append(
                ContractIssue(
                    check_id=f"eco1_rt.sampling.thread_plan_{field}_mismatch",
                    message=f"thread_plan.yaml field {field!r} must match the profile and mask set",
                    path=str(path),
                )
            )
    _validate_fixed_position_source(issues, plan=plan, expected=expected, path=path)
    if plan.get("mutable_positions") != expected["mutable_positions"]:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_mutable_position_mismatch",
                message="thread_plan.yaml mutable_positions must match mapped non-fixed mask rows only",
                path=str(path),
            )
        )
    _validate_backend_manifest(issues, plan=plan, expected=expected, path=path)


def _validate_fixed_position_source(
    issues: list[ContractIssue],
    *,
    plan: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: Path,
) -> None:
    source = plan.get("fixed_position_source")
    expected_source = expected["fixed_position_source"]
    if not isinstance(source, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_fixed_position_source_mismatch",
                message="thread_plan.yaml fixed_position_source must be a mapping",
                path=str(path),
            )
        )
        return
    for field in ("artifact_id", "hash", "mask_policy_id"):
        if source.get(field) != expected_source[field]:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.sampling.thread_plan_fixed_position_source_mismatch",
                    message=f"thread_plan.yaml fixed_position_source field {field!r} must match the mask set",
                    path=str(path),
                )
            )
    if not str(source.get("path", "")).strip():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_fixed_position_source_mismatch",
                message="thread_plan.yaml fixed_position_source must keep a non-empty source path",
                path=str(path),
            )
        )


def validate_request_hash(issues: list[ContractIssue], *, plan: Mapping[str, Any], path: Path) -> None:
    observed = plan.get("request_hash")
    payload = {key: value for key, value in plan.items() if key != "request_hash"}
    expected = request_hash(payload)
    if observed != expected:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_request_hash_mismatch",
                message="thread_plan.yaml request_hash must match the canonical planned request payload",
                path=str(path),
            )
        )


def _validate_backend_manifest(
    issues: list[ContractIssue],
    *,
    plan: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: Path,
) -> None:
    manifest = plan.get("backend_request_manifest")
    if not isinstance(manifest, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_missing_backend_manifest",
                message="thread_plan.yaml must include backend_request_manifest",
                path=str(path),
            )
        )
        return
    manifest_expected = {
        "backend_kind": expected["backend_kind"],
        "profile_id": expected["profile_id"],
        "mask_policy_id": MASK_POLICY_ID,
        "fixed_positions": expected["fixed_positions"],
        "mutable_positions": expected["mutable_positions"],
        "excluded_positions": expected["excluded_non_fixed_missing_backbone_positions"],
        "seed_set": expected["seed_set"],
        "temperature_schedule": expected["temperature_schedule"],
        "batch_id": expected["batch_id"],
        "num_seq_per_target": expected["num_seq_per_target"],
        "batch_size": expected["batch_size"],
        "fallback_policy": EXPLICIT_NO_FALLBACK,
    }
    for field, value in manifest_expected.items():
        if manifest.get(field) != value:
            issues.append(
                ContractIssue(
                    check_id=f"eco1_rt.sampling.thread_plan_backend_manifest_{field}_mismatch",
                    message=f"backend_request_manifest field {field!r} must match the validated thread plan",
                    path=str(path),
                )
            )
    if plan.get("fallback_policy") != EXPLICIT_NO_FALLBACK or manifest.get("fallback_policy") != EXPLICIT_NO_FALLBACK:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_fallback_policy_mismatch",
                message="thread_plan.yaml must use explicit_no_fallback at top level and in backend_request_manifest",
                path=str(path),
            )
        )
