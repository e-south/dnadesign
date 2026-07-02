"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/artifact_chain.py

Eco1 RT repack contract validation primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _as_string_list,
    _require_known_phase,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _CONTRACT_STATES,
    _EXPECTED_ARTIFACT_ORDER,
    _REQUIRED_ARTIFACT_INVARIANTS,
    _SHARED_ARTIFACT_FIELDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_artifact_chain_schema_payload(
    schema: Mapping[str, Any],
    *,
    phase: str = "phase0_scaffold",
) -> ContractReport:
    """Validate the study-local artifact-chain schema shape."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []

    if schema.get("schema_id") != "thread_artifact_chain_v1":
        issues.append(
            ContractIssue(
                check_id="thread.artifact_chain.schema_id_mismatch",
                message="artifact-chain schema_id must be 'thread_artifact_chain_v1'",
                path="schema_id",
            )
        )

    status = str(schema.get("status", "")).strip()
    if status not in _CONTRACT_STATES:
        issues.append(
            ContractIssue(
                check_id="thread.artifact_chain.unknown_state",
                message=f"artifact-chain schema status must be one of {sorted(_CONTRACT_STATES)}",
                path="status",
            )
        )

    artifact_order = tuple(_as_string_list(schema.get("artifact_order")))
    artifacts = schema.get("artifacts")
    artifact_keys = tuple(str(key) for key in artifacts) if isinstance(artifacts, Mapping) else ()
    if artifact_order != _EXPECTED_ARTIFACT_ORDER:
        issues.append(
            ContractIssue(
                check_id="thread.artifact_chain.order_mismatch",
                message="artifact order must match the current thread artifact chain exactly",
                path="artifact_order",
            )
        )
    if set(artifact_order) != set(artifact_keys):
        issues.append(
            ContractIssue(
                check_id="thread.artifact_chain.artifact_key_mismatch",
                message="artifact_order and artifacts keys must name the same artifacts",
                path="artifacts",
            )
        )

    shared_fields = set(_as_string_list(schema.get("shared_required_fields")))
    missing_shared_fields = sorted(_SHARED_ARTIFACT_FIELDS - shared_fields)
    if missing_shared_fields:
        issues.append(
            ContractIssue(
                check_id="thread.artifact_chain.missing_shared_fields",
                message=f"shared_required_fields is missing {missing_shared_fields}",
                path="shared_required_fields",
            )
        )

    if isinstance(artifacts, Mapping):
        for artifact_name in artifact_order:
            artifact_contract = artifacts.get(artifact_name)
            if not isinstance(artifact_contract, Mapping):
                issues.append(
                    ContractIssue(
                        check_id="thread.artifact_chain.missing_artifact_contract",
                        message=f"artifact {artifact_name!r} is missing its contract block",
                        path=f"artifacts.{artifact_name}",
                    )
                )
                continue
            has_required_fields = bool(_as_string_list(artifact_contract.get("required_fields")))
            has_required_columns = bool(_as_string_list(artifact_contract.get("required_columns")))
            if not has_required_fields and not has_required_columns:
                issues.append(
                    ContractIssue(
                        check_id="thread.artifact_chain.missing_artifact_shape",
                        message=f"artifact {artifact_name!r} must declare required_fields or required_columns",
                        path=f"artifacts.{artifact_name}",
                    )
                )

    invariants = set(_as_string_list(schema.get("invariants")))
    for invariant, check_id in _REQUIRED_ARTIFACT_INVARIANTS.items():
        if invariant not in invariants:
            issues.append(
                ContractIssue(
                    check_id=check_id,
                    message=f"artifact-chain schema must declare invariant {invariant!r}",
                    path="invariants",
                )
            )

    return ContractReport(phase=phase, issues=tuple(issues))
