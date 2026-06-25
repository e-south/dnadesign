"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/artifacts.py

Materialized sampling artifact validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.candidate_table import (
    validate_candidate_table_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.proteinmpnn_request import (
    validate_proteinmpnn_request_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.sample_table import (
    validate_sample_table_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan import (
    validate_thread_plan_content,
)


def validate_sampling_artifacts(*, repo_root: Path, structure_root: Path) -> list[ContractIssue]:
    """Validate Phase 2 sampling handoff artifacts without requiring backend execution in Phase 1."""

    issues: list[ContractIssue] = []
    thread_plan = structure_root / "thread_plan.yaml"
    proteinmpnn_request = structure_root / "proteinmpnn_request/request_manifest.yaml"
    sample_table = structure_root / "sample_table.parquet"
    candidate_table = structure_root / "candidate_table.parquet"
    if not thread_plan.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.thread_plan_not_materialized",
                message="Phase 2 backend ingest requires materialized thread_plan.yaml",
                path=str(thread_plan),
            )
        )
    else:
        issues.extend(validate_thread_plan_content(thread_plan, repo_root=repo_root, output_root=structure_root))
    if not proteinmpnn_request.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.proteinmpnn_request_not_materialized",
                message="Phase 2 backend ingest requires materialized proteinmpnn_request/request_manifest.yaml",
                path=str(proteinmpnn_request),
            )
        )
    else:
        issues.extend(
            validate_proteinmpnn_request_content(
                proteinmpnn_request,
                repo_root=repo_root,
                output_root=structure_root,
            )
        )
    if not sample_table.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.sample_table_not_materialized",
                message="Phase 2 backend ingest requires materialized sample_table.parquet",
                path=str(sample_table),
            )
        )
    else:
        issues.extend(validate_sample_table_content(sample_table, output_root=structure_root))
    if not candidate_table.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.sampling.candidate_table_not_materialized",
                message="Phase 2 backend ingest requires materialized candidate_table.parquet",
                path=str(candidate_table),
            )
        )
    else:
        issues.extend(validate_candidate_table_content(candidate_table, output_root=structure_root))
    return issues
