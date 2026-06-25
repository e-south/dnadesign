"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/thread_plan/report.py

Thread-plan validation orchestration for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _resolve_output_root
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import _CONTRACT_ROOT
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.expected import (
    expected_request_fields,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.io import (
    load_yaml_mapping,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.validation import (
    validate_metadata,
    validate_request_fields,
    validate_request_hash,
    validate_upstream_hashes,
)


def validate_thread_plan_content(
    path: Path, *, repo_root: Path, output_root: Path | None = None
) -> list[ContractIssue]:
    """Validate thread_plan.yaml against the current mask set and profile fixture."""

    issues: list[ContractIssue] = []
    structure_root = _resolve_output_root(repo_root, output_root)
    profile_path = repo_root / _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"
    mask_set_path = structure_root / "mask_set.yaml"
    backbone_bundle_path = structure_root / "backbone_bundle.yaml"
    residue_map_path = structure_root / "residue_map.parquet"
    plan = load_yaml_mapping(path)
    profile = load_yaml_mapping(profile_path)
    mask_set = load_yaml_mapping(mask_set_path)

    validate_metadata(issues, plan=plan, path=path)
    validate_upstream_hashes(
        issues,
        plan=plan,
        path=path,
        expected_paths={
            "profile": profile_path,
            "backbone_bundle": backbone_bundle_path,
            "residue_map": residue_map_path,
            "mask_set": mask_set_path,
        },
    )
    expected = expected_request_fields(profile=profile, mask_set=mask_set, mask_set_path=mask_set_path)
    validate_request_fields(issues, plan=plan, expected=expected, path=path)
    validate_request_hash(issues, plan=plan, path=path)
    return issues
