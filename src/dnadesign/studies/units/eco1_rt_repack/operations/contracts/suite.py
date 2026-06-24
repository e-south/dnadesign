"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/suite.py

Top-level checked-in contract validation suite for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.artifact_chain import (
    validate_artifact_chain_schema_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _find_repo_root,
    _load_yaml,
    _merge_reports,
    _require_known_phase,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.conservation import (
    validate_conservation_sources_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _CONTRACT_ROOT,
    _DOCS_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks import (
    validate_conservative_mask_cases_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractReport
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.profile import validate_profile_payload
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure import (
    validate_authority_consistency_payload,
    validate_materialized_structure_artifacts,
    validate_residue_numbering_policy_payload,
    validate_structure_authority_payload,
)


def validate_checked_in_contracts(
    *,
    repo_root: Path | None = None,
    phase: str,
    output_root: Path | None = None,
) -> ContractReport:
    """Validate checked-in Eco1 RT repack contracts for the requested phase."""

    _require_known_phase(phase)
    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()

    profile = _load_yaml(root / _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml")
    profile_schema = _load_yaml(root / _CONTRACT_ROOT / "schemas/eco1-rt-profile.schema.yaml")
    artifact_chain_schema = _load_yaml(root / _CONTRACT_ROOT / "schemas/thread-artifact-chain.schema.yaml")
    mask_cases = _load_yaml(root / _CONTRACT_ROOT / "fixtures/thread/conservative_mask_cases.yaml")
    structure_sources = _load_yaml(root / _DOCS_ROOT / "workbench/provenance/structure-sources.yaml")
    residue_numbering_policy = _load_yaml(root / _DOCS_ROOT / "workbench/provenance/residue-numbering-policy.yaml")
    conservation_sources = _load_yaml(root / _DOCS_ROOT / "workbench/provenance/conservation-sources.yaml")

    reports = (
        validate_profile_payload(profile=profile, schema=profile_schema, phase=phase),
        validate_artifact_chain_schema_payload(artifact_chain_schema, phase=phase),
        validate_conservative_mask_cases_payload(mask_cases, phase=phase),
        validate_structure_authority_payload(structure_sources, phase=phase),
        validate_residue_numbering_policy_payload(
            residue_numbering_policy,
            structure_sources=structure_sources,
            phase=phase,
        ),
        validate_authority_consistency_payload(
            profile=profile,
            structure_sources=structure_sources,
            numbering_policy=residue_numbering_policy,
            phase=phase,
        ),
        validate_conservation_sources_payload(
            conservation_sources,
            profile=profile,
            numbering_policy=residue_numbering_policy,
            phase=phase,
        ),
        validate_materialized_structure_artifacts(
            repo_root=root,
            structure_sources=structure_sources,
            numbering_policy=residue_numbering_policy,
            conservation_sources=conservation_sources,
            profile=profile,
            phase=phase,
            output_root=output_root,
        ),
    )
    return _merge_reports(phase, reports)
