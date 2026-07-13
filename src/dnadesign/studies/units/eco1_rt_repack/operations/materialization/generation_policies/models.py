"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/models.py

Typed value objects for Eco1 RT generation-policy materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationPolicySpec:
    """One complete ProteinMPNN generation policy."""

    policy_id: str
    open_set_id: str
    alphabet_rule_id: str
    requested_variants: int
    purpose: str


@dataclass(frozen=True)
class GenerationPolicyConfig:
    """Validated generation-policy configuration."""

    generation_policy_version: int
    generation_total_target_raw: int
    enabled_policies: tuple[GenerationPolicySpec, ...]


@dataclass(frozen=True)
class MaterializedGenerationPolicies:
    """Paths emitted by generation-policy materialization."""

    manifest_path: Path
    positions_path: Path
    alphabets_path: Path


@dataclass(frozen=True)
class MaterializedGenerationPolicyRequests:
    """Paths emitted by per-policy ProteinMPNN request materialization."""

    policy_manifest_path: Path
    positions_path: Path
    alphabets_path: Path
    request_manifest_paths: tuple[Path, ...]


@dataclass(frozen=True)
class MaterializedGenerationPolicyCandidatePool:
    """Paths emitted by candidate-pool aggregation."""

    policy_manifest_path: Path
    candidate_pool_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class MaterializedGenerationPolicyFoldCheckRequest:
    """Paths emitted by fold-check request materialization."""

    candidate_pool_path: Path
    candidate_pool_manifest_path: Path
    input_fasta_path: Path
    request_manifest_path: Path
