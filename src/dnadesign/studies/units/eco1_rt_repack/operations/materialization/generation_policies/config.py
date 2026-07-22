"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/config.py

Generation-policy configuration validation for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DEFAULT_GENERATION_TOTAL_TARGET_RAW,
    DEFAULT_REQUESTED_VARIANTS_PER_POLICY,
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    PRIMARY_POLICY_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    GenerationPolicyConfig,
    GenerationPolicySpec,
)

_POLICY_TEMPLATES: dict[str, dict[str, str]] = {
    DISTAL_SCAFFOLD_POLICY_ID: {
        "open_set_id": "distal_scaffold",
        "alphabet_rule_id": "broad_no_new_cysteine",
        "purpose": "Increase scaffold diversity away from protected and near retained DNA/RNA contexts.",
    },
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID: {
        "open_set_id": "near_dna_rna_gt5_le10_excluding_protected",
        "alphabet_rule_id": "msa_observed_acid_free_basic_polar_neutral",
        "purpose": "Sample non-acidifying near retained DNA/RNA chemistry outside direct contacts.",
    },
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: {
        "open_set_id": "combined_near_acid_free_plus_distal",
        "alphabet_rule_id": "region_specific_near_acid_free_distal_broad",
        "purpose": "Jointly design near retained DNA/RNA chemistry and distal scaffold diversity.",
    },
}


def build_default_generation_policy_config() -> dict[str, Any]:
    """Return the v3 generation-policy config as a plain mapping."""

    return {
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "generation_total_target_raw": DEFAULT_GENERATION_TOTAL_TARGET_RAW,
        "generation_policies": {
            policy_id: {
                "enabled": True,
                "requested_variants": DEFAULT_REQUESTED_VARIANTS_PER_POLICY,
            }
            for policy_id in PRIMARY_POLICY_IDS
        },
    }


def validate_generation_policy_config(payload: Mapping[str, Any]) -> GenerationPolicyConfig:
    """Validate the v3 config and reject retired design-class identifiers."""

    version = payload.get("generation_policy_version")
    if version != GENERATION_POLICY_VERSION:
        raise ValueError(f"generation_policy_version must be {GENERATION_POLICY_VERSION}")
    policy_payloads = payload.get("generation_policies")
    if not isinstance(policy_payloads, Mapping) or not policy_payloads:
        raise ValueError("generation_policies must be a non-empty mapping")

    enabled_policies: list[GenerationPolicySpec] = []
    for policy_id, raw_policy in policy_payloads.items():
        policy_id = _require_text(policy_id, "policy_id")
        if _looks_like_design_class_id(policy_id):
            raise ValueError(
                f"design-class id {policy_id!r} is not a generation-policy id for generation_policy_version "
                f"{GENERATION_POLICY_VERSION}; use one of: {', '.join(PRIMARY_POLICY_IDS)}"
            )
        if policy_id not in PRIMARY_POLICY_IDS:
            raise ValueError(
                f"unknown generation policy id {policy_id!r}; expected one of: {', '.join(PRIMARY_POLICY_IDS)}"
            )
        policy = _require_mapping(raw_policy, f"generation_policies.{policy_id}")
        if policy.get("enabled", True) is not True:
            continue
        requested_variants = _require_positive_int(
            policy.get("requested_variants"),
            f"generation_policies.{policy_id}.requested_variants",
        )
        template = _POLICY_TEMPLATES[policy_id]
        enabled_policies.append(
            GenerationPolicySpec(
                policy_id=policy_id,
                open_set_id=template["open_set_id"],
                alphabet_rule_id=template["alphabet_rule_id"],
                requested_variants=requested_variants,
                purpose=template["purpose"],
            )
        )

    if not enabled_policies:
        raise ValueError("at least one generation policy must be enabled")
    target = _require_positive_int(payload.get("generation_total_target_raw"), "generation_total_target_raw")
    requested_total = sum(policy.requested_variants for policy in enabled_policies)
    if target != requested_total:
        raise ValueError(
            f"generation_total_target_raw must match enabled requested variants: {target} != {requested_total}"
        )
    return GenerationPolicyConfig(
        generation_policy_version=GENERATION_POLICY_VERSION,
        generation_total_target_raw=target,
        enabled_policies=tuple(enabled_policies),
    )


def _looks_like_design_class_id(policy_id: str) -> bool:
    return policy_id.startswith("eco1_rt_") and "contact" in policy_id and policy_id.endswith("_v1")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
