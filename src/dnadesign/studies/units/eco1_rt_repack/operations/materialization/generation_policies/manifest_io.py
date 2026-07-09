"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/manifest_io.py

Shared manifest helpers for Eco1 RT v3 generation-policy materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    load_yaml,
    require_mapping,
    require_text,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    GENERATION_POLICY_VERSION,
    PRIMARY_POLICY_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.pipeline import (
    generation_policy_payload_hash,
)


def load_valid_generation_policy_manifest(path: Path) -> dict[str, Any]:
    """Load a v3 generation-policy manifest and reject legacy design-class ids."""

    if not path.exists():
        raise FileNotFoundError(path)
    manifest = load_yaml(path)
    if manifest.get("schema_id") != "eco1_rt.generation_policy_manifest":
        raise ValueError("generation_policy_manifest.yaml must use schema_id eco1_rt.generation_policy_manifest")
    if manifest.get("generation_policy_version") != GENERATION_POLICY_VERSION:
        raise ValueError(f"generation_policy_version must be {GENERATION_POLICY_VERSION}")
    policies = manifest.get("generation_policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("generation_policy_manifest.yaml must declare generation_policies")
    for policy in policies:
        policy_map = require_mapping(policy, "generation_policies[]")
        policy_id = require_text(policy_map, "policy_id")
        if looks_like_legacy_design_class_id(policy_id):
            raise ValueError(f"legacy design-class id {policy_id!r} is not valid for generation-policy materialization")
        if policy_id not in PRIMARY_POLICY_IDS:
            raise ValueError(f"unknown generation policy id {policy_id!r}")
    observed_hash = manifest.get("policy_manifest_hash")
    without_hash = {key: value for key, value in manifest.items() if key != "policy_manifest_hash"}
    expected_hash = generation_policy_payload_hash(without_hash)
    if observed_hash != expected_hash:
        raise ValueError(f"policy_manifest_hash mismatch: {observed_hash!r} != {expected_hash!r}")
    return manifest


def resolve_recorded_path(root: Path, value: Any) -> Path:
    """Resolve a manifest-recorded path relative to a manifest root."""

    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def looks_like_legacy_design_class_id(policy_id: str) -> bool:
    """Return True for old contact-distance design-class ids."""

    return policy_id.startswith("eco1_rt_") and "contact" in policy_id and policy_id.endswith("_v1")
