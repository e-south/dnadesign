"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts/conservation_sources.py

Typed accessors for the Eco1 conservation source contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import load_yaml_mapping


@dataclass(frozen=True)
class ConservationSourceContract:
    """Selected source groups and shared MSA/source-sequence policy."""

    sources: Mapping[str, Any]
    profile_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    provider_accession_patterns: Mapping[str, tuple[str, ...]]
    source_groups: Mapping[str, Mapping[str, Any]]
    target_row_id: str
    target_sequence_hash: str
    known_public_target_accession: str
    accession_field: str


def load_conservation_source_contract(path: Path) -> ConservationSourceContract:
    """Load and parse ``conservation-sources.yaml``."""

    return parse_conservation_source_contract(load_yaml_mapping(path))


def parse_conservation_source_contract(sources: Mapping[str, Any]) -> ConservationSourceContract:
    """Parse the selected conservation source contract and fail on missing authority."""

    profile_ids = tuple(_required_profile_ids(sources))
    provider_ids = tuple(_required_provider_ids(sources))
    source_groups = _source_groups_by_id(sources)
    return ConservationSourceContract(
        sources=sources,
        profile_ids=profile_ids,
        provider_ids=provider_ids,
        provider_accession_patterns=_provider_accession_patterns(sources, provider_ids),
        source_groups=source_groups,
        target_row_id=require_nested_text(sources, ("alignment_policy", "target_row_id")),
        target_sequence_hash=require_nested_text(sources, ("target_sequence", "reference_sequence_hash")),
        known_public_target_accession=_known_public_target_accession(sources),
        accession_field=_accession_field(source_groups=source_groups, profile_ids=profile_ids),
    )


def validate_profile_provider_contract(contract: ConservationSourceContract) -> None:
    """Verify every selected source group uses the phase acceptance providers."""

    for profile_id in contract.profile_ids:
        group = require_mapping(contract.source_groups.get(profile_id), f"source group {profile_id}")
        group_provider_ids = group.get("provider_ids")
        if group_provider_ids != list(contract.provider_ids):
            raise ValueError(f"source group {profile_id!r} provider_ids must match phase1_acceptance")


def _required_profile_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    profile_ids = acceptance.get("required_profile_ids")
    if not isinstance(profile_ids, list) or not all(isinstance(item, str) and item for item in profile_ids):
        raise ValueError("phase1_acceptance.required_profile_ids must be a non-empty list of strings")
    return list(profile_ids)


def _required_provider_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    provider_ids = acceptance.get("required_provider_ids")
    if not isinstance(provider_ids, list) or not all(isinstance(item, str) and item for item in provider_ids):
        raise ValueError("phase1_acceptance.required_provider_ids must be a non-empty list of strings")
    return list(provider_ids)


def _source_groups_by_id(sources: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = sources.get("source_groups")
    if not isinstance(groups, list):
        raise ValueError("conservation-sources.yaml must declare source_groups")
    grouped: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        mapping = require_mapping(group, "source group")
        grouped[require_text(mapping, "profile_id")] = mapping
    return grouped


def _provider_accession_patterns(sources: Mapping[str, Any], provider_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    providers = sources.get("sequence_providers")
    if not isinstance(providers, list):
        raise ValueError("conservation-sources.yaml must declare sequence_providers")
    providers_by_id: dict[str, Mapping[str, Any]] = {}
    for provider in providers:
        mapping = require_mapping(provider, "sequence provider")
        providers_by_id[require_text(mapping, "id")] = mapping

    patterns_by_provider: dict[str, tuple[str, ...]] = {}
    for provider_id in provider_ids:
        provider = require_mapping(providers_by_id.get(provider_id), f"sequence provider {provider_id}")
        patterns = provider.get("accession_patterns")
        if not isinstance(patterns, list) or not all(isinstance(item, str) and item for item in patterns):
            raise ValueError(f"sequence provider {provider_id!r} must declare non-empty accession_patterns")
        patterns_by_provider[provider_id] = tuple(patterns)
    return patterns_by_provider


def _accession_field(
    *,
    source_groups: Mapping[str, Mapping[str, Any]],
    profile_ids: Sequence[str],
) -> str:
    fields = {
        require_text(
            require_mapping(source_groups[profile_id].get("roster_source"), "roster_source"),
            "accession_field",
        )
        for profile_id in profile_ids
    }
    if len(fields) != 1:
        raise ValueError("all selected source groups must use one accession_field")
    return next(iter(fields))


def _known_public_target_accession(sources: Mapping[str, Any]) -> str:
    target = require_mapping(sources.get("target_sequence"), "target_sequence")
    known = require_mapping(target.get("known_public_accession"), "known_public_accession")
    return require_text(known, "accession")


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a contract error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def require_nested_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Read a nested non-empty string field from a mapping."""

    current: Any = payload
    for field in fields:
        current = require_mapping(current, ".".join(fields)).get(field)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{'.'.join(fields)} must be a non-empty string")
    return current.strip()


def require_text(payload: Mapping[str, Any], field: str) -> str:
    """Read a non-empty string field from a mapping."""

    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
