"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/models.py

USR registry dataclasses and reserved namespace declarations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

REGISTRY_FILENAME = "registry.yaml"
USR_STATE_NAMESPACE = "usr_state"


@dataclass(frozen=True)
class RegistryColumn:
    name: str
    type: str


@dataclass(frozen=True)
class RegistryEntry:
    namespace: str
    owner: str | None
    description: str | None
    columns: list[RegistryColumn]


USR_STATE_COLUMNS: list[RegistryColumn] = [
    RegistryColumn("usr_state__masked", "bool"),
    RegistryColumn("usr_state__qc_status", "string"),
    RegistryColumn("usr_state__split", "string"),
    RegistryColumn("usr_state__supersedes", "string"),
    RegistryColumn("usr_state__lineage", "list<string>"),
]


def _clone_registry_entries(entries: dict[str, RegistryEntry]) -> dict[str, RegistryEntry]:
    return {
        name: RegistryEntry(
            namespace=entry.namespace,
            owner=entry.owner,
            description=entry.description,
            columns=list(entry.columns),
        )
        for name, entry in entries.items()
    }


def usr_state_entry() -> RegistryEntry:
    return RegistryEntry(
        namespace=USR_STATE_NAMESPACE,
        owner="usr",
        description="Reserved record-state overlay (masked/qc/split/lineage).",
        columns=list(USR_STATE_COLUMNS),
    )
