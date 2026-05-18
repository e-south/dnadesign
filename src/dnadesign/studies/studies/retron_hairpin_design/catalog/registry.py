"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/catalog/registry.py

Study registry loading for Retron MSD design references.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.contracts.sequence import MsdDesignReferenceV1

from .msd_ids import ParsedMsdConstructLabel

_REGISTRY_RELATIVE_PATH = Path("compiler") / "catalog" / "msd_design_registry.yaml"


class RetronMsdRegistryError(ValueError):
    """Raised when the Retron MSD study registry is missing or malformed."""


@dataclass(frozen=True)
class RetronMsdRegistry:
    path: Path
    payloads: dict[str, dict[str, Any]]
    caps: dict[str, dict[str, Any]]
    constructs: dict[str, dict[str, Any]]

    def build_reference(self, parsed: ParsedMsdConstructLabel) -> MsdDesignReferenceV1:
        return self.build_reference_from_parts(parsed)

    def build_reference_from_parts(
        self,
        parsed: ParsedMsdConstructLabel,
        *,
        cap_metadata: dict[str, Any] | None = None,
        scar_nick_metadata: dict[str, Any] | None = None,
        source_notes: str | None = None,
    ) -> MsdDesignReferenceV1:
        payload = self._require_mapping(self.payloads, parsed.payload_id, label="payload")
        cap = self._optional_mapping(self.caps, parsed.cap_id, label="cap")
        if cap is None and cap_metadata is None:
            self._require_mapping(self.caps, parsed.cap_id, label="cap")
        cap = dict(cap or {})
        if cap_metadata is not None:
            if not isinstance(cap_metadata, dict):
                raise RetronMsdRegistryError(f"cap metadata must be a mapping: {parsed.cap_id}")
            cap.update(cap_metadata)
        construct = self.constructs.get(parsed.construct_id, {})
        if not isinstance(construct, dict):
            raise RetronMsdRegistryError(f"construct registry entry must be a mapping: {parsed.construct_id}")
        scar_nick = construct.get("scar_nick", {})
        if scar_nick is None:
            scar_nick = {}
        if not isinstance(scar_nick, dict):
            raise RetronMsdRegistryError(f"construct scar_nick entry must be a mapping: {parsed.construct_id}")
        scar_nick = dict(scar_nick)
        if scar_nick_metadata is not None:
            if not isinstance(scar_nick_metadata, dict):
                raise RetronMsdRegistryError(f"scar_nick metadata must be a mapping: {parsed.construct_id}")
            scar_nick.update(scar_nick_metadata)
        return MsdDesignReferenceV1.model_validate(
            {
                "construct_id": parsed.construct_id,
                "construct_label": parsed.construct_label,
                "msd_design_id": parsed.msd_design_id,
                "payload_or_target": {
                    "id": parsed.payload_id,
                    "display_name": payload.get("display_name"),
                },
                "cap": {
                    "id": parsed.cap_id,
                    "source_construct": cap.get("source_construct"),
                    "display_name": cap.get("display_name"),
                    "snapback_topology": cap.get("snapback_topology"),
                },
                "scar_nick": {
                    "left_base": parsed.left_base,
                    "right_base": parsed.right_base,
                    "profile_s3s2s1s0": parsed.profile_s3s2s1s0,
                    "route_status": scar_nick.get("route_status", "unresolved"),
                    "nick_orientation": scar_nick.get("nick_orientation"),
                    "nickase": scar_nick.get("nickase"),
                    "route_note": scar_nick.get("route_note"),
                },
                "source_notes": source_notes or construct.get("source_notes"),
            }
        )

    @staticmethod
    def _require_mapping(source: dict[str, dict[str, Any]], key: str, *, label: str) -> dict[str, Any]:
        value = source.get(key)
        if not isinstance(value, dict):
            available = ", ".join(sorted(source)) or "(none)"
            raise RetronMsdRegistryError(f"Unknown {label} '{key}' in MSD construct label. Available: {available}.")
        return value

    @staticmethod
    def _optional_mapping(source: dict[str, dict[str, Any]], key: str, *, label: str) -> dict[str, Any] | None:
        value = source.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RetronMsdRegistryError(f"{label} registry entry must be a mapping: {key}")
        return value


def load_retron_msd_registry(study_dir: str | Path) -> RetronMsdRegistry:
    study_path = Path(study_dir).expanduser().resolve()
    registry_path = study_path / _REGISTRY_RELATIVE_PATH
    if not registry_path.is_file():
        raise RetronMsdRegistryError(f"Retron MSD registry not found: {registry_path}")
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RetronMsdRegistryError(f"Retron MSD registry is invalid YAML: {registry_path}") from exc
    if not isinstance(payload, dict):
        raise RetronMsdRegistryError(f"Retron MSD registry must be a mapping: {registry_path}")
    if payload.get("contract") != "retron_msd_design_registry_v1":
        raise RetronMsdRegistryError("Retron MSD registry contract must be retron_msd_design_registry_v1.")
    payloads = _mapping_of_mappings(payload.get("payloads", {}), label="payloads")
    caps = _mapping_of_mappings(payload.get("caps", {}), label="caps")
    constructs = _mapping_of_mappings(payload.get("constructs", {}), label="constructs")
    return RetronMsdRegistry(path=registry_path, payloads=payloads, caps=caps, constructs=constructs)


def _mapping_of_mappings(raw: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise RetronMsdRegistryError(f"Retron MSD registry field '{label}' must be a mapping.")
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise RetronMsdRegistryError(f"Retron MSD registry field '{label}' has a blank key.")
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise RetronMsdRegistryError(f"Retron MSD registry entry '{label}.{key}' must be a mapping.")
        out[key] = dict(value)
    return out


__all__ = ["RetronMsdRegistry", "RetronMsdRegistryError", "load_retron_msd_registry"]
