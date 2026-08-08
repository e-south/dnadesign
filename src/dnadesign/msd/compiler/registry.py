"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/msd/compiler/registry.py

Load a Retron MSD design registry from an explicit artifact path.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.contracts.sequence import MsdDesignReferenceV1

from .identifiers import ParsedMsdConstructLabel
from .yaml_io import DuplicateMappingKeyError, load_unique_yaml


class RetronMsdRegistryError(ValueError):
    """Raised when the Retron MSD study registry is missing or malformed."""


@dataclass(frozen=True)
class RetronMsdRegistry:
    path: Path
    payloads: dict[str, dict[str, Any]]
    caps: dict[str, dict[str, Any]]
    constructs: dict[str, dict[str, Any]]

    def build_reference(
        self,
        parsed: ParsedMsdConstructLabel,
        *,
        payload_metadata: dict[str, Any] | None = None,
        cap_metadata: dict[str, Any] | None = None,
        scar_nick_metadata: dict[str, Any] | None = None,
        variant_metadata: dict[str, Any] | None = None,
        source_notes: str | None = None,
        allow_unregistered_construct: bool = False,
        use_construct_metadata: bool = True,
    ) -> MsdDesignReferenceV1:
        return self.build_reference_from_parts(
            parsed,
            payload_metadata=payload_metadata,
            cap_metadata=cap_metadata,
            scar_nick_metadata=scar_nick_metadata,
            variant_metadata=variant_metadata,
            source_notes=source_notes,
            allow_unregistered_construct=allow_unregistered_construct,
            use_construct_metadata=use_construct_metadata,
        )

    def build_reference_from_parts(
        self,
        parsed: ParsedMsdConstructLabel,
        *,
        payload_metadata: dict[str, Any] | None = None,
        cap_metadata: dict[str, Any] | None = None,
        scar_nick_metadata: dict[str, Any] | None = None,
        variant_metadata: dict[str, Any] | None = None,
        source_notes: str | None = None,
        allow_unregistered_construct: bool = False,
        use_construct_metadata: bool = True,
    ) -> MsdDesignReferenceV1:
        payload = self._optional_mapping(self.payloads, parsed.payload_id, label="payload")
        if payload is None and payload_metadata is None:
            self._require_mapping(self.payloads, parsed.payload_id, label="payload")
        payload = dict(payload or {})
        if payload_metadata is not None:
            if not isinstance(payload_metadata, dict):
                raise RetronMsdRegistryError(f"payload metadata must be a mapping: {parsed.payload_id}")
            payload.update(payload_metadata)
        cap = self._optional_mapping(self.caps, parsed.cap_id, label="cap")
        if cap is None and cap_metadata is None:
            self._require_mapping(self.caps, parsed.cap_id, label="cap")
        cap = dict(cap or {})
        if cap_metadata is not None:
            if not isinstance(cap_metadata, dict):
                raise RetronMsdRegistryError(f"cap metadata must be a mapping: {parsed.cap_id}")
            cap.update(cap_metadata)
        construct = self._optional_mapping(self.constructs, parsed.construct_id, label="construct")
        if construct is None:
            if not allow_unregistered_construct:
                self._require_mapping(self.constructs, parsed.construct_id, label="construct")
            construct = {}
        construct = dict(construct)
        scar_nick = construct.get("scar_nick", {}) if use_construct_metadata else {}
        if scar_nick is None:
            scar_nick = {}
        if not isinstance(scar_nick, dict):
            raise RetronMsdRegistryError(f"construct scar_nick entry must be a mapping: {parsed.construct_id}")
        scar_nick = dict(scar_nick)
        if scar_nick_metadata is not None:
            if not isinstance(scar_nick_metadata, dict):
                raise RetronMsdRegistryError(f"scar_nick metadata must be a mapping: {parsed.construct_id}")
            scar_nick.update(scar_nick_metadata)
        if variant_metadata is not None and not isinstance(variant_metadata, dict):
            raise RetronMsdRegistryError(f"variant metadata must be a mapping: {parsed.construct_id}")
        return MsdDesignReferenceV1.model_validate(
            {
                "construct_id": parsed.construct_id,
                "construct_label": parsed.construct_label,
                "msd_design_id": parsed.msd_design_id,
                "payload_or_target": {
                    "id": parsed.payload_id,
                    "display_name": payload.get("display_name"),
                    "parent_payload_id": payload.get("parent_payload_id"),
                    "payload_trim_id": payload.get("payload_trim_id"),
                    "trim_class": payload.get("trim_class"),
                    "trim_5p_nt": payload.get("trim_5p_nt"),
                    "trim_3p_nt": payload.get("trim_3p_nt"),
                    "retained_parent_span_0": payload.get("retained_parent_span_0"),
                    "pwm_source_ref": payload.get("pwm_source_ref"),
                    "information_content_parent": payload.get("information_content_parent"),
                    "information_content_retained": payload.get("information_content_retained"),
                    "retained_information_fraction": payload.get("retained_information_fraction"),
                    "selection_basis": payload.get("selection_basis"),
                    "protected_positions_or_reason": payload.get("protected_positions_or_reason"),
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
                    "s0_match_required": parsed.s0_match_required,
                    "route_status": scar_nick.get("route_status", "unresolved"),
                    "nick_orientation": scar_nick.get("nick_orientation"),
                    "nickase": scar_nick.get("nickase"),
                    "route_note": scar_nick.get("route_note"),
                },
                "variant_metadata": variant_metadata,
                "source_notes": source_notes or (construct.get("source_notes") if use_construct_metadata else None),
            }
        )

    @staticmethod
    def _require_mapping(source: dict[str, dict[str, Any]], key: str, *, label: str) -> dict[str, Any]:
        value = source.get(key)
        if not isinstance(value, dict):
            available = ", ".join(sorted(source)) or "(none)"
            if label == "cap" and key.startswith("C"):
                raise RetronMsdRegistryError(
                    f"Unknown {label} '{key}' in MSD construct label. C### cap ids are source handles and are not "
                    "inferred from de033 by pattern; add an explicit registry entry or materialize with a "
                    f"5'->3' cap sequence/source. Available: {available}."
                )
            if label == "construct":
                raise RetronMsdRegistryError(
                    f"Unknown construct '{key}' in MSD construct label. Plain labels must reference a registered "
                    "construct; use a typed compiler spec with explicit payload and cap sequences for a manual "
                    f"construct. Available: {available}."
                )
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


def load_retron_msd_registry(path: str | Path) -> RetronMsdRegistry:
    registry_path = Path(path).expanduser().resolve()
    if not registry_path.is_file():
        raise RetronMsdRegistryError(f"Retron MSD registry not found: {registry_path}")
    try:
        payload = load_unique_yaml(registry_path) or {}
    except DuplicateMappingKeyError as exc:
        raise RetronMsdRegistryError(f"Retron MSD registry contains {exc}") from exc
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
