"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/models.py

Neutral data models for metadata-driven ops status surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FIELD_TYPES = frozenset({"str", "path", "int", "bool", "enum"})
_PATH_BASES = frozenset({"repo", "manifest", "cwd"})
_STATUS_STATES = frozenset({"ok", "attention", "missing"})


@dataclass(frozen=True)
class InputFieldSpec:
    name: str
    cli_flag: str
    placeholder: str
    summary: str
    type: Literal["str", "path", "int", "bool", "enum"] = "str"
    required: bool = True
    default: object | None = None
    choices: tuple[str, ...] = ()
    path_base: Literal["repo", "manifest", "cwd"] | None = None
    scaffold_required: bool | None = None
    manifest_key: str | None = None

    def __post_init__(self) -> None:
        normalized_name = str(self.name or "").strip()
        if not normalized_name:
            raise ValueError("status input field spec must define a non-empty name")
        normalized_flag = str(self.cli_flag or "").strip()
        if not normalized_flag.startswith("--"):
            raise ValueError(f"status input field spec {normalized_name} must define a --flag cli_flag")
        normalized_type = str(self.type or "").strip()
        if normalized_type not in _FIELD_TYPES:
            raise ValueError(
                f"status input field spec {normalized_name} has unsupported type {normalized_type!r}: "
                f"{sorted(_FIELD_TYPES)}"
            )
        if self.path_base is not None and self.path_base not in _PATH_BASES:
            raise ValueError(
                f"status input field spec {normalized_name} has unsupported path_base {self.path_base!r}: "
                f"{sorted(_PATH_BASES)}"
            )
        if normalized_type != "enum" and self.choices:
            raise ValueError(f"status input field spec {normalized_name} only supports choices for enum fields")
        if normalized_type == "enum" and not self.choices:
            raise ValueError(f"status input field spec {normalized_name} must define choices for enum fields")
        if self.default is not None and self.required:
            raise ValueError(f"status input field spec {normalized_name} cannot be both required and defaulted")
        if self.manifest_key is not None and not str(self.manifest_key).strip():
            raise ValueError(f"status input field spec {normalized_name} manifest_key must be non-empty")

    @property
    def resolved_manifest_key(self) -> str:
        return str(self.manifest_key or self.name).strip()

    @property
    def display_required(self) -> bool:
        return bool(self.required or self.scaffold_required)

    def as_dict(self) -> dict[str, str]:
        return {
            "manifest_key": self.resolved_manifest_key,
            "cli_flag": self.cli_flag,
            "placeholder": self.placeholder,
            "summary": self.summary,
        }

    def as_schema_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "manifest_key": self.resolved_manifest_key,
            "cli_flag": self.cli_flag,
            "placeholder": self.placeholder,
            "summary": self.summary,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "choices": list(self.choices),
            "path_base": self.path_base,
        }


@dataclass(frozen=True)
class StatusKindSpec:
    status_kind: str
    provider_id: str
    provider_ref: str
    description: str
    input_schema: tuple[InputFieldSpec, ...] = ()
    notes: tuple[str, ...] = ()
    surface_type: str = "artifact_state"
    cost_class: Literal["cheap", "deep"] = "cheap"
    summary_scope: Literal["repo", "workspace", "host", "cluster"] = "workspace"

    def __post_init__(self) -> None:
        status_kind = str(self.status_kind or "").strip()
        provider_id = str(self.provider_id or "").strip()
        provider_ref = str(self.provider_ref or "").strip()
        if not status_kind:
            raise ValueError("status kind spec must define a non-empty status_kind")
        if not provider_id:
            raise ValueError(f"status kind spec {status_kind} must define a non-empty provider_id")
        if not provider_ref or ":" not in provider_ref:
            raise ValueError(
                f"status kind spec {status_kind} must define provider_ref as module:function, received {provider_ref!r}"
            )
        seen_names: set[str] = set()
        for field_spec in self.input_schema:
            if field_spec.name in seen_names:
                raise ValueError(f"status kind spec {status_kind} defines duplicate input field: {field_spec.name}")
            seen_names.add(field_spec.name)

    @property
    def required_inputs(self) -> tuple[InputFieldSpec, ...]:
        return tuple(field for field in self.input_schema if field.display_required)

    @property
    def optional_inputs(self) -> tuple[InputFieldSpec, ...]:
        return tuple(field for field in self.input_schema if not field.display_required)

    def as_inventory_dict(self) -> dict[str, object]:
        return {
            "status_kind": self.status_kind,
            "provider_id": self.provider_id,
            "description": self.description,
            "required_inputs": [field.as_dict() for field in self.required_inputs],
            "optional_inputs": [
                {
                    "cli_flag": field.cli_flag,
                    "summary": field.summary,
                }
                for field in self.optional_inputs
            ],
            "notes": list(self.notes),
            "surface_type": self.surface_type,
            "cost_class": self.cost_class,
            "summary_scope": self.summary_scope,
        }


@dataclass(frozen=True)
class ProcedureStatus:
    registry_id: str
    title: str
    doc_path: str
    owner_boundary: str
    status_kind: str
    label: str | None
    state: str
    summary: str
    evidence: dict[str, object]

    def __post_init__(self) -> None:
        if self.state not in _STATUS_STATES:
            raise ValueError(f"invalid procedure status state: {self.state}")

    def as_dict(self) -> dict[str, object]:
        return {
            "registry_id": self.registry_id,
            "title": self.title,
            "doc_path": self.doc_path,
            "owner_boundary": self.owner_boundary,
            "status_kind": self.status_kind,
            "label": self.label,
            "state": self.state,
            "summary": self.summary,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CampaignScaffoldStep:
    registry_id: str
    title: str
    doc_path: str
    owner_boundary: str
    status_kind: str
    label: str
    input_schema: tuple[InputFieldSpec, ...]

    def manifest_step(self) -> dict[str, object]:
        return {
            "label": self.label,
            "registry_id": self.registry_id,
            "inputs": {
                field.resolved_manifest_key: field.placeholder for field in self.input_schema if field.display_required
            },
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "registry_id": self.registry_id,
            "title": self.title,
            "doc_path": self.doc_path,
            "owner_boundary": self.owner_boundary,
            "status_kind": self.status_kind,
            "label": self.label,
            "required_inputs": [field.as_dict() for field in self.input_schema if field.display_required],
            "optional_inputs": [
                {
                    "cli_flag": field.cli_flag,
                    "summary": field.summary,
                }
                for field in self.input_schema
                if not field.display_required
            ],
            "manifest_step": self.manifest_step(),
        }


@dataclass(frozen=True)
class CampaignScaffold:
    campaign_id: str
    steps: tuple[CampaignScaffoldStep, ...]
    version: int = 2
    path_base: Literal["repo", "manifest", "cwd"] = "repo"

    def as_manifest_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "path_base": self.path_base,
            "campaign_id": self.campaign_id,
            "steps": [step.manifest_step() for step in self.steps],
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "manifest": self.as_manifest_dict(),
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class CampaignStatus:
    manifest_path: Path
    campaign_id: str
    steps: tuple[ProcedureStatus, ...]
    manifest_version: int = 2
    path_base: str | None = None

    def counts(self) -> dict[str, int]:
        state_counts = Counter(step.state for step in self.steps)
        return {state: int(state_counts.get(state, 0)) for state in ("ok", "attention", "missing")}

    def overall_state(self) -> str:
        counts = self.counts()
        if counts["attention"] > 0:
            return "attention"
        if counts["missing"] > 0:
            return "missing"
        return "ok"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "campaign_id": self.campaign_id,
            "manifest_path": str(self.manifest_path),
            "manifest_version": self.manifest_version,
            "path_base": self.path_base,
            "overall_state": self.overall_state(),
            "counts": self.counts(),
            "steps": [step.as_dict() for step in self.steps],
        }
        return payload


__all__ = [
    "CampaignStatus",
    "CampaignScaffold",
    "CampaignScaffoldStep",
    "InputFieldSpec",
    "ProcedureStatus",
    "StatusKindSpec",
]
