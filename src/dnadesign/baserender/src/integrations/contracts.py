"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/contracts.py

Descriptor contracts for built-in BaseRender integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..core import InputEnvelope, RenderContractDescriptor

PolicyNormalizer = Callable[[Mapping[str, Any], str], dict[str, Any]]
AdapterFactory = Callable[[Any, str], Any]
AdapterPathResolver = Callable[[str, Any], str]
TransformFactory = Callable[[Mapping[str, Any]], Any]
TransformParamsValidator = Callable[[Mapping[str, Any], str], None]
TransformPathResolver = Callable[[str, Any], str]
StyleFactory = Callable[[], dict[str, object]]


def passthrough_policies(policies: Mapping[str, Any], _ctx: str) -> dict[str, Any]:
    return dict(policies)


def accept_transform_params(_params: Mapping[str, Any], _ctx: str) -> None:
    return None


@dataclass(frozen=True)
class AdapterDescriptor:
    kind: str
    owner_tool: str | None
    contract_kind: str
    supported_renderers: tuple[str, ...]
    supported_alphabets: tuple[str, ...]
    factory: AdapterFactory
    docs_slug: str
    allowed_config_columns: tuple[str, ...]
    required_config_columns: tuple[str, ...]
    required_source_columns: tuple[str, ...]
    optional_source_columns: tuple[str, ...] = ()
    allowed_policy_keys: tuple[str, ...] = ()
    resolved_path_columns: tuple[str, ...] = ()
    normalize_policies: PolicyNormalizer = passthrough_policies
    sensitivity: Literal["public", "private"] = "public"
    input_envelope: InputEnvelope | None = None
    output_kinds: tuple[Literal["images", "video"], ...] = ("images", "video")
    image_output_modes: tuple[Literal["directory", "single_file"], ...] = (
        "directory",
        "single_file",
    )
    max_grid_records: int | None = None
    validation_scope: Literal["row", "document"] = "row"


@dataclass(frozen=True)
class TransformDescriptor:
    name: str
    owner_tool: str | None
    factory: TransformFactory
    docs_slug: str
    allowed_params: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ()
    path_params: tuple[str, ...] = ()
    validate_params: TransformParamsValidator = accept_transform_params


@dataclass(frozen=True)
class SequencePanelDefaults:
    adapter_kind: str
    supported_profiles: tuple[str, ...]
    columns: tuple[tuple[str, object], ...]
    policies: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class StyleProfileDescriptor:
    name: str
    owner_tool: str | None
    docs_slug: str
    style_factory: StyleFactory


@dataclass(frozen=True)
class IntegrationProvider:
    name: str
    adapters: tuple[AdapterDescriptor, ...] = ()
    transforms: tuple[TransformDescriptor, ...] = ()
    sequence_panels: tuple[SequencePanelDefaults, ...] = ()
    style_profiles: tuple[StyleProfileDescriptor, ...] = ()
    render_contracts: tuple[RenderContractDescriptor, ...] = ()


__all__ = [
    "AdapterDescriptor",
    "AdapterFactory",
    "AdapterPathResolver",
    "IntegrationProvider",
    "PolicyNormalizer",
    "SequencePanelDefaults",
    "StyleFactory",
    "StyleProfileDescriptor",
    "TransformDescriptor",
    "TransformFactory",
    "TransformParamsValidator",
    "TransformPathResolver",
    "accept_transform_params",
    "passthrough_policies",
]
