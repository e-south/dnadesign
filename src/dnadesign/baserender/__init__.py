"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/__init__.py

Public BaseRender package facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dnadesign.baserender.src.cli import app  # noqa: F401
    from dnadesign.baserender.src.config import (  # noqa: F401
        RenderContractDescriptor,
        RenderJobV4,
        Style,
        list_style_presets,
        resolve_preset_path,
        resolve_style,
    )
    from dnadesign.baserender.src.core import (  # noqa: F401
        ContractError,
        Display,
        Effect,
        Feature,
        LayoutError,
        Record,
        RenderingError,
        SchemaError,
        Span,
    )
    from dnadesign.baserender.src.public import (  # noqa: F401
        BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
        BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
        SequencePanelConfig,
        SequencePanelDiagnostics,
        SequencePanelImage,
        adapt_record,
        adapt_records,
        get_adapter_descriptor,
        get_render_contract_descriptor,
        get_renderer_descriptor,
        get_style_profile_descriptor,
        get_transform_descriptor,
        list_adapters,
        list_render_contracts,
        list_renderers,
        list_style_profiles,
        list_transforms,
        load_record_from_parquet,
        load_records_from_parquet,
        render,
        render_parquet_record_figure,
        render_record_figure,
        render_record_grid_figure,
        render_sequence_panel_image,
        run_job,
        run_render_job,
        sequence_panel_config_for_adapter,
        style_profile_overrides,
        validate_job,
        validate_render_job,
    )
    from dnadesign.baserender.src.render.palette import Palette  # noqa: F401
    from dnadesign.baserender.src.runtime import initialize_runtime  # noqa: F401

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "initialize_runtime": ("dnadesign.baserender.src.runtime", "initialize_runtime"),
    "app": ("dnadesign.baserender.src.cli", "app"),
    "adapt_record": ("dnadesign.baserender.src.public", "adapt_record"),
    "adapt_records": ("dnadesign.baserender.src.public", "adapt_records"),
    "RenderJobV4": ("dnadesign.baserender.src.config", "RenderJobV4"),
    "RenderContractDescriptor": ("dnadesign.baserender.src.config", "RenderContractDescriptor"),
    "Style": ("dnadesign.baserender.src.config", "Style"),
    "resolve_style": ("dnadesign.baserender.src.config", "resolve_style"),
    "resolve_preset_path": ("dnadesign.baserender.src.config", "resolve_preset_path"),
    "list_style_presets": ("dnadesign.baserender.src.config", "list_style_presets"),
    "run_render_job": ("dnadesign.baserender.src.public", "run_render_job"),
    "validate_render_job": ("dnadesign.baserender.src.public", "validate_render_job"),
    "run_job": ("dnadesign.baserender.src.public", "run_job"),
    "validate_job": ("dnadesign.baserender.src.public", "validate_job"),
    "list_adapters": ("dnadesign.baserender.src.public.catalog", "list_adapters"),
    "list_renderers": ("dnadesign.baserender.src.public.catalog", "list_renderers"),
    "list_style_profiles": ("dnadesign.baserender.src.public.catalog", "list_style_profiles"),
    "list_render_contracts": ("dnadesign.baserender.src.public.catalog", "list_render_contracts"),
    "get_adapter_descriptor": ("dnadesign.baserender.src.public.catalog", "get_adapter_descriptor"),
    "get_renderer_descriptor": ("dnadesign.baserender.src.public.catalog", "get_renderer_descriptor"),
    "get_style_profile_descriptor": (
        "dnadesign.baserender.src.public.catalog",
        "get_style_profile_descriptor",
    ),
    "get_transform_descriptor": ("dnadesign.baserender.src.public.catalog", "get_transform_descriptor"),
    "list_transforms": ("dnadesign.baserender.src.public.catalog", "list_transforms"),
    "get_render_contract_descriptor": ("dnadesign.baserender.src.public.catalog", "get_render_contract_descriptor"),
    "render": ("dnadesign.baserender.src.public", "render"),
    "style_profile_overrides": ("dnadesign.baserender.src.public", "style_profile_overrides"),
    "Record": ("dnadesign.baserender.src.core", "Record"),
    "Feature": ("dnadesign.baserender.src.core", "Feature"),
    "Effect": ("dnadesign.baserender.src.core", "Effect"),
    "Display": ("dnadesign.baserender.src.core", "Display"),
    "Span": ("dnadesign.baserender.src.core", "Span"),
    "Palette": ("dnadesign.baserender.src.render.palette", "Palette"),
    "SchemaError": ("dnadesign.baserender.src.core", "SchemaError"),
    "ContractError": ("dnadesign.baserender.src.core", "ContractError"),
    "LayoutError": ("dnadesign.baserender.src.core", "LayoutError"),
    "RenderingError": ("dnadesign.baserender.src.core", "RenderingError"),
    "load_records_from_parquet": ("dnadesign.baserender.src.public", "load_records_from_parquet"),
    "load_record_from_parquet": ("dnadesign.baserender.src.public", "load_record_from_parquet"),
    "render_record_figure": ("dnadesign.baserender.src.public", "render_record_figure"),
    "render_record_grid_figure": ("dnadesign.baserender.src.public", "render_record_grid_figure"),
    "render_parquet_record_figure": ("dnadesign.baserender.src.public", "render_parquet_record_figure"),
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID": (
        "dnadesign.baserender.src.public",
        "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID",
    ),
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION": (
        "dnadesign.baserender.src.public",
        "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION",
    ),
    "SequencePanelConfig": ("dnadesign.baserender.src.public", "SequencePanelConfig"),
    "SequencePanelDiagnostics": ("dnadesign.baserender.src.public", "SequencePanelDiagnostics"),
    "SequencePanelImage": ("dnadesign.baserender.src.public", "SequencePanelImage"),
    "sequence_panel_config_for_adapter": ("dnadesign.baserender.src.public", "sequence_panel_config_for_adapter"),
    "render_sequence_panel_image": ("dnadesign.baserender.src.public", "render_sequence_panel_image"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
