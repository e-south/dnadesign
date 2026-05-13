"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/__init__.py

Public BaseRender package facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "initialize_runtime": ("dnadesign.baserender.src.runtime", "initialize_runtime"),
    "app": ("dnadesign.baserender.src.cli", "app"),
    "adapt_record": ("dnadesign.baserender.src.public", "adapt_record"),
    "adapt_records": ("dnadesign.baserender.src.public", "adapt_records"),
    "BaseRenderJobV3": ("dnadesign.baserender.src.config", "BaseRenderJobV3"),
    "RenderJobV3": ("dnadesign.baserender.src.config", "RenderJobV3"),
    "RenderContractDescriptor": ("dnadesign.baserender.src.config", "RenderContractDescriptor"),
    "Style": ("dnadesign.baserender.src.config", "Style"),
    "resolve_style": ("dnadesign.baserender.src.config", "resolve_style"),
    "resolve_preset_path": ("dnadesign.baserender.src.config", "resolve_preset_path"),
    "list_style_presets": ("dnadesign.baserender.src.config", "list_style_presets"),
    "run_sequence_rows_job": ("dnadesign.baserender.src.public", "run_sequence_rows_job"),
    "run_render_job": ("dnadesign.baserender.src.public", "run_render_job"),
    "run_cruncher_showcase_job": ("dnadesign.baserender.src.public", "run_cruncher_showcase_job"),
    "validate_sequence_rows_job": ("dnadesign.baserender.src.public", "validate_sequence_rows_job"),
    "validate_render_job": ("dnadesign.baserender.src.public", "validate_render_job"),
    "validate_cruncher_showcase_job": ("dnadesign.baserender.src.public", "validate_cruncher_showcase_job"),
    "run_job": ("dnadesign.baserender.src.public", "run_job"),
    "validate_job": ("dnadesign.baserender.src.public", "validate_job"),
    "list_adapters": ("dnadesign.baserender.src.public", "list_adapters"),
    "list_renderers": ("dnadesign.baserender.src.public", "list_renderers"),
    "list_render_contracts": ("dnadesign.baserender.src.public", "list_render_contracts"),
    "get_adapter_descriptor": ("dnadesign.baserender.src.public", "get_adapter_descriptor"),
    "get_renderer_descriptor": ("dnadesign.baserender.src.public", "get_renderer_descriptor"),
    "get_render_contract_descriptor": ("dnadesign.baserender.src.public", "get_render_contract_descriptor"),
    "render": ("dnadesign.baserender.src.public", "render"),
    "cruncher_showcase_style_overrides": ("dnadesign.baserender.src.public", "cruncher_showcase_style_overrides"),
    "Record": ("dnadesign.baserender.src.core", "Record"),
    "Feature": ("dnadesign.baserender.src.core", "Feature"),
    "Effect": ("dnadesign.baserender.src.core", "Effect"),
    "Display": ("dnadesign.baserender.src.core", "Display"),
    "Span": ("dnadesign.baserender.src.core", "Span"),
    "Palette": ("dnadesign.baserender.src.render.palette", "Palette"),
    "SchemaError": ("dnadesign.baserender.src.core", "SchemaError"),
    "ContractError": ("dnadesign.baserender.src.core", "ContractError"),
    "LayoutError": ("dnadesign.baserender.src.core", "LayoutError"),
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
    "DEFAULT_SEQUENCE_PANEL_PROFILE": ("dnadesign.baserender.src.public", "DEFAULT_SEQUENCE_PANEL_PROFILE"),
    "SequencePanelConfig": ("dnadesign.baserender.src.public", "SequencePanelConfig"),
    "SequencePanelDiagnostics": ("dnadesign.baserender.src.public", "SequencePanelDiagnostics"),
    "SequencePanelImage": ("dnadesign.baserender.src.public", "SequencePanelImage"),
    "sequence_panel_config_for_adapter": ("dnadesign.baserender.src.public", "sequence_panel_config_for_adapter"),
    "render_sequence_panel_image": ("dnadesign.baserender.src.public", "render_sequence_panel_image"),
    "DENSEGEN_TFBS_REQUIRED_KEYS": ("dnadesign.baserender.src.contracts", "DENSEGEN_TFBS_REQUIRED_KEYS"),
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
