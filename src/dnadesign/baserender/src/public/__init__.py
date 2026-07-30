"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/__init__.py

Public BaseRender API implementation package, loaded on first use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_CATALOG_EXPORTS = {
    "get_adapter_descriptor",
    "get_render_contract_descriptor",
    "get_renderer_descriptor",
    "list_adapters",
    "list_render_contracts",
    "list_renderers",
}
_API_EXPORTS = {
    "adapt_record",
    "adapt_records",
    "cruncher_showcase_style_overrides",
    "load_record_from_parquet",
    "load_records_from_parquet",
    "render",
    "render_parquet_record_figure",
    "render_record_figure",
    "render_record_grid_figure",
    "render_sequence_panel_image",
    "run_job",
    "run_render_job",
    "validate_job",
    "validate_render_job",
}
_SEQUENCE_PANEL_EXPORTS = {
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID",
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION",
    "DEFAULT_SEQUENCE_PANEL_PROFILE",
    "SequencePanelConfig",
    "SequencePanelDiagnostics",
    "SequencePanelImage",
    "sequence_panel_config_for_adapter",
}

__all__ = sorted(_API_EXPORTS | _CATALOG_EXPORTS | _SEQUENCE_PANEL_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _CATALOG_EXPORTS:
        module_name = ".catalog"
    elif name in _SEQUENCE_PANEL_EXPORTS:
        module_name = ".sequence_panel"
    elif name in _API_EXPORTS:
        module_name = ".api"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
