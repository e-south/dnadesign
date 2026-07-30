"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/__init__.py

Public BaseRender API implementation package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .api import (
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
    DEFAULT_SEQUENCE_PANEL_PROFILE,
    SequencePanelConfig,
    SequencePanelDiagnostics,
    SequencePanelImage,
    adapt_record,
    adapt_records,
    cruncher_showcase_style_overrides,
    get_adapter_descriptor,
    get_render_contract_descriptor,
    get_renderer_descriptor,
    list_adapters,
    list_render_contracts,
    list_renderers,
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
    validate_job,
    validate_render_job,
)

__all__ = [
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID",
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION",
    "DEFAULT_SEQUENCE_PANEL_PROFILE",
    "SequencePanelConfig",
    "SequencePanelDiagnostics",
    "SequencePanelImage",
    "adapt_record",
    "adapt_records",
    "cruncher_showcase_style_overrides",
    "get_adapter_descriptor",
    "get_render_contract_descriptor",
    "get_renderer_descriptor",
    "list_adapters",
    "list_render_contracts",
    "list_renderers",
    "load_record_from_parquet",
    "load_records_from_parquet",
    "render",
    "render_parquet_record_figure",
    "render_record_figure",
    "render_record_grid_figure",
    "render_sequence_panel_image",
    "run_job",
    "run_render_job",
    "sequence_panel_config_for_adapter",
    "validate_job",
    "validate_render_job",
]
