"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/__init__.py

Baserender vNext implementation package facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dnadesign.baserender.src.config import BaseRenderJobV3, RenderContractDescriptor, RenderJobV3  # noqa: F401
    from dnadesign.baserender.src.contracts import DENSEGEN_TFBS_REQUIRED_KEYS  # noqa: F401
    from dnadesign.baserender.src.core import (  # noqa: F401
        ContractError,
        Display,
        Effect,
        Feature,
        LayoutError,
        Record,
        SchemaError,
        Span,
    )
    from dnadesign.baserender.src.public import (  # noqa: F401
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
        run_cruncher_showcase_job,
        run_job,
        run_render_job,
        run_sequence_rows_job,
        validate_cruncher_showcase_job,
        validate_job,
        validate_render_job,
        validate_sequence_rows_job,
    )
    from dnadesign.baserender.src.runtime import initialize_runtime  # noqa: F401

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "initialize_runtime": ("dnadesign.baserender.src.runtime", "initialize_runtime"),
    "RenderJobV4": ("dnadesign.baserender.src.config", "RenderJobV4"),
    "RenderContractDescriptor": ("dnadesign.baserender.src.config", "RenderContractDescriptor"),
    "run_render_job": ("dnadesign.baserender.src.public", "run_render_job"),
    "validate_render_job": ("dnadesign.baserender.src.public", "validate_render_job"),
    "run_job": ("dnadesign.baserender.src.public", "run_job"),
    "validate_job": ("dnadesign.baserender.src.public", "validate_job"),
    "list_adapters": ("dnadesign.baserender.src.public.catalog", "list_adapters"),
    "list_render_contracts": ("dnadesign.baserender.src.public.catalog", "list_render_contracts"),
    "list_renderers": ("dnadesign.baserender.src.public.catalog", "list_renderers"),
    "get_adapter_descriptor": ("dnadesign.baserender.src.public.catalog", "get_adapter_descriptor"),
    "get_render_contract_descriptor": ("dnadesign.baserender.src.public.catalog", "get_render_contract_descriptor"),
    "get_renderer_descriptor": ("dnadesign.baserender.src.public.catalog", "get_renderer_descriptor"),
    "render": ("dnadesign.baserender.src.public", "render"),
    "cruncher_showcase_style_overrides": ("dnadesign.baserender.src.public", "cruncher_showcase_style_overrides"),
    "Record": ("dnadesign.baserender.src.core", "Record"),
    "Feature": ("dnadesign.baserender.src.core", "Feature"),
    "Effect": ("dnadesign.baserender.src.core", "Effect"),
    "Display": ("dnadesign.baserender.src.core", "Display"),
    "Span": ("dnadesign.baserender.src.core", "Span"),
    "SchemaError": ("dnadesign.baserender.src.core", "SchemaError"),
    "ContractError": ("dnadesign.baserender.src.core", "ContractError"),
    "LayoutError": ("dnadesign.baserender.src.core", "LayoutError"),
    "load_records_from_parquet": ("dnadesign.baserender.src.public", "load_records_from_parquet"),
    "load_record_from_parquet": ("dnadesign.baserender.src.public", "load_record_from_parquet"),
    "render_record_figure": ("dnadesign.baserender.src.public", "render_record_figure"),
    "render_record_grid_figure": ("dnadesign.baserender.src.public", "render_record_grid_figure"),
    "render_parquet_record_figure": ("dnadesign.baserender.src.public", "render_parquet_record_figure"),
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
