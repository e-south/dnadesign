"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/__init__.py

Baserender package root exports for vNext runtime located under internal src/.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .src.api import (
    adapt_record,
    adapt_records,
    cruncher_showcase_style_overrides,
    get_adapter_descriptor,
    get_renderer_descriptor,
    list_adapters,
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
from .src.config import RenderJobV3
from .src.contracts import DENSEGEN_TFBS_REQUIRED_KEYS
from .src.core import ContractError, Display, Effect, Feature, LayoutError, Record, SchemaError, Span
from .src.render.palette import Palette
from .src.runtime import initialize_runtime

__all__ = [
    "initialize_runtime",
    "adapt_record",
    "adapt_records",
    "RenderJobV3",
    "run_sequence_rows_job",
    "run_render_job",
    "run_cruncher_showcase_job",
    "validate_sequence_rows_job",
    "validate_render_job",
    "validate_cruncher_showcase_job",
    "run_job",
    "validate_job",
    "list_adapters",
    "list_renderers",
    "get_adapter_descriptor",
    "get_renderer_descriptor",
    "render",
    "cruncher_showcase_style_overrides",
    "Record",
    "Feature",
    "Effect",
    "Display",
    "Span",
    "Palette",
    "SchemaError",
    "ContractError",
    "LayoutError",
    "load_records_from_parquet",
    "load_record_from_parquet",
    "render_record_figure",
    "render_record_grid_figure",
    "render_parquet_record_figure",
    "DENSEGEN_TFBS_REQUIRED_KEYS",
]
