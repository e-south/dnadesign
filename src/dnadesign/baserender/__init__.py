"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/__init__.py

Baserender package root exports for vNext runtime located under internal src/.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .src.api import (
    cruncher_showcase_style_overrides,
    load_record_from_parquet,
    render,
    render_parquet_record_figure,
    render_record_figure,
    render_record_grid_figure,
    run_cruncher_showcase_job,
    run_job,
    run_sequence_rows_job,
    validate_cruncher_showcase_job,
    validate_job,
    validate_sequence_rows_job,
)
from .src.core import ContractError, Display, Effect, Feature, LayoutError, Record, SchemaError, Span
from .src.runtime import initialize_runtime

__all__ = [
    "initialize_runtime",
    "run_sequence_rows_job",
    "run_cruncher_showcase_job",
    "validate_sequence_rows_job",
    "validate_cruncher_showcase_job",
    "run_job",
    "validate_job",
    "render",
    "cruncher_showcase_style_overrides",
    "Record",
    "Feature",
    "Effect",
    "Display",
    "Span",
    "SchemaError",
    "ContractError",
    "LayoutError",
    "load_record_from_parquet",
    "render_record_figure",
    "render_record_grid_figure",
    "render_parquet_record_figure",
]
