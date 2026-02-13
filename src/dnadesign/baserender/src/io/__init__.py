"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/io/__init__.py

Data-source IO exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .parquet_source import iter_parquet_rows

__all__ = ["iter_parquet_rows"]
