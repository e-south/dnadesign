"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/src/runtime/__init__.py

SeqLedger runtime helper package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .duckdb import connect_duckdb_utc

__all__ = ["connect_duckdb_utc"]
