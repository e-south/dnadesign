"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/runtime/__init__.py

USR runtime helper package.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from .duckdb import connect_duckdb_utc

__all__ = ["connect_duckdb_utc"]
