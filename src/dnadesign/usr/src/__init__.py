"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/__init__.py

Internal package root for USR implementation.

Internal modules:

- dataset: Parquet-backed dataset with schema enforcement and namespaced attach
- cli:     Thin CLI wrapper around Dataset
- errors:  Small, typed exception hierarchy (ValidationError, NamespaceError, ...)
- io:      Atomic Parquet writes, snapshots, append-only event log
- normalize: Case-preserving sequence normalization & deterministic ID
- schema:  Arrow schema for required columns
- version: Semantic version for USR

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""
