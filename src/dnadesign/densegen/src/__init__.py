"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/__init__.py

DenseGen — Dense Array Generator

Public modules:
- cli: Typer/Rich CLI entrypoint
- config: strict config schema + loaders
- core: domain logic (pipeline, sampler, metadata)
- adapters: optimizer, sources, outputs
- viz: plotting and plot registry
- utils: shared utilities

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

__all__ = [
    "adapters",
    "cli",
    "config",
    "core",
    "utils",
    "viz",
]
