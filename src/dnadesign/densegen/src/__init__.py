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

from .utils.logging_utils import install_native_stderr_filters as _install_native_stderr_filters

_install_native_stderr_filters(suppress_solver_messages=False)

__all__ = [
    "adapters",
    "cli",
    "config",
    "core",
    "utils",
    "viz",
]
