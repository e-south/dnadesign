"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/__init__.py

DenseGen — Dense Array Generator.

Module Author(s): Eric J. South
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
