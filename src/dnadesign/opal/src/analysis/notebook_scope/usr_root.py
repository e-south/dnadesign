"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_scope/usr_root.py

Validates the USR coordinate bound to an OPAL review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from dnadesign.usr import require_explicit_usr_root


def resolve_notebook_usr_root(serialized_root: str | Path | None) -> Path | None:
    """Verify that a marimo invocation agrees with the notebook's USR coordinate."""

    notebook_root = None if serialized_root is None else require_explicit_usr_root(serialized_root)
    invocation_text = os.environ.get("OPAL_NOTEBOOK_USR_ROOT")
    if invocation_text is None:
        return notebook_root
    invocation_root = require_explicit_usr_root(invocation_text)
    if notebook_root is None:
        raise RuntimeError("This notebook does not bind an explicit USR root; regenerate it with opal --usr-root.")
    if invocation_root != notebook_root:
        raise RuntimeError("The marimo invocation USR root does not match the generated notebook.")
    return notebook_root


__all__ = ["resolve_notebook_usr_root"]
