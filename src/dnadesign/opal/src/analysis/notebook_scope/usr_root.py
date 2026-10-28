"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_scope/usr_root.py

Validates the SeqLedger coordinate bound to an OPAL review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_notebook_usr_root(serialized_root: str | Path | None) -> Path | None:
    """Verify that a marimo invocation agrees with the notebook's SeqLedger coordinate."""

    invocation_text = os.environ.get("OPAL_NOTEBOOK_SEQLEDGER_ROOT")
    if serialized_root is None and invocation_text is None:
        return None

    from dnadesign.seqledger import require_explicit_seqledger_root

    notebook_root = None if serialized_root is None else require_explicit_seqledger_root(serialized_root)
    if invocation_text is None:
        return notebook_root
    invocation_root = require_explicit_seqledger_root(invocation_text)
    if notebook_root is None:
        raise RuntimeError(
            "This notebook does not bind an explicit SeqLedger root; regenerate it with opal --seqledger-root."
        )
    if invocation_root != notebook_root:
        raise RuntimeError("The marimo invocation SeqLedger root does not match the generated notebook.")
    return notebook_root


__all__ = ["resolve_notebook_usr_root"]
