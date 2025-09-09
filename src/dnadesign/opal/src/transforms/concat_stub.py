"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms/concat_stub.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..utils import OpalError


class ConcatStubTransform:
    """
    Placeholder for future multi-column concatenation.
    v1: not implemented; provided so registry is extendible.
    """

    def __init__(self, **kwargs):
        raise OpalError("Transform 'concat' is not implemented in v1.")
