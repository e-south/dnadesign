"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/contracts/publication/limits.py

Shared byte ceilings for planned and verified TriJunction publications.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from types import MappingProxyType

from ..request.limits import MAX_REQUEST_BYTES

_MIB = 1024 * 1024

MANIFEST_BYTE_LIMIT = _MIB
ARTIFACT_BYTE_LIMITS = MappingProxyType(
    {
        "request": MAX_REQUEST_BYTES,
        "plan": 256 * _MIB,
        "checks": 16 * _MIB,
        "orders": 256 * _MIB,
        "review": 256 * _MIB,
    }
)

__all__ = ["ARTIFACT_BYTE_LIMITS", "MANIFEST_BYTE_LIMIT"]
