"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/contracts/request/limits.py

Shared resource limits for every junction request boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUEST_IDENTIFIER_BYTES = 128
MAX_REQUEST_INTEGER = (1 << 64) - 1
MAX_REQUEST_PLAIN_TEXT_BYTES = 128

__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_IDENTIFIER_BYTES",
    "MAX_REQUEST_INTEGER",
    "MAX_REQUEST_PLAIN_TEXT_BYTES",
]
