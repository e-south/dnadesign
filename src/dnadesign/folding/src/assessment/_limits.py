"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/_limits.py

Resource limits for one isolated structure-assessment publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

ARTIFACT_FILE_SIZE_LIMIT_BYTES = 1_048_576
ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES = 16_777_216
ARTIFACT_ENTRY_COUNT_LIMIT = 256

__all__ = [
    "ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES",
    "ARTIFACT_ENTRY_COUNT_LIMIT",
    "ARTIFACT_FILE_SIZE_LIMIT_BYTES",
]
