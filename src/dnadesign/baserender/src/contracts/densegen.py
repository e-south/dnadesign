"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/contracts/densegen.py

DenseGen adapter contract constants shared across runtime, tests, and docs checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DENSEGEN_TFBS_REQUIRED_KEYS: tuple[str, ...] = ("regulator", "sequence", "orientation", "offset")
