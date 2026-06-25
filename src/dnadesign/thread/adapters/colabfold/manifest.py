"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/manifest.py

ColabFold runtime manifest helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def runtime_parameters_hash(runtime_parameters: Mapping[str, Any]) -> str:
    """Return a stable hash URI for declared ColabFold runtime parameters."""

    encoded = json.dumps(dict(runtime_parameters), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
