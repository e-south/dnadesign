"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/runtime/batch_policy.py

Resolves runtime micro-batch policy from model config and environment contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def resolve_micro_batch_size(*, model_batch_size: Optional[int]) -> int:
    raw_batch = model_batch_size if model_batch_size is not None else int(os.environ.get("DNADESIGN_INFER_BATCH", "0"))
    return int(raw_batch) if raw_batch else 0


def resolve_default_extract_batch_size() -> int:
    return int(os.environ.get("DNADESIGN_INFER_DEFAULT_BS", "64"))


def resolve_extract_batch_policy(*, model_batch_size: Optional[int]) -> Tuple[int, int]:
    return resolve_micro_batch_size(model_batch_size=model_batch_size), resolve_default_extract_batch_size()
