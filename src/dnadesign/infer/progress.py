"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/progress.py

Creates progress handles for infer runtime jobs with optional environment-based suppression.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from ._logging import get_logger

_LOG = get_logger(__name__)

ProgressFactory = Optional[Callable[[str, int], Any]]


def resolve_tqdm_factory():
    show = os.environ.get("DNADESIGN_PROGRESS", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    if not show:
        return _NoTQDM, False
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm, True
    except Exception:
        _LOG.info("tqdm not available; continuing without a progress bar.")
        return _NoTQDM, False


def create_progress_handle(*, progress_factory: ProgressFactory, label: str, total: int, unit: str):
    if progress_factory:
        return progress_factory(label, total)
    tqdm, _ = resolve_tqdm_factory()
    return tqdm(total=total, unit=unit, desc=label)


class _NoTQDM:
    def __init__(self, total=None, **_kwargs):
        self.total = total

    def update(self, _n):
        return None

    def close(self):
        return None
