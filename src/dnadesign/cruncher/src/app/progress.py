"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/progress.py

Provide progress adapters for optimizer iterations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm import tqdm

from dnadesign.cruncher.core.optimizers.progress import ProgressAdapter, passthrough_progress

T = TypeVar("T")


def progress_adapter(enabled: bool) -> ProgressAdapter:
    if not enabled:
        return passthrough_progress

    def _progress(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
        return tqdm(iterable, **kwargs)

    return _progress
