"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/progress.py

Provide optional progress wrappers for optimizer iterations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class ProgressAdapter(Protocol):
    def __call__(self, iterable: Iterable[T], **kwargs: Any) -> Iterable[T]: ...


def passthrough_progress(iterable: Iterable[T], **_kwargs: Any) -> Iterable[T]:
    return iterable
