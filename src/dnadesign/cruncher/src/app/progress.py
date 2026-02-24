"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/progress.py

Provide progress adapters for optimizer iterations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any, TypeVar

from tqdm import tqdm

from dnadesign.cruncher.core.optimizers.progress import ProgressAdapter, passthrough_progress

T = TypeVar("T")
_NONINTERACTIVE_ENV_VAR = "CRUNCHER_NONINTERACTIVE"


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def progress_output_enabled() -> bool:
    if _env_truthy(_NONINTERACTIVE_ENV_VAR) or _env_truthy("CI"):
        return False
    try:
        return sys.stderr.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def progress_adapter(enabled: bool) -> ProgressAdapter:
    if not enabled:
        return passthrough_progress

    def _progress(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
        tqdm_kwargs = dict(kwargs)
        disable = bool(tqdm_kwargs.pop("disable", False))
        tqdm_kwargs["disable"] = disable or not progress_output_enabled()
        tqdm_kwargs.setdefault("dynamic_ncols", True)
        return tqdm(iterable, **tqdm_kwargs)

    return _progress
