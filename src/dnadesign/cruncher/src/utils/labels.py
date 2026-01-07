"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/labels.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Iterable, Sequence

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(label: str, *, max_len: int = 60) -> str:
    cleaned = _SAFE_RE.sub("_", label).strip("_")
    if max_len and len(cleaned) > max_len:
        return cleaned[:max_len].rstrip("_")
    return cleaned


def format_regulator_label(tfs: Sequence[str]) -> str:
    return "-".join(tfs)


def format_regulator_slug(tfs: Sequence[str], *, max_len: int = 60) -> str:
    return _slugify(format_regulator_label(tfs), max_len=max_len)


def build_run_name(stage: str, tfs: Sequence[str], *, set_index: int | None = None) -> str:
    date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = format_regulator_slug(tfs)
    seed = f"{stage}|{date_stamp}|{','.join(tfs)}|{set_index or ''}"
    short_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6]
    if set_index is not None:
        return f"{stage}_set{set_index}_{label}_{date_stamp}_{short_hash}"
    return f"{stage}_{label}_{date_stamp}_{short_hash}"


def regulator_sets(regulator_sets: Iterable[Iterable[str]]) -> list[list[str]]:
    return [list(group) for group in regulator_sets]
