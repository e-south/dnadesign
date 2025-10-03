"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/util/slug.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def auto_run_name(prefix: str, params: dict) -> str:
    p = "_".join(f"{k}{str(v).replace('.', 'p')}" for k, v in sorted(params.items()))
    ts = datetime.utcnow().strftime("%y%m%d%H%M")
    base = slugify(f"{prefix}_{p}_{ts}")
    return base
