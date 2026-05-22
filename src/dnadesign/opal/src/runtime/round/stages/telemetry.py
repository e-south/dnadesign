"""
Small formatting and operator-log helpers for OPAL round stages.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ....core.utils import print_stderr


def log(enabled: bool, msg: str) -> None:
    if enabled:
        print_stderr(msg)


def format_summary_stats_for_log(summary_stats: Dict[str, Any]) -> List[str]:
    kvs: List[str] = []
    for k in sorted(summary_stats.keys()):
        v = summary_stats[k]
        if isinstance(v, (bool, np.bool_)):
            kvs.append(f"{k}={v}")
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            kvs.append(f"{k}={float(v):.6g}")
            continue
        kvs.append(f"{k}={v}")
    return kvs
