"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/util/meta.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
import json


def compact_meta(
    ver: str,
    algo: str,
    x_col: str | None,
    n: int,
    params: dict,
    source: dict,
    sweep: dict | None = None,
    umap: dict | None = None,
    sig_hash: str | None = None,
) -> str:
    obj = {
        "ver": ver,
        "utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algo": algo,
        "x_col": x_col,
        "n": n,
        "p": params or {},
        "src": source or {},
        "sweep": sweep,
        "umap": umap or {"attached": False},
    }
    if sig_hash:
        obj["sig"] = sig_hash
    return json.dumps(obj, separators=(",", ":"))
