from __future__ import annotations

import os
from pathlib import Path


def ensure_mpl_cache_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    target = cache_root / "densegen" / "matplotlib"
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(target)
    except Exception:
        tmp = Path(os.getenv("TMPDIR") or "/tmp") / "densegen-matplotlib"
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)
