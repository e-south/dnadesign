from __future__ import annotations

import logging
import os
from pathlib import Path


def ensure_mpl_cache_dir(target: Path | str) -> Path:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if os.environ.get("MPLCONFIGDIR"):
        return Path(os.environ["MPLCONFIGDIR"])
    if not target:
        raise ValueError("Matplotlib cache directory must be provided.")
    dest = Path(target).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to create matplotlib cache dir: {dest}") from exc
    os.environ["MPLCONFIGDIR"] = str(dest)
    return dest
