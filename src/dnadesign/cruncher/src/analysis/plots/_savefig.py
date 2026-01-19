"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/_savefig.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def savefig(
    fig: Any,
    path: Path,
    *,
    dpi: int,
    png_compress_level: int,
    bbox_inches: str | None = "tight",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {"dpi": dpi}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    if path.suffix.lower() == ".png":
        kwargs["pil_kwargs"] = {"compress_level": png_compress_level, "optimize": True}
    fig.savefig(path, **kwargs)
