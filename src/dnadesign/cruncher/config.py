"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/config.py

Typed YAML → Pydantic models for cruncher.
Now includes optional `sample` and `analysis` dictionaries so that
`cfg.sample` / `cfg.analysis` are still available even if we don't model
them in detail yet.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


# motif block
class MotifPlotCfg(BaseModel):
    logo: bool = True
    bits_mode: Literal["probability", "information"] = "information"
    dpi: int = 200


class MotifCfg(BaseModel):
    root: Path
    formats: Dict[str, str] = Field(
        default_factory=lambda: {".meme": "MEME", ".pfm": "JASPAR"}
    )
    plot: MotifPlotCfg = MotifPlotCfg()

    @validator("root", pre=True)
    def _expand_root(cls, v):
        return Path(v).expanduser().resolve()


# ────────────────────── cruncher root block ───────────────────────
class CruncherCfg(BaseModel):
    mode: Literal["parse", "sample", "analyse"] = "parse"
    motif: MotifCfg
    sample: Optional[dict] = None     # parsed but not validated yet
    analysis: Optional[dict] = None


# ────────────────────────── loader helper ─────────────────────────
def load_yaml(path: Path) -> CruncherCfg:
    data = yaml.safe_load(path.read_text())
    if "cruncher" not in data:
        raise KeyError(f"'cruncher:' top-level key missing in {path}")
    return CruncherCfg.parse_obj(data["cruncher"])
