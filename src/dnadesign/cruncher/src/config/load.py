"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/load.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.config.schema_v2 import CruncherConfig, CruncherRoot


def load_config(path: Path) -> CruncherConfig:
    raw = yaml.safe_load(path.read_text())
    cfg = CruncherRoot.model_validate(raw).cruncher
    # catalog_root remains relative; app orchestration resolves against the cruncher repo root
    return cfg
