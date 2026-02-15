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

from dnadesign.cruncher.config.schema_v3 import CruncherConfig, CruncherRoot


def load_config(path: Path) -> CruncherConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "cruncher" not in raw:
        raise ValueError("Config schema v3 required (missing root key: cruncher)")
    payload = raw.get("cruncher", {})
    if not isinstance(payload, dict):
        raise ValueError("Config schema v3 required (cruncher must be a mapping)")
    if payload.get("schema_version") != 3:
        raise ValueError("Config schema v3 required (schema_version: 3)")
    cfg = CruncherRoot.model_validate(raw).cruncher
    # catalog_root remains relative; app orchestration resolves against the cruncher repo root
    return cfg
