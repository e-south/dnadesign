"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_campaign_lock_stale.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.locks import CampaignLock


def test_campaign_lock_detects_stale_pid(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    lock_path = workdir / ".opal.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "ts": "2026-01-01T00:00:00Z"}), encoding="utf-8")

    with pytest.raises(OpalError, match="stale lock"):
        with CampaignLock(workdir):
            raise AssertionError("lock should not be acquired")
