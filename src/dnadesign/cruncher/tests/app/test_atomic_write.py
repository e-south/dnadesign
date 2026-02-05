"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_atomic_write.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json


def test_atomic_write_json_writes_valid_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    payload = {"status": "ok", "value": 7}
    atomic_write_json(path, payload)
    assert path.exists()
    assert json.loads(path.read_text()) == payload
