"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/workspace_storage/test_cli.py

Tests the machine-readable workspace-storage validation command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dnadesign.contracts.workspace_storage import MANIFEST_NAME


def test_cli_emits_machine_readable_verified_summary(tmp_path: Path) -> None:
    workspace_root = tmp_path / "pilot"
    workspace_root.mkdir()
    payload = b"payload\n"
    (workspace_root / "payload.txt").write_bytes(payload)
    manifest = {
        "schema": "dnadesign.workspace-storage/v1",
        "workspace_id": "pilot",
        "owner_repository": "dnadesign",
        "owner_tool": "latentdna",
        "workspace_schema": "latentdna.workspace",
        "workspace_schema_version": "1",
        "producer_revision": "test-revision-1",
        "storage_class": "reproducible",
        "retention_policy": "rebuildable",
        "demo": False,
        "inputs": [
            {
                "path": "payload.txt",
                "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            }
        ],
        "artifacts": [],
    }
    (workspace_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.workspace_storage",
            "validate",
            str(workspace_root),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "artifact_count": 0,
        "input_count": 1,
        "owner_repository": "dnadesign",
        "owner_tool": "latentdna",
        "schema": "dnadesign.workspace-storage/v1",
        "status": "verified",
        "storage_class": "reproducible",
        "workspace_id": "pilot",
    }
