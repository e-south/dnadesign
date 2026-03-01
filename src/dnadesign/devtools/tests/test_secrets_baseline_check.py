"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_secrets_baseline_check.py

Tests for detect-secrets baseline hygiene checks used by CI and local workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.devtools.secrets_baseline_check import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_main_passes_when_all_baseline_result_paths_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    baseline_path = repo_root / ".secrets.baseline"
    flagged = repo_root / "src" / "example.py"
    flagged.parent.mkdir(parents=True, exist_ok=True)
    flagged.write_text("print('ok')\n", encoding="utf-8")
    _write_json(
        baseline_path,
        {
            "version": "1.5.0",
            "results": {
                "src/example.py": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "src/example.py",
                        "hashed_secret": "abc",
                        "line_number": 1,
                        "is_verified": False,
                    }
                ]
            },
        },
    )

    rc = main(["--repo-root", str(repo_root), "--baseline", str(baseline_path)])
    assert rc == 0


def test_main_fails_when_baseline_contains_missing_file_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    baseline_path = repo_root / ".secrets.baseline"
    _write_json(
        baseline_path,
        {
            "version": "1.5.0",
            "results": {
                "src/missing.py": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "src/missing.py",
                        "hashed_secret": "abc",
                        "line_number": 1,
                        "is_verified": False,
                    }
                ]
            },
        },
    )

    rc = main(["--repo-root", str(repo_root), "--baseline", str(baseline_path)])
    assert rc == 1
