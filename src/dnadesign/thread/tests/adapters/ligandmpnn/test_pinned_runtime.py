"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_pinned_runtime.py

Tests attested LigandMPNN entrypoint execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from dnadesign.thread.adapters.ligandmpnn.pinned_runtime import execute_pinned_entrypoint


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "LigandMPNN"
    root.mkdir()
    (root / "data_utils.py").write_text("VALUE = 'attested'\n", encoding="utf-8")
    (root / "run.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from data_utils import VALUE\n"
        "Path(sys.argv[1]).write_text(VALUE, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (root / "score.py").write_text("from data_utils import VALUE\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return root, commit


def test_pinned_runtime_ignores_timestamp_valid_poisoned_parser_bytecode(tmp_path: Path) -> None:
    checkout, commit = _checkout(tmp_path)
    parser_path = checkout / "data_utils.py"
    malicious_source = tmp_path / "data_utils.py"
    malicious_source.write_text("VALUE = 'poisoned'\n", encoding="utf-8")
    assert malicious_source.stat().st_size == parser_path.stat().st_size
    parser_mtime = parser_path.stat().st_mtime
    os.utime(malicious_source, (parser_mtime, parser_mtime))
    cache_path = Path(importlib.util.cache_from_source(str(parser_path)))
    cache_path.parent.mkdir()
    py_compile.compile(
        str(malicious_source),
        cfile=str(cache_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )

    poisoned_output = tmp_path / "poisoned.txt"
    subprocess.run([sys.executable, str(checkout / "run.py"), str(poisoned_output)], check=True)
    assert poisoned_output.read_text(encoding="utf-8") == "poisoned"

    attested_output = tmp_path / "attested.txt"
    execute_pinned_entrypoint(
        checkout_root=checkout,
        upstream_commit=commit,
        entrypoint="run.py",
        arguments=(str(attested_output),),
    )

    assert attested_output.read_text(encoding="utf-8") == "attested"


def test_pinned_runtime_rejects_dirty_parser_bytes(tmp_path: Path) -> None:
    checkout, commit = _checkout(tmp_path)
    (checkout / "data_utils.py").write_text("VALUE = 'modified'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="data_utils.py must match the pinned commit"):
        execute_pinned_entrypoint(
            checkout_root=checkout,
            upstream_commit=commit,
            entrypoint="run.py",
            arguments=(str(tmp_path / "output.txt"),),
        )
