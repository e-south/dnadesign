"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/test_package_layout.py

Package-layout regression tests for generic thread workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

_THREAD_ROOT_FILES = {"__init__.py"}
_THREAD_DIRECTORIES = {"adapters", "assets", "candidates", "docs", "tests"}
_PROTEINMPNN_FILES = {
    "__init__.py",
    "execution.py",
    "execution_preflight.py",
    "hashing.py",
    "manifest.py",
    "models.py",
    "positions.py",
    "samples.py",
    "sidecars.py",
    "structure.py",
    "validation.py",
}


def test_thread_root_is_small_public_tool_surface() -> None:
    root = _repo_root() / "src/dnadesign/thread"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_THREAD_ROOT_FILES)
    assert sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "__pycache__") == sorted(
        _THREAD_DIRECTORIES
    )


def test_proteinmpnn_adapter_owns_generic_request_mechanics() -> None:
    root = _repo_root() / "src/dnadesign/thread/adapters/proteinmpnn"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_PROTEINMPNN_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "ProteinMPNN" in (root / "validation.py").read_text(encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
