"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_meme_env.py

Tests for MEME/FIMO environment resolution helpers used by the CI external integration lane.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.meme_env import main, resolve_meme_bin


def _create_fimo_binary(repo_root: Path) -> Path:
    meme_bin = repo_root / ".pixi" / "envs" / "default" / "bin"
    meme_bin.mkdir(parents=True, exist_ok=True)
    fimo_path = meme_bin / "fimo"
    fimo_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    current_mode = fimo_path.stat().st_mode
    fimo_path.chmod(current_mode | 0o111)
    return meme_bin


def test_resolve_meme_bin_returns_expected_path(tmp_path: Path) -> None:
    meme_bin = _create_fimo_binary(tmp_path)

    resolved = resolve_meme_bin(repo_root=tmp_path)

    assert resolved == meme_bin


def test_main_writes_github_env_and_path(tmp_path: Path) -> None:
    meme_bin = _create_fimo_binary(tmp_path)
    github_env = tmp_path / "github_env.txt"
    github_path = tmp_path / "github_path.txt"

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--github-env",
            str(github_env),
            "--github-path",
            str(github_path),
        ]
    )

    assert rc == 0
    assert github_env.read_text(encoding="utf-8") == f"MEME_BIN={meme_bin}\n"
    assert github_path.read_text(encoding="utf-8") == f"{meme_bin}\n"


def test_main_fails_when_fimo_binary_is_missing(tmp_path: Path) -> None:
    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_prints_shell_exports(tmp_path: Path, capsys) -> None:
    meme_bin = _create_fimo_binary(tmp_path)

    rc = main(["--repo-root", str(tmp_path), "--print-shell-export"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == (f'export MEME_BIN={meme_bin}\nexport PATH="$MEME_BIN:$PATH"\n')


def test_main_shell_exports_quote_paths_with_spaces(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path / "repo with space"
    meme_bin = _create_fimo_binary(repo_root)

    rc = main(["--repo-root", str(repo_root), "--print-shell-export"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == (f"export MEME_BIN='{meme_bin}'\nexport PATH=\"$MEME_BIN:$PATH\"\n")
