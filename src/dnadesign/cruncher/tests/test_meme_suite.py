from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.integrations.meme_suite import resolve_tool_path


def test_resolve_tool_path_none() -> None:
    assert resolve_tool_path(None, config_path=None) is None


def test_resolve_tool_path_absolute(tmp_path: Path) -> None:
    absolute = tmp_path / "bin"
    resolved = resolve_tool_path(absolute, config_path=tmp_path / "config.yaml")
    assert resolved == absolute


def test_resolve_tool_path_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "workspace" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    resolved = resolve_tool_path(Path("tools/meme"), config_path=config_path)
    assert resolved == (config_path.parent / "tools/meme").resolve()
