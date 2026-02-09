"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_discover_meme_command.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.cli.commands.discover import (
    build_meme_command,
    format_discovery_width_bounds,
)


def test_build_meme_command_includes_prior_and_mod() -> None:
    cmd = build_meme_command(
        exe=Path("/usr/local/bin/meme"),
        fasta_path=Path("sites.fasta"),
        run_dir=Path("run_dir"),
        minw=8,
        maxw=12,
        nmotifs=1,
        meme_mod="oops",
        meme_prior="addone",
    )
    assert "-mod" in cmd
    assert "oops" in cmd
    assert "-prior" in cmd
    assert "addone" in cmd


def test_build_meme_command_omits_width_flags_when_unset() -> None:
    cmd = build_meme_command(
        exe=Path("/usr/local/bin/meme"),
        fasta_path=Path("sites.fasta"),
        run_dir=Path("run_dir"),
        minw=None,
        maxw=None,
        nmotifs=1,
        meme_mod=None,
        meme_prior=None,
    )
    assert "-minw" not in cmd
    assert "-maxw" not in cmd


def test_format_discovery_width_bounds_uses_tool_defaults_when_unset() -> None:
    assert format_discovery_width_bounds(minw=None, maxw=None) == "minw=tool_default maxw=tool_default"


def test_format_discovery_width_bounds_renders_explicit_bounds() -> None:
    assert format_discovery_width_bounds(minw=8, maxw=14) == "minw=8 maxw=14"
