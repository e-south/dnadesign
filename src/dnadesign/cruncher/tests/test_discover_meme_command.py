"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_discover_meme_command.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.cli.commands.discover import build_meme_command


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
