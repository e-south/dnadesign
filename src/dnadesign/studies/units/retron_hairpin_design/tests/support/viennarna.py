"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/viennarna.py

ViennaRNA test doubles for Retron hairpin materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def install_fake_viennarna_python_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = tmp_path / "python_api"
    module_dir.mkdir()
    (module_dir / "RNA.py").write_text(
        """
__version__ = "2.7.0"

class fold_compound:
    def __init__(self, sequence):
        self.sequence = sequence

    def mfe(self):
        half = len(self.sequence) // 2
        structure = ["." for _ in self.sequence]
        for index in range(min(6, half)):
            structure[index] = "("
            structure[len(self.sequence) - index - 1] = ")"
        return "".join(structure), -1.0

def plot_layout_naview(structure):
    return {"layout": "naview", "structure": structure}

def plot_structure_svg(filename, sequence, structure, layout=None):
    if "U" in sequence or "T" not in sequence:
        return 0
    if layout != {"layout": "naview", "structure": structure}:
        return 0
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 80">\\n')
        handle.write('<g id="pairs">\\n')
        handle.write('<line class="basepairs" id="1,88" x1="0" y1="20" x2="220" y2="20" />\\n')
        handle.write('</g><g id="seq">\\n')
        for index, base in enumerate(sequence):
            handle.write(f'<text class="nucleotide" x="{index * 2}" y="50">{base}</text>\\n')
        handle.write('</g>\\n</svg>\\n')
    return 1
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(module_dir.as_posix())
    sys.modules.pop("RNA", None)
