"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_public_imports.py

Public contract facade import-boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def _fresh_process_lines(code: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def test_contract_root_defers_family_implementations_until_symbol_access() -> None:
    lines = _fresh_process_lines(
        "\n".join(
            [
                "import sys",
                "import dnadesign.contracts as contracts",
                "print('dnadesign.contracts.visual.sequence_evidence_map_v1' in sys.modules)",
                "print('dnadesign.contracts.sequence.linear_ssdna_composition_v1' in sys.modules)",
                "contracts.SequenceEvidenceMapV1",
                "print('dnadesign.contracts.visual.sequence_evidence_map_v1' in sys.modules)",
                "print('dnadesign.contracts.sequence.linear_ssdna_composition_v1' in sys.modules)",
            ]
        )
    )

    assert lines == ["False", "False", "True", "False"]


def test_contract_family_defers_sibling_implementations() -> None:
    lines = _fresh_process_lines(
        "\n".join(
            [
                "import sys",
                "import dnadesign.contracts.visual as visual",
                "visual.SequenceEvidenceMapV1",
                "print('dnadesign.contracts.visual.sequence_evidence_map_v1' in sys.modules)",
                "print('dnadesign.contracts.visual.snapback_visual_v1' in sys.modules)",
            ]
        )
    )

    assert lines == ["True", "False"]


def test_removed_sequence_contract_spelling_alias_is_not_exported() -> None:
    lines = _fresh_process_lines(
        "import dnadesign.contracts.sequence as sequence\n"
        "print('LinearSsDnaCompositionV1' in sequence.__all__)\n"
        "print(hasattr(sequence, 'LinearSsDnaCompositionV1'))\n"
    )

    assert lines == ["False", "False"]
