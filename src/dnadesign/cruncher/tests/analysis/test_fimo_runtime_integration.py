"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_fimo_runtime_integration.py

Runs a real MEME Suite FIMO execution against generated motif and FASTA inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnadesign.cruncher.analysis.fimo_concordance import _run_fimo, _write_minimal_meme_motif
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.integrations.meme_suite import resolve_executable


def _resolve_test_fimo_executable() -> Path | None:
    resolved = resolve_executable("fimo", tool_path=None)
    if resolved is not None:
        return resolved
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        candidate = parent / ".pixi" / "envs" / "default" / "bin" / "fimo"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


_FIMO_EXECUTABLE = _resolve_test_fimo_executable()

pytestmark = [
    pytest.mark.fimo,
    pytest.mark.skipif(
        _FIMO_EXECUTABLE is None,
        reason="fimo executable not available (set MEME_BIN, PATH, or install repo pixi env).",
    ),
]


def test_run_fimo_executes_with_real_binary(tmp_path) -> None:
    motif_path = tmp_path / "toy.meme"
    fasta_path = tmp_path / "toy.fa"

    pwm = PWM(name="toy", matrix=[[0.97, 0.01, 0.01, 0.01]])
    _write_minimal_meme_motif(pwm, motif_path)
    fasta_path.write_text(">seq_000000\nAAAA\n", encoding="utf-8")

    rows = _run_fimo(
        motif_path=motif_path,
        fasta_path=fasta_path,
        bidirectional=True,
        threshold=1.0,
        tool_path=_FIMO_EXECUTABLE.parent if _FIMO_EXECUTABLE is not None else None,
    )

    assert rows
    first = rows[0]
    assert {"sequence_name", "start", "stop", "strand", "score", "p_value"} <= set(first)
