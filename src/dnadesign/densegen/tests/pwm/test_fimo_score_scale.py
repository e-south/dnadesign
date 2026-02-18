"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_fimo_score_scale.py

Integration tests for FIMO score scale alignment.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dnadesign.densegen.src.adapters.sources.pwm_fimo import (
    aggregate_best_hits,
    build_candidate_records,
    run_fimo,
    write_candidates_fasta,
    write_minimal_meme_motif,
)
from dnadesign.densegen.src.core.stage_a.stage_a_sampling_utils import (
    _pwm_theoretical_max_score,
    build_log_odds,
)
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable

pytestmark = pytest.mark.fimo


def _consensus(matrix: list[dict[str, float]]) -> str:
    return "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)


def test_fimo_score_scale_matches_log2(tmp_path: Path) -> None:
    if resolve_executable("fimo") is None:
        pytest.skip("fimo not available")
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.05, "C": 0.05, "G": 0.85, "T": 0.05},
        {"A": 0.1, "C": 0.6, "G": 0.1, "T": 0.2},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    motif = PWMMotif(motif_id="scale_test", matrix=matrix, background=background)
    meme_path = tmp_path / "motif.meme"
    write_minimal_meme_motif(motif, meme_path)
    seq = _consensus(matrix)
    records = build_candidate_records(motif.motif_id, [seq], start_index=0)
    fasta_path = tmp_path / "candidates.fasta"
    write_candidates_fasta(records, fasta_path)
    rows, _ = run_fimo(
        meme_motif_path=meme_path,
        fasta_path=fasta_path,
        bgfile="motif-file",
        norc=True,
        thresh=1.0,
        include_matched_sequence=False,
        return_tsv=False,
    )
    best_hits = aggregate_best_hits(rows)
    assert len(best_hits) == 1
    hit = best_hits[records[0][0]]
    assert math.isfinite(hit.score)
    log_odds_log2 = build_log_odds(matrix, background, smoothing_alpha=0.0, log_base=2.0)
    theoretical_max_log2 = _pwm_theoretical_max_score(log_odds_log2)
    log_odds_ln = build_log_odds(matrix, background, smoothing_alpha=0.0, log_base=math.e)
    theoretical_max_ln = _pwm_theoretical_max_score(log_odds_ln)
    assert abs(hit.score - theoretical_max_log2) <= 0.05
    assert abs(hit.score - theoretical_max_log2) < abs(hit.score - theoretical_max_ln)
