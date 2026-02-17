from __future__ import annotations

import math
from pathlib import Path

import pytest

from dnadesign.densegen.src.adapters.sources.pwm_fimo import (
    aggregate_best_hits,
    build_candidate_records,
    parse_fimo_tsv,
    run_fimo,
    write_candidates_fasta,
    write_minimal_meme_motif,
)
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable


def test_write_minimal_meme_motif(tmp_path: Path) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
            {"A": 0.2, "C": 0.3, "G": 0.4, "T": 0.1},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    out = tmp_path / "motif.meme"
    motif_id = write_minimal_meme_motif(motif, out)
    text = out.read_text()
    assert "MEME version" in text
    assert "Background letter frequencies" in text
    assert f"MOTIF {motif_id}" in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("letter-probability matrix"))
    matrix_lines = lines[idx + 1 : idx + 1 + len(motif.matrix)]
    assert len(matrix_lines) == len(motif.matrix)
    for row in matrix_lines:
        vals = [float(x) for x in row.split()]
        assert abs(sum(vals) - 1.0) < 1e-6


def test_write_candidates_fasta(tmp_path: Path) -> None:
    records = build_candidate_records("My Motif", ["ACG", "TTT"], start_index=5)
    out = tmp_path / "candidates.fasta"
    write_candidates_fasta(records, out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith(">")
    assert lines[1] == "ACG"
    assert lines[2].startswith(">")
    assert lines[3] == "TTT"
    assert records[0][0].endswith("|cand5")
    assert records[1][0].endswith("|cand6")


def test_parse_fimo_tsv_and_best_hits() -> None:
    tsv = "\n".join(
        [
            "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence",
            "M1\t.\tcand0\t2\t4\t+\t5.2\t1e-4\t0.01\tACG",
            "M1\t.\tcand0\t1\t3\t-\t4.0\t1e-3\t0.1\tTGC",
            "M1\t.\tcand1\t1\t3\t+\t2.0\t0.5\t1.0\tAAA",
        ]
    )
    rows = parse_fimo_tsv(tsv)
    best = aggregate_best_hits(rows)
    assert best["cand0"].score == pytest.approx(5.2)
    assert best["cand0"].matched_sequence == "ACG"
    assert best["cand0"].strand == "+"
    assert best["cand1"].score == pytest.approx(2.0)


def test_parse_fimo_tsv_without_pvalue_column() -> None:
    tsv = "\n".join(
        [
            "motif_id\tsequence_name\tstart\tstop\tstrand\tscore\tmatched_sequence",
            "M1\tcand0\t2\t4\t+\t5.2\tACG",
        ]
    )
    rows = parse_fimo_tsv(tsv)
    assert rows[0]["sequence_name"] == "cand0"
    assert "p_value" not in rows[0]


def test_aggregate_best_hits_rejects_non_finite_scores() -> None:
    rows = [
        {"sequence_name": "cand0", "start": 1, "stop": 3, "strand": "+", "score": float("nan")},
    ]
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_best_hits(rows)


@pytest.mark.skipif(
    resolve_executable("fimo", tool_path=None) is None,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_run_fimo_smoke(tmp_path: Path) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
            {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
            {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    meme_path = tmp_path / "motif.meme"
    fasta_path = tmp_path / "candidates.fasta"
    write_minimal_meme_motif(motif, meme_path)
    records = build_candidate_records("M1", ["AAA", "CCC"])
    write_candidates_fasta(records, fasta_path)
    rows, _raw = run_fimo(meme_motif_path=meme_path, fasta_path=fasta_path, thresh=1.0)
    assert rows
    for row in rows:
        score = float(row["score"])
        assert math.isfinite(score)
