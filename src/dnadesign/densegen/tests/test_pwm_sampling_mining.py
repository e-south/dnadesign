"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_mining.py

FIMO mining behavior for Stage-A PWM sampling.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dnadesign.densegen.src.adapters.sources import pwm_fimo
from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def _parse_fasta(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line.strip().lstrip(">"))
    return ids


def test_pwm_sampling_fimo_mining_retain_depth(monkeypatch) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    def fake_run_fimo(*, meme_motif_path, fasta_path, **_kwargs):  # type: ignore[override]
        ids = _parse_fasta(Path(fasta_path))
        rows = []
        for idx, rec_id in enumerate(ids):
            pval = 1e-6 if idx < 2 else 1e-2
            rows.append(
                {
                    "sequence_name": rec_id,
                    "start": 1,
                    "stop": 3,
                    "strand": "+",
                    "score": 5.0,
                    "p_value": pval,
                    "matched_sequence": "AAA",
                }
            )
        return rows, None

    monkeypatch.setattr(pwm_fimo, "run_fimo", fake_run_fimo)

    rng = np.random.default_rng(0)
    selected, meta = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=2,
        oversample_factor=2,
        max_candidates=None,
        max_seconds=None,
        score_threshold=None,
        score_percentile=None,
        scoring_backend="fimo",
        pvalue_strata=[1e-5, 1e-3, 1.0],
        retain_depth=1,
        mining={
            "batch_size": 2,
            "max_batches": 2,
            "log_every_batches": 1,
        },
        include_matched_sequence=True,
        return_metadata=True,
    )

    assert len(selected) == 2
    for seq in selected:
        info = meta[seq]
        assert info["fimo_bin_id"] == 0
        assert info["fimo_matched_sequence"] == "AAA"


def test_pwm_sampling_fimo_mining_max_candidates_guard(monkeypatch, caplog) -> None:
    motif = PWMMotif(
        motif_id="M2",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    def fake_run_fimo(*, meme_motif_path, fasta_path, **_kwargs):  # type: ignore[override]
        ids = _parse_fasta(Path(fasta_path))
        rows = []
        for rec_id in ids:
            rows.append(
                {
                    "sequence_name": rec_id,
                    "start": 1,
                    "stop": 2,
                    "strand": "+",
                    "score": 5.0,
                    "p_value": 1e-6,
                    "matched_sequence": "AA",
                }
            )
        return rows, None

    monkeypatch.setattr(pwm_fimo, "run_fimo", fake_run_fimo)
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        selected = sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=5,
            oversample_factor=1,
            max_candidates=None,
            max_seconds=None,
            score_threshold=None,
            score_percentile=None,
            scoring_backend="fimo",
            pvalue_strata=[1e-5, 1e-3, 1.0],
            retain_depth=2,
            mining={"batch_size": 2, "max_candidates": 2},
        )
    assert len(selected) == 2
    assert "shortfall" in caplog.text
