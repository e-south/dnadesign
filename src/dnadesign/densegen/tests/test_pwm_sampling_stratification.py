"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_stratification.py

FIMO stratification behavior for Stage-A PWM sampling.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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


def test_fimo_stratification_selects_top_n_within_retain_depth(monkeypatch) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    def fake_run_fimo(*, meme_motif_path, fasta_path, **_kwargs):  # type: ignore[override]
        ids = _parse_fasta(Path(fasta_path))
        rows = []
        for idx, rec_id in enumerate(ids):
            if idx == 0:
                pval, score = 1e-8, 3.0
            elif idx == 1:
                pval, score = 1e-8, 7.0
            elif idx == 2:
                pval, score = 1e-6, 9.0
            else:
                pval, score = 1e-3, 2.0
            rows.append(
                {
                    "sequence_name": rec_id,
                    "start": 1,
                    "stop": 2,
                    "strand": "+",
                    "score": score,
                    "p_value": pval,
                    "matched_sequence": "AA",
                }
            )
        return rows, None

    monkeypatch.setattr(pwm_fimo, "run_fimo", fake_run_fimo)

    rng = np.random.default_rng(0)
    selected, meta = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=1,
        oversample_factor=4,
        max_candidates=None,
        max_seconds=None,
        score_threshold=None,
        score_percentile=None,
        scoring_backend="fimo",
        pvalue_strata=[1e-8, 1e-6, 1e-4],
        retain_depth=1,
        mining={"batch_size": 4, "max_batches": 1},
        return_metadata=True,
    )

    assert len(selected) == 1
    info = meta[selected[0]]
    assert info["fimo_pvalue"] == 1e-8
    assert info["fimo_score"] == 7.0
