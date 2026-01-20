from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import pwm_fimo
from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites


def _parse_fasta(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line.strip().lstrip(">"))
    return ids


def test_pwm_sampling_fimo_mining_retain_bins(monkeypatch) -> None:
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
            pval = 1e-6 if idx % 2 == 0 else 1e-2
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
        pvalue_threshold=1e-1,
        pvalue_bins=[1e-5, 1e-3, 1.0],
        selection_policy="random_uniform",
        mining={
            "batch_size": 2,
            "max_batches": 2,
            "retain_bin_ids": [0],
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


def test_pwm_sampling_fimo_mining_max_candidates_guard() -> None:
    motif = PWMMotif(
        motif_id="M2",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mining.max_candidates must be >= n_sites"):
        sample_pwm_sites(
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
            pvalue_threshold=1e-2,
            mining={"batch_size": 2, "max_candidates": 2},
            selection_policy="random_uniform",
        )
