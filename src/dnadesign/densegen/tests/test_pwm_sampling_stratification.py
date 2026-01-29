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
import pandas as pd
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_fimo_retains_top_scores_with_dedup(tmp_path: Path) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
            {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )

    rng = np.random.default_rng(0)
    selected, meta = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=3,
        mining=fixed_candidates_mining(batch_size=10, candidates=50, log_every_batches=1),
        selection=selection_top_score(),
        keep_all_candidates_debug=True,
        run_id="test",
        debug_output_dir=tmp_path,
        return_metadata=True,
    )

    assert selected
    parquet_path = next(tmp_path.glob("candidates__*.parquet"))
    df = pd.read_parquet(parquet_path)
    accepted = df[df["accepted"]].copy()
    assert not accepted.empty
    dedup = (
        accepted.groupby("sequence", as_index=False)["best_hit_score"]
        .max()
        .sort_values(["best_hit_score", "sequence"], ascending=[False, True])
    )
    expected = dedup.head(len(selected))["sequence"].tolist()
    assert selected == expected
    scores = [meta[seq]["best_hit_score"] for seq in selected]
    assert scores == sorted(scores, reverse=True)
