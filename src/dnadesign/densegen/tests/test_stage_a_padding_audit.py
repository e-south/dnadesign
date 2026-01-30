"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_padding_audit.py

Stage-A padding audit stats for core offsets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.adapters.sources.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _motif() -> PWMMotif:
    matrix = [
        {"A": 0.9, "C": 0.0333, "G": 0.0333, "T": 0.0334},
        {"A": 0.9, "C": 0.0333, "G": 0.0333, "T": 0.0334},
        {"A": 0.9, "C": 0.0333, "G": 0.0333, "T": 0.0334},
        {"A": 0.9, "C": 0.0333, "G": 0.0333, "T": 0.0334},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    return PWMMotif(motif_id="M1", matrix=matrix, background=background)


def test_padding_offsets_non_constant_and_reproducible() -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    rng = np.random.default_rng(42)
    mining = fixed_candidates_mining(batch_size=20, candidates=120)
    selection = selection_top_score()
    args = dict(
        input_name="demo_input",
        motif=_motif(),
        strategy="stochastic",
        n_sites=5,
        mining=mining,
        selection=selection,
        length_policy="range",
        length_range=[6, 8],
        return_summary=True,
    )
    selected1, summary1 = sample_pwm_sites(rng, **args)
    rng = np.random.default_rng(42)
    selected2, summary2 = sample_pwm_sites(rng, **args)
    assert selected1 == selected2
    assert summary1 is not None
    padding = summary1.padding_audit or {}
    histogram = padding.get("core_offset_histogram", {})
    bins = histogram.get("bins", [])
    assert len(bins) > 1
