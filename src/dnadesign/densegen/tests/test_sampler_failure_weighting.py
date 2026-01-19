from __future__ import annotations

import pandas as pd

from dnadesign.densegen.src.core.sampler import TFSampler


class _DummyRng:
    def __init__(self):
        self.choice_calls = []

    def choice(self, candidates, p=None):
        self.choice_calls.append(p)
        return candidates[0]

    def shuffle(self, values):
        return values


def test_failure_weighting_reduces_weight_for_failed_sites() -> None:
    df = pd.DataFrame(
        {
            "tf": ["TF1", "TF1"],
            "tfbs": ["AAAA", "CCCC"],
        }
    )
    rng = _DummyRng()
    sampler = TFSampler(df, rng)
    sampler.generate_binding_site_library(
        1,
        sequence_length=10,
        budget_overhead=0,
        sampling_strategy="coverage_weighted",
        usage_counts={},
        coverage_boost_alpha=0.0,
        coverage_boost_power=1.0,
        failure_counts={("TF1", "AAAA"): 10},
        avoid_failed_motifs=True,
        failure_penalty_alpha=1.0,
        failure_penalty_power=1.0,
    )
    assert rng.choice_calls, "Expected weighted choice call"
    weights = rng.choice_calls[0]
    assert weights[0] < weights[1]
