"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_config.py

Sampling config contract tests for Stage-A PWM sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites, sampling_kwargs_from_config
from dnadesign.densegen.src.adapters.sources.stage_a_types import PWMMotif
from dnadesign.densegen.src.config import PWMMiningBudgetConfig, PWMMiningConfig, PWMSamplingConfig, PWMSelectionConfig


def test_sampling_kwargs_requires_config_object() -> None:
    with pytest.raises(ValueError, match="pwm.sampling config"):
        sampling_kwargs_from_config({"n_sites": 10})


def test_sampling_kwargs_from_config_maps_fields() -> None:
    sampling = PWMSamplingConfig.model_validate(
        {
            "n_sites": 10,
            "mining": {
                "batch_size": 50,
                "budget": {"mode": "fixed_candidates", "candidates": 100},
            },
            "selection": {"policy": "top_score"},
        }
    )
    kwargs = sampling_kwargs_from_config(sampling)
    assert kwargs["n_sites"] == 10
    assert kwargs["strategy"] == "stochastic"
    assert kwargs["selection"] is sampling.selection


def test_sample_pwm_sites_requires_config_objects() -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[{"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    selection = PWMSelectionConfig(policy="top_score")
    with pytest.raises(ValueError, match="pwm.sampling.mining"):
        sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=1,
            mining={"batch_size": 1, "budget": {"mode": "fixed_candidates", "candidates": 1}},
            selection=selection,
        )
    mining = PWMMiningConfig(
        batch_size=1,
        budget=PWMMiningBudgetConfig(mode="fixed_candidates", candidates=1),
    )
    with pytest.raises(ValueError, match="pwm.sampling.selection"):
        sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=1,
            mining=mining,
            selection={"policy": "top_score"},
        )


def test_selection_tier_widening_requires_ladder() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PWMSelectionConfig.model_validate(
            {
                "policy": "mmr",
                "pool": {},
                "tier_widening": {"enabled": True, "ladder": [0.1, 0.2, 1.0]},
            }
        )


def test_selection_mmr_requires_pool() -> None:
    with pytest.raises(ValueError, match="selection.pool"):
        PWMSelectionConfig.model_validate({"policy": "mmr"})


def test_selection_mmr_requires_min_score_norm() -> None:
    selection = PWMSelectionConfig.model_validate({"policy": "mmr", "pool": {}})
    assert selection.pool is not None
    assert selection.pool.min_score_norm is None


def test_sampling_tier_fractions_validation() -> None:
    with pytest.raises(ValueError, match="Tier fractions must contain exactly three values"):
        PWMSamplingConfig.model_validate(
            {
                "n_sites": 10,
                "mining": {
                    "batch_size": 50,
                    "budget": {"mode": "fixed_candidates", "candidates": 100},
                },
                "selection": {"policy": "top_score"},
                "tier_fractions": [0.1, 0.2],
            }
        )
