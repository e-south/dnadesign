"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_sampling_config.py

Sampling config contract tests for Stage-A PWM sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sampling_kwargs_from_config
from dnadesign.densegen.src.config import PWMSamplingConfig


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
