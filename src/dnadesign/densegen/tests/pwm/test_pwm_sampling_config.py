"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_sampling_config.py

Sampling config contract tests for Stage-A PWM sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    enforce_cross_regulator_core_collisions,
    sample_pwm_sites,
    sampling_kwargs_from_config,
    validate_mmr_core_length,
)
from dnadesign.densegen.src.config import PWMMiningBudgetConfig, PWMMiningConfig, PWMSamplingConfig, PWMSelectionConfig
from dnadesign.densegen.src.core.stage_a.stage_a_pipeline import StageAPipelineResult
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif


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
    assert kwargs["cross_regulator_core_collisions"] == "warn"


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


def test_sampling_uniqueness_cross_regulator_core_collisions_validation() -> None:
    with pytest.raises(ValueError, match="cross_regulator_core_collisions"):
        PWMSamplingConfig.model_validate(
            {
                "n_sites": 10,
                "mining": {
                    "batch_size": 50,
                    "budget": {"mode": "fixed_candidates", "candidates": 100},
                },
                "selection": {"policy": "top_score"},
                "uniqueness": {"key": "core", "cross_regulator_core_collisions": "invalid"},
            }
        )


def test_pwm_sampling_length_defaults_to_range_16_20() -> None:
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
    assert sampling.length.policy == "range"
    assert sampling.length.range == (16, 20)


def test_pwm_sampling_length_exact_allows_implicit_default_range() -> None:
    sampling = PWMSamplingConfig.model_validate(
        {
            "n_sites": 10,
            "mining": {
                "batch_size": 50,
                "budget": {"mode": "fixed_candidates", "candidates": 100},
            },
            "selection": {"policy": "top_score"},
            "length": {"policy": "exact"},
        }
    )
    assert sampling.length.policy == "exact"
    assert sampling.length.range is None


def test_pwm_sampling_length_exact_rejects_explicit_range() -> None:
    with pytest.raises(ValueError, match="pwm.sampling.length.range is not allowed when policy=exact"):
        PWMSamplingConfig.model_validate(
            {
                "n_sites": 10,
                "mining": {
                    "batch_size": 50,
                    "budget": {"mode": "fixed_candidates", "candidates": 100},
                },
                "selection": {"policy": "top_score"},
                "length": {"policy": "exact", "range": [16, 20]},
            }
        )


def test_validate_mmr_core_length_requires_explicit_selection_policy() -> None:
    with pytest.raises(ValueError, match="selection.policy must be a non-empty string"):
        validate_mmr_core_length(
            motif_id="M1",
            motif_width=20,
            selection_policy="",
            length_policy="range",
            length_range=(16, 20),
            trim_window_length=16,
        )


def test_enforce_cross_regulator_core_collisions_requires_explicit_mode() -> None:
    with pytest.raises(ValueError, match="cross_regulator_core_collisions must be a non-empty string"):
        enforce_cross_regulator_core_collisions(
            rows=[],
            mode="",
            input_name="demo",
            source_kind="pwm_meme",
        )


def test_sample_pwm_sites_passes_stage_a_request_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
            {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
            {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
            {"A": 0.1, "C": 0.1, "G": 0.1, "T": 0.7},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    mining = PWMMiningConfig(
        batch_size=1,
        budget=PWMMiningBudgetConfig(mode="fixed_candidates", candidates=1),
    )
    selection = PWMSelectionConfig(policy="top_score")
    captured: dict[str, object] = {}

    def _fake_run_stage_a_pipeline(**kwargs):
        captured.update(kwargs)
        return StageAPipelineResult(sequences=["ACGT"], meta_by_seq={}, summary=None)

    monkeypatch.setattr(
        "dnadesign.densegen.src.adapters.sources.pwm_sampling.run_stage_a_pipeline",
        _fake_run_stage_a_pipeline,
    )
    rng = np.random.default_rng(0)

    selected = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=mining,
        selection=selection,
        length_policy="exact",
    )

    assert selected == ["ACGT"]
    assert "mining_request" in captured
    assert "selection_request" in captured
    assert "summary_request" in captured
