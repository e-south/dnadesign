"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_init_length_validation.py

Validates fixed-length constraints against PWM widths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.app.sample.run_set import _assert_init_length_fits_pwms
from dnadesign.cruncher.config.schema_v3 import SampleBudgetConfig, SampleConfig, SampleObjectiveConfig
from dnadesign.cruncher.core.pwm import PWM


def _sample_config(*, length: int) -> SampleConfig:
    return SampleConfig(
        seed=7,
        sequence_length=length,
        budget=SampleBudgetConfig(tune=1, draws=2),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
    )


def _pwm(name: str, length: int) -> PWM:
    matrix = np.full((length, 4), 0.25, dtype=float)
    return PWM(name=name, matrix=matrix)


def test_init_length_rejects_shorter_than_max_pwm() -> None:
    cfg = _sample_config(length=3)
    pwms = {"lexA": _pwm("lexA", 4), "cpxR": _pwm("cpxR", 3)}
    with pytest.raises(ValueError, match="cruncher.catalog.pwm_window_lengths"):
        _assert_init_length_fits_pwms(cfg, pwms)


def test_init_length_accepts_equal_to_max_pwm() -> None:
    cfg = _sample_config(length=4)
    pwms = {"lexA": _pwm("lexA", 4), "cpxR": _pwm("cpxR", 3)}
    _assert_init_length_fits_pwms(cfg, pwms)
