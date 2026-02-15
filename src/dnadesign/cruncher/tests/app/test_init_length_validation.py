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

from dnadesign.cruncher.app.sample.resources import _apply_sampling_pwm_window
from dnadesign.cruncher.app.sample.run_set import _assert_init_length_fits_pwms
from dnadesign.cruncher.config.schema_v3 import (
    SampleBudgetConfig,
    SampleConfig,
    SampleMotifWidthConfig,
    SampleObjectiveConfig,
)
from dnadesign.cruncher.core.pwm import PWM


def _sample_config(*, length: int, minw: int | None = None, maxw: int | None = None) -> SampleConfig:
    return SampleConfig(
        seed=7,
        sequence_length=length,
        budget=SampleBudgetConfig(tune=1, draws=2),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
        motif_width=SampleMotifWidthConfig(minw=minw, maxw=maxw),
    )


def _pwm(name: str, length: int) -> PWM:
    matrix = np.full((length, 4), 0.25, dtype=float)
    return PWM(name=name, matrix=matrix)


def test_init_length_rejects_shorter_than_max_pwm() -> None:
    cfg = _sample_config(length=3)
    pwms = {"lexA": _pwm("lexA", 4), "cpxR": _pwm("cpxR", 3)}
    with pytest.raises(ValueError, match="cruncher.sample.motif_width.maxw"):
        _assert_init_length_fits_pwms(cfg, pwms)


def test_init_length_accepts_equal_to_max_pwm() -> None:
    cfg = _sample_config(length=4)
    pwms = {"lexA": _pwm("lexA", 4), "cpxR": _pwm("cpxR", 3)}
    _assert_init_length_fits_pwms(cfg, pwms)


def test_sampling_window_trims_to_default_sequence_length_max() -> None:
    cfg = _sample_config(length=4)
    pwm = _pwm("lexA", 6)
    trimmed = _apply_sampling_pwm_window(tf_name="lexA", pwm=pwm, sample_cfg=cfg)
    assert trimmed.length == 4
    assert trimmed.window_strategy == "max_info"
    assert trimmed.source_length == 6


def test_sampling_window_rejects_below_min_width() -> None:
    cfg = _sample_config(length=16, minw=12, maxw=15)
    pwm = _pwm("lexA", 10)
    with pytest.raises(ValueError, match="below sample.motif_width.minw=12"):
        _apply_sampling_pwm_window(tf_name="lexA", pwm=pwm, sample_cfg=cfg)


def test_sampling_window_uses_explicit_maxw() -> None:
    cfg = _sample_config(length=16, maxw=10)
    pwm = _pwm("lexA", 12)
    trimmed = _apply_sampling_pwm_window(tf_name="lexA", pwm=pwm, sample_cfg=cfg)
    assert trimmed.length == 10


def test_sampling_window_logs_effective_width_without_trim(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _sample_config(length=16, maxw=16)
    pwm = _pwm("lexA", 11)
    with caplog.at_level("INFO"):
        trimmed = _apply_sampling_pwm_window(tf_name="lexA", pwm=pwm, sample_cfg=cfg)
    assert trimmed.length == 11
    assert "Sampling PWM width lexA: source=11bp effective=11bp" in caplog.text
    assert "action=unchanged" in caplog.text


def test_sampling_window_logs_trim_action(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _sample_config(length=16, maxw=10)
    pwm = _pwm("lexA", 12)
    with caplog.at_level("INFO"):
        trimmed = _apply_sampling_pwm_window(tf_name="lexA", pwm=pwm, sample_cfg=cfg)
    assert trimmed.length == 10
    assert "Sampling PWM width lexA: source=12bp effective=10bp" in caplog.text
    assert "action=trimmed" in caplog.text


def test_sampling_window_rejects_maxw_above_sequence_length() -> None:
    with pytest.raises(ValueError, match="sample.motif_width.maxw must be <= sample.sequence_length"):
        _sample_config(length=16, maxw=20)


def test_sampling_window_rejects_minw_above_sequence_length() -> None:
    with pytest.raises(ValueError, match="sample.motif_width.minw must be <= sample.sequence_length"):
        _sample_config(length=16, minw=20)
