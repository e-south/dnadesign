"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_pwm_validation.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from dnadesign.cruncher.core.pwm import PWM


def test_pwm_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        PWM(name="bad", matrix=np.array([[0.5, 0.5]]))


def test_pwm_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        PWM(name="bad", matrix=np.array([[0.5, -0.5, 1.0, 0.0]]))


def test_pwm_rejects_rows_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        PWM(name="bad", matrix=np.array([[0.2, 0.2, 0.2, 0.2]]))


def test_pwm_accepts_valid_matrix() -> None:
    pwm = PWM(name="ok", matrix=np.array([[0.25, 0.25, 0.25, 0.25]]))
    assert pwm.length == 1
