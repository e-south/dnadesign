import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.pwm_window import select_pwm_window


def test_select_pwm_window_max_info():
    matrix = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
        ]
    )
    pwm = PWM(name="tf", matrix=matrix)
    trimmed = select_pwm_window(pwm, length=2, strategy="max_info")
    assert trimmed.length == 2
    assert trimmed.window_start == 1
    assert trimmed.source_length == 3
    assert trimmed.window_strategy == "max_info"
