from __future__ import annotations

import numpy as np

from dnadesign.opal.api.sfxi import SFXI_API_VERSION, SFXIScoringConfig, score_vec8


def test_score_vec8_public_api_uses_objective_channel_names() -> None:
    vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cfg = SFXIScoringConfig(
        setpoint_vector=(0.0, 0.0, 0.0, 1.0),
        scaling_percentile=95,
        scaling_min_n=1,
        scaling_eps=1.0e-8,
        logic_exponent_beta=1.0,
        intensity_exponent_gamma=1.0,
        intensity_log2_offset_delta=0.0,
    )

    result = score_vec8(vec8, cfg, scaling_vec8=vec8[0:1, :])

    assert result.api_version == SFXI_API_VERSION
    assert result.objective_name == "sfxi_v1"
    assert result.denom_used == 2.0
    assert np.allclose(result.logic_fidelity, np.array([1.0, 1.0]))
    assert np.allclose(result.effect_raw, np.array([2.0, 1.0]))
    assert np.allclose(result.effect_scaled, np.array([1.0, 0.5]))
    assert np.allclose(result.sfxi, np.array([1.0, 0.5]))

    row = result.to_records()[0]
    assert {"logic_fidelity", "effect_raw", "effect_scaled", "sfxi"} <= set(row)
    assert "f_logic" not in row
    assert "e_scaled" not in row
    assert "score" not in row
