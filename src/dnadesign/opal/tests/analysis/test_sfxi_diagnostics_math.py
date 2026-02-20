"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_diagnostics_math.py

Tests core SFXI diagnostics math utilities (factorial effects, gates, support,
and intensity scaling).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dnadesign.opal.src.analysis.sfxi.factorial_effects import compute_factorial_effects
from dnadesign.opal.src.analysis.sfxi.gates import nearest_gate
from dnadesign.opal.src.analysis.sfxi.intensity_scaling import summarize_intensity_scaling
from dnadesign.opal.src.analysis.sfxi.setpoint_sweep import sweep_setpoints
from dnadesign.opal.src.analysis.sfxi.support import dist_to_labeled_logic
from dnadesign.opal.src.objectives import sfxi_math


def test_factorial_effects_known_vectors():
    v = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # AND
            [1.0, 1.0, 1.0, 1.0],  # TRUE
        ],
        dtype=float,
    )
    a, b, ab = compute_factorial_effects(v, state_order=sfxi_math.STATE_ORDER)
    assert np.allclose(a[0], 0.5)
    assert np.allclose(b[0], 0.5)
    assert np.allclose(ab[0], 0.5)
    assert np.allclose(a[1], 0.0)
    assert np.allclose(b[1], 0.0)
    assert np.allclose(ab[1], 0.0)


def test_nearest_gate_assigns_truth_table_to_self():
    v = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    classes, distances = nearest_gate(v, state_order=sfxi_math.STATE_ORDER)
    assert classes[0] == "0001"
    assert classes[1] == "1111"
    assert np.allclose(distances, 0.0)


def test_nearest_gate_prefers_expected_table():
    v = np.array([[0.05, 0.1, 0.05, 0.9]], dtype=float)
    classes, distances = nearest_gate(v, state_order=sfxi_math.STATE_ORDER)
    assert classes[0] == "0001"
    assert distances[0] >= 0.0


def test_dist_to_labeled_logic_min_l2():
    labels = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    candidates = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    dists = dist_to_labeled_logic(candidates, labels, state_order=sfxi_math.STATE_ORDER)
    expected_second = min(
        np.linalg.norm(candidates[1] - labels[0]),
        np.linalg.norm(candidates[1] - labels[1]),
    )
    assert np.allclose(dists[0], 0.0)
    assert np.allclose(dists[1], expected_second)


def test_intensity_scaling_denom_and_clipping():
    y_star = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    setpoint = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    summary = summarize_intensity_scaling(
        y_star,
        setpoint=setpoint,
        delta=0.0,
        percentile=50,
        min_n=1,
        eps=1e-8,
        state_order=sfxi_math.STATE_ORDER,
    )
    assert np.isclose(summary.denom, 2.0)
    assert np.isclose(summary.clip_hi_fraction, 2.0 / 3.0)
    assert np.isclose(summary.clip_lo_fraction, 0.0)


def test_effect_raw_from_y_star_matches_manual():
    y_star = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    setpoint = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    effect_raw, weights = sfxi_math.effect_raw_from_y_star(
        y_star,
        setpoint,
        delta=0.0,
        eps=1e-8,
        state_order=sfxi_math.STATE_ORDER,
    )
    manual_weights = sfxi_math.weights_from_setpoint(setpoint, eps=1e-8)
    manual_lin = sfxi_math.recover_linear_intensity(y_star, delta=0.0)
    manual_raw = sfxi_math.effect_raw(manual_lin, manual_weights)
    assert np.allclose(weights, manual_weights)
    assert np.allclose(effect_raw, manual_raw)


def test_setpoint_sweep_pool_clip_uses_label_denom():
    labels_vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    pool_vec8 = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0],
        ],
        dtype=float,
    )

    sweep_df = sweep_setpoints(
        labels_vec8=labels_vec8,
        current_setpoint=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        percentile=50,
        min_n=1,
        eps=1e-8,
        delta=0.0,
        beta=1.0,
        gamma=1.0,
        pool_vec8=pool_vec8,
        state_order=sfxi_math.STATE_ORDER,
    )
    assert "setpoint_label" in sweep_df.columns
    assert "setpoint_vector" in sweep_df.columns
    assert "logic_fidelity" in sweep_df.columns
    assert "effect_scaled" in sweep_df.columns
    assert "score" in sweep_df.columns
    target = sweep_df.filter(pl.col("setpoint_name") == "0001")
    assert target.height == 1
    pool_clip_hi = float(target.get_column("pool_clip_hi_fraction").item())
    pool_clip_lo = float(target.get_column("pool_clip_lo_fraction").item())
    assert np.isclose(pool_clip_hi, 1.0)
    assert np.isclose(pool_clip_lo, 0.0)
