"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/metric_contract.py

Synthetic contracts for canonical SFXI and RMF behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dnadesign.opal import (
    SFXIScoringConfig,
    response_magnitude_feasibility_components,
    score_response_magnitude_feasibility,
    score_vec8_with_denom,
)

from ..core.contracts import StressTargetView


def build_metric_contract_tests(target_views: Sequence[StressTargetView]) -> pd.DataFrame:
    rows = [_setpoint_identity_check(target_view, target_views) for target_view in target_views]
    rows.append(_off_state_intensity_check(target_views[0]))
    rows.append(_exponent_scale_invariance_check(target_views[0]))
    rows.extend((_rmf_noncompensation_check(), _rmf_permutation_check()))
    return pd.DataFrame(rows)


def build_rmf_cardinality_pressure(
    *,
    state_counts: tuple[int, ...] = (2, 4, 8, 16),
    draws: int = 5_000,
    noise_sd: float = 0.25,
    seed: int = 20260712,
) -> pd.DataFrame:
    """Measure worst-state noise bias as the state panel and mask change."""

    if not state_counts or any(count < 2 for count in state_counts):
        raise ValueError("RMF cardinality screen requires state counts >= 2.")
    if draws < 1_000 or not np.isfinite(noise_sd) or noise_sd <= 0.0:
        raise ValueError("RMF cardinality screen requires >= 1000 draws and positive finite noise_sd.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for state_count in state_counts:
        on_counts = sorted({1, state_count // 2, state_count - 1})
        for on_count in on_counts:
            target = np.zeros(state_count, dtype=int)
            target[-on_count:] = 1
            response_truth = target.astype(float)
            magnitude_truth = np.where(target == 1, 0.5, -0.5)
            truth = np.concatenate((response_truth, magnitude_truth))
            noisy = truth[None, :] + rng.normal(0.0, noise_sd, size=(draws, 2 * state_count))
            components = response_magnitude_feasibility_components(noisy, target_mask=target)
            rows.append(
                {
                    "state_count": state_count,
                    "on_count": on_count,
                    "off_count": state_count - on_count,
                    "mask_topology": _mask_topology(state_count=state_count, on_count=on_count),
                    "draws": draws,
                    "noise_sd": noise_sd,
                    "response_separation_bias": float(np.mean(components.response_separation) - 1.0),
                    "on_magnitude_floor_bias": float(np.mean(components.on_magnitude_floor) - 0.5),
                    "off_magnitude_ceiling_bias": float(np.mean(components.off_magnitude_ceiling) + 0.5),
                    "all_zero_constraints_pass_fraction": float(
                        np.mean(
                            (components.response_separation > 0.0)
                            & (components.on_magnitude_floor >= 0.0)
                            & (components.off_magnitude_ceiling <= 0.0)
                        )
                    ),
                    "interpretation_boundary": "synthetic_iid_noise_not_assay_calibration",
                }
            )
    return pd.DataFrame.from_records(rows)


def _rmf_noncompensation_check() -> dict[str, object]:
    values = np.asarray(
        [
            [0.0, 2.0, 0.2, 1.8, -1.0, 0.5, -0.8, 0.7],
            [0.0, 0.2, 2.0, 1.8, -1.0, 4.0, 4.0, 4.0],
        ]
    )
    calibration = {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 1.0,
        "on_magnitude_scale": 1.0,
        "off_magnitude_scale": 1.0,
    }
    result = score_response_magnitude_feasibility(
        values,
        target_mask=[0, 1, 0, 1],
        calibration=calibration,
    )
    passes = bool(result.feasibility_margin[0] > 0.0 and result.feasibility_margin[1] < 0.0)
    return _row(
        check_id="rmf_noncompensation",
        status="pass" if passes else "fail",
        severity="blocker",
        premise="Large output magnitude cannot compensate for failed target response separation.",
        evidence=(
            f"right_and_bright={result.feasibility_margin[0]:.3f}; bright_but_wrong={result.feasibility_margin[1]:.3f}"
        ),
        interpretation="RMF is controlled by the weakest signed requirement.",
        action="Keep every RMF component and the limiting requirement in selection-view evidence and plots.",
    )


def _rmf_permutation_check() -> dict[str, object]:
    values = np.asarray([[0.0, 2.0, 1.0, -0.5, 0.8, 0.2]])
    direct = response_magnitude_feasibility_components(values, target_mask=[0, 1, 1])
    permuted = response_magnitude_feasibility_components(
        values[:, [2, 0, 1, 5, 3, 4]],
        target_mask=[1, 0, 1],
    )
    passes = all(
        np.allclose(left, right)
        for left, right in (
            (direct.response_separation, permuted.response_separation),
            (direct.on_magnitude_floor, permuted.on_magnitude_floor),
            (direct.off_magnitude_ceiling, permuted.off_magnitude_ceiling),
        )
    )
    return _row(
        check_id="rmf_state_permutation_equivariance",
        status="pass" if passes else "fail",
        severity="blocker",
        premise="Jointly reordering state values and the target mask must preserve RMF components.",
        evidence=f"components_equal={passes}",
        interpretation="The objective depends on explicit state alignment, not two-factor state names.",
        action="Require ordered state_ids and reject state/mask length mismatch.",
    )


def _mask_topology(*, state_count: int, on_count: int) -> str:
    if on_count == 1:
        return "one ON"
    if on_count == state_count - 1:
        return "one OFF"
    return "balanced"


def _setpoint_identity_check(
    target_view: StressTargetView,
    candidates: Sequence[StressTargetView],
) -> dict[str, object]:
    vec8 = np.asarray([(*candidate.target_mask, 0.0, 0.0, 0.0, 0.0) for candidate in candidates], dtype=float)
    result = score_vec8_with_denom(vec8, _config(target_view), denom=1.0)
    own_index = [candidate.id for candidate in candidates].index(target_view.id)
    unique_best = bool(
        np.isclose(result.logic_fidelity[own_index], 1.0)
        and np.count_nonzero(np.isclose(result.logic_fidelity, np.max(result.logic_fidelity))) == 1
    )
    return _row(
        check_id=f"setpoint_identity_{target_view.id}",
        status="pass" if unique_best else "fail",
        severity="blocker",
        premise="With equal target-state intensity, an exact setpoint match must uniquely maximize logic fidelity.",
        evidence=(
            f"own_logic={result.logic_fidelity[own_index]:.3f}; "
            f"other_logic={np.delete(result.logic_fidelity, own_index).round(3).tolist()}"
        ),
        interpretation="The canonical setpoint geometry distinguishes the three declared response shapes.",
        action="Treat later target-view collapse as a data or scalarization issue, not target identity failure.",
    )


def _off_state_intensity_check(target_view: StressTargetView) -> dict[str, object]:
    base = np.asarray([(*target_view.target_mask, 0.0, 0.0, 0.0, 0.0)], dtype=float)
    changed = base.copy()
    off_mask = np.asarray(target_view.target_mask) <= 0.0
    changed[0, 4:8][off_mask] = 8.0
    base_score = float(score_vec8_with_denom(base, _config(target_view), denom=1.0).sfxi[0])
    changed_score = float(score_vec8_with_denom(changed, _config(target_view), denom=1.0).sfxi[0])
    unchanged = bool(np.isclose(base_score, changed_score, rtol=0.0, atol=1.0e-12))
    return _row(
        check_id="off_state_absolute_intensity_boundary",
        status="attention" if unchanged else "fail",
        severity="high",
        premise="Canonical SFXI weights absolute intensity only in states valued by the setpoint.",
        evidence=f"base_score={base_score:.6g}; changed_off_state_score={changed_score:.6g}",
        interpretation=(
            "High absolute fluorescence in setpoint-OFF states is invisible to the effect term by design; only the "
            "normalized logic block can penalize it."
        ),
        action="Do not describe the study-local OFF-state logic penalty as an absolute-fluorescence penalty.",
    )


def _exponent_scale_invariance_check(target_view: StressTargetView) -> dict[str, object]:
    logic = np.asarray([0.2, 0.4, 0.6, 0.8])
    effect = np.asarray([0.9, 0.7, 0.5, 0.3])
    score = logic * effect
    scaled = np.power(logic, 4.0) * np.power(effect, 4.0)
    same_order = bool(np.array_equal(np.argsort(-score, kind="stable"), np.argsort(-scaled, kind="stable")))
    return _row(
        check_id="positive_exponent_scale_rank_invariance",
        status="pass" if same_order else "fail",
        severity="blocker",
        premise="Multiplying beta and gamma by the same positive constant cannot change Top-N ordering.",
        evidence=f"selection_view_id={target_view.id}; rank_order_equal={same_order}",
        interpretation="Only the beta-to-gamma tradeoff is identifiable for deterministic Top-N ranking.",
        action="Keep beta + gamma normalized in parameter screens.",
    )


def _config(target_view: StressTargetView) -> SFXIScoringConfig:
    return SFXIScoringConfig(setpoint_vector=target_view.target_mask)


def _row(
    *,
    check_id: str,
    status: str,
    severity: str,
    premise: str,
    evidence: str,
    interpretation: str,
    action: str,
) -> dict[str, object]:
    return {
        "agent": "intrinsic_metric",
        "check_id": check_id,
        "status": status,
        "severity": severity,
        "premise": premise,
        "evidence": evidence,
        "threshold": "canonical mathematical contract",
        "interpretation": interpretation,
        "action": action,
    }
