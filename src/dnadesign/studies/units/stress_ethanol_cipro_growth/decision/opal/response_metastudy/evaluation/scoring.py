"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/scoring.py

Score persisted SFXI evidence under an audit policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.opal import SFXIScoringConfig, score_vec8_with_denom

from ..core.contracts import PolicySpec, SfxiEvidenceFrame


def score_sfxi_evidence(evidence: SfxiEvidenceFrame, policy: PolicySpec) -> pd.DataFrame:
    setpoint = np.asarray(evidence.target_view.target_mask, dtype=float)
    v_hat = np.clip(evidence.y_hat[:, 0:4].astype(float), 0.0, 1.0)
    y_star = evidence.y_hat[:, 4:8].astype(float)
    canonical = score_vec8_with_denom(
        evidence.y_hat,
        SFXIScoringConfig(
            setpoint_vector=evidence.target_view.target_mask,
            scaling_percentile=evidence.scaling_percentile,
            scaling_min_n=evidence.scaling_min_n,
            scaling_eps=evidence.scaling_eps,
            logic_exponent_beta=policy.beta,
            intensity_exponent_gamma=policy.gamma,
            intensity_log2_offset_delta=evidence.intensity_log2_offset_delta,
        ),
        denom=evidence.denom,
    )
    logic = canonical.logic_fidelity
    raw = canonical.effect_raw
    effect = canonical.effect_scaled
    off_state_level = off_state_logic_level(v_hat, setpoint)
    base = canonical.sfxi

    if policy.kind == "multiplicative":
        score = base
        eligible = np.ones(len(score), dtype=bool)
        sort_cols = ["score", "id"]
        ascending = [False, True]
    elif policy.kind == "logic_gate":
        if policy.logic_gate is None:
            raise ValueError(f"{policy.id}: logic_gate policy requires a threshold.")
        eligible = logic >= float(policy.logic_gate)
        score = np.where(eligible, np.power(effect, policy.gamma), np.nan)
        sort_cols = ["eligible", "score", "logic_fidelity", "id"]
        ascending = [False, False, False, True]
    elif policy.kind == "lexicographic":
        eligible = np.ones(len(logic), dtype=bool)
        score = logic
        sort_cols = ["logic_fidelity", "effect_scaled", "id"]
        ascending = [False, False, True]
    elif policy.kind == "off_state_logic_penalty":
        eligible = np.ones(len(logic), dtype=bool)
        penalty = np.power(
            np.clip(1.0 - off_state_level, 0.0, 1.0),
            policy.off_state_logic_eta,
        )
        score = base * penalty
        sort_cols = ["score", "id"]
        ascending = [False, True]
    else:
        raise ValueError(f"Unsupported policy kind: {policy.kind}")

    frame = pd.DataFrame(
        {
            "id": evidence.predictions["id"].astype(str).to_numpy(),
            "sequence": evidence.predictions["sequence"].astype(str).to_numpy(),
            "selection_view_id": evidence.target_view.id,
            "run_id": evidence.run_id,
            "policy_id": policy.id,
            "score": np.asarray(score, dtype=float),
            "logic_fidelity": np.asarray(logic, dtype=float),
            "effect_raw": np.asarray(raw, dtype=float),
            "effect_scaled": np.asarray(effect, dtype=float),
            "off_state_logic_level": np.asarray(off_state_level, dtype=float),
            "eligible": np.asarray(eligible, dtype=bool),
            "v00": v_hat[:, 0],
            "v10": v_hat[:, 1],
            "v01": v_hat[:, 2],
            "v11": v_hat[:, 3],
            "y00_star": y_star[:, 0],
            "y10_star": y_star[:, 1],
            "y01_star": y_star[:, 2],
            "y11_star": y_star[:, 3],
        }
    )
    ranked = frame.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def off_state_logic_level(v_hat: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
    """Return mean normalized logic in states declared OFF by the setpoint.

    This is not an absolute-fluorescence measure; SFXI logic channels and
    intensity channels are distinct parts of vec8.
    """
    off_mask = np.asarray(setpoint, dtype=float) <= 0.0
    if not np.any(off_mask):
        return np.zeros(v_hat.shape[0], dtype=float)
    return np.mean(np.clip(v_hat[:, off_mask], 0.0, 1.0), axis=1)
