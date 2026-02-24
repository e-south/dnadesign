"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_points.py

Validate trajectory point construction for plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.analysis.trajectory import (
    add_raw_llr_objective,
    build_chain_trajectory_points,
    build_trajectory_points,
)
from dnadesign.cruncher.core.pwm import PWM


def test_build_trajectory_points_keeps_all_chains() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "phase": ["tune", "draw", "tune", "draw"],
            "score_lexA": [0.1, 0.2, 0.3, 0.4],
            "score_cpxR": [0.2, 0.3, 0.1, 0.2],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=100,
        beta_ladder=[1.0, 0.5],
    )

    assert set(trajectory_df["chain"].astype(int).unique()) == {0, 1}
    assert "is_cold_chain" not in trajectory_df.columns


def test_build_trajectory_points_subsample_keeps_dense_early_sweeps() -> None:
    draws = list(range(100))
    sequences_df = pd.DataFrame(
        {
            "chain": [0] * 100,
            "draw": draws,
            "phase": ["draw"] * 100,
            "score_lexA": [0.5 + (i * 0.001) for i in draws],
            "score_cpxR": [0.3 + (i * 0.001) for i in draws],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=10,
        beta_ladder=[1.0],
    )

    assert len(trajectory_df) == 10
    assert list(trajectory_df["sweep"].astype(int).head(4)) == [0, 1, 2, 3]


def test_build_trajectory_points_subsample_preserves_chain_objective_maximum() -> None:
    draws = list(range(100))
    chain0_scores = [0.1] * 100
    chain0_scores[50] = 9.9
    chain1_scores = [0.2] * 100
    chain1_scores[70] = 8.8
    sequences_df = pd.DataFrame(
        {
            "chain": [0] * 100 + [1] * 100,
            "draw": draws + draws,
            "phase": ["draw"] * 200,
            "score_lexA": chain0_scores + chain1_scores,
            "score_cpxR": chain0_scores + chain1_scores,
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=20,
        beta_ladder=[1.0, 0.5],
    )

    by_chain = trajectory_df.groupby("chain")["objective_scalar"].max().to_dict()
    assert by_chain[0] == pytest.approx(9.9)
    assert by_chain[1] == pytest.approx(8.8)


def test_build_trajectory_points_subsample_retains_requested_sequences() -> None:
    draws = list(range(40))
    sequences_df = pd.DataFrame(
        {
            "chain": [0] * 40,
            "draw": draws,
            "phase": ["draw"] * 40,
            "sequence": [f"seq_{idx}" for idx in draws],
            "score_lexA": [0.1 + (idx * 0.001) for idx in draws],
            "score_cpxR": [0.2 + (idx * 0.001) for idx in draws],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=8,
        beta_ladder=[1.0],
        retain_sequences={"seq_23", "seq_31"},
    )

    observed = set(trajectory_df["sequence"].astype(str))
    assert "seq_23" in observed
    assert "seq_31" in observed


def test_build_trajectory_points_requires_chain_column() -> None:
    sequences_df = pd.DataFrame(
        {
            "slot_id": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "score_lexA": [0.1, 0.2, 0.8, 0.9],
            "score_cpxR": [0.2, 0.3, 0.7, 0.8],
        }
    )

    with pytest.raises(ValueError, match="missing required column 'chain'"):
        build_trajectory_points(
            sequences_df,
            ["lexA", "cpxR"],
            max_points=100,
            beta_ladder=[0.2, 1.0],
        )


def test_build_trajectory_points_carries_chain_fields() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 1, 0, 1],
            "draw": [0, 0, 1, 1],
            "sweep_idx": [0, 0, 1, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "beta": [1.0, 0.5, 1.0, 0.5],
            "score_lexA": [0.1, 0.4, 0.2, 0.5],
            "score_cpxR": [0.2, 0.3, 0.3, 0.4],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=100,
        beta_ladder=[1.0, 0.5],
    )

    for required in ("chain", "beta", "sweep", "phase"):
        assert required in trajectory_df.columns
    assert sorted(trajectory_df["chain"].astype(int).unique()) == [0, 1]


def test_build_chain_trajectory_points_tracks_chain_progression() -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 1, 0, 1],
            "sweep": [0, 0, 1, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "beta": [1.0, 0.5, 1.0, 0.5],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.2, 0.1, 0.4, 0.3],
            "x_metric": ["score_lexA"] * 4,
            "y_metric": ["score_cpxR"] * 4,
            "objective_scalar": [0.1, 0.2, 0.3, 0.4],
            "raw_llr_objective": [1.1, 1.2, 1.3, 1.4],
            "norm_llr_objective": [0.2, 0.3, 0.4, 0.5],
        }
    )

    chains_df = build_chain_trajectory_points(trajectory_df, max_points=100)
    chain_zero = chains_df[chains_df["chain"].astype(int) == 0].sort_values("sweep_idx")

    assert list(chain_zero["sweep_idx"].astype(int)) == [0, 1]
    assert sorted(chains_df["chain"].astype(int).unique()) == [0, 1]
    assert "x_tf" in chains_df.columns
    assert "y_tf" in chains_df.columns


def test_build_chain_trajectory_points_subsample_preserves_chain_objective_maximum() -> None:
    chain = list(range(20))
    chain0_objective = [0.1] * 20
    chain0_objective[3] = 10.0
    chain1_objective = [0.2] * 20
    chain1_objective[7] = 7.0
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 20 + [1] * 20,
            "sweep": chain + chain,
            "phase": ["draw"] * 40,
            "beta": [1.0] * 20 + [0.5] * 20,
            "x": [float(v) for v in chain] + [float(v) for v in chain],
            "y": [float(v) for v in chain] + [float(v) for v in chain],
            "x_metric": ["score_lexA"] * 40,
            "y_metric": ["score_cpxR"] * 40,
            "objective_scalar": chain0_objective + chain1_objective,
            "raw_llr_objective": chain0_objective + chain1_objective,
            "norm_llr_objective": chain0_objective + chain1_objective,
        }
    )

    chains_df = build_chain_trajectory_points(trajectory_df, max_points=6)
    by_chain = chains_df.groupby("chain")["objective_scalar"].max().to_dict()
    assert by_chain[0] == pytest.approx(10.0)
    assert by_chain[1] == pytest.approx(7.0)


def test_build_chain_trajectory_points_subsample_retains_requested_sequences() -> None:
    chain = list(range(30))
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 30,
            "sweep": chain,
            "phase": ["draw"] * 30,
            "beta": [1.0] * 30,
            "x": [float(v) for v in chain],
            "y": [float(v) for v in chain],
            "x_metric": ["score_lexA"] * 30,
            "y_metric": ["score_cpxR"] * 30,
            "objective_scalar": [float(v) for v in chain],
            "raw_llr_objective": [float(v) for v in chain],
            "norm_llr_objective": [float(v) for v in chain],
            "sequence": [f"seq_{idx}" for idx in chain],
        }
    )

    chains_df = build_chain_trajectory_points(
        trajectory_df,
        max_points=7,
        retain_sequences={"seq_11", "seq_27"},
    )

    observed = set(chains_df["sequence"].astype(str))
    assert "seq_11" in observed
    assert "seq_27" in observed


def test_add_raw_llr_objective_adds_combined_raw_llr_column() -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence": ["AACCAA", "TTGGTT"],
            "chain": [0, 1],
            "sweep_idx": [0, 0],
        }
    )
    pwms = {
        "lexA": PWM(
            name="lexA",
            matrix=np.asarray(
                [
                    [0.90, 0.03, 0.03, 0.04],
                    [0.04, 0.90, 0.03, 0.03],
                ],
                dtype=float,
            ),
        ),
        "cpxR": PWM(
            name="cpxR",
            matrix=np.asarray(
                [
                    [0.03, 0.03, 0.90, 0.04],
                    [0.03, 0.04, 0.03, 0.90],
                ],
                dtype=float,
            ),
        ),
    }

    enriched = add_raw_llr_objective(
        trajectory_df,
        ["lexA", "cpxR"],
        pwms=pwms,
        objective_config={"combine": "min", "softmin": {"enabled": False}},
        bidirectional=True,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
    )

    assert "raw_llr_lexA" in enriched.columns
    assert "raw_llr_cpxR" in enriched.columns
    assert "norm_llr_lexA" in enriched.columns
    assert "norm_llr_cpxR" in enriched.columns
    assert "raw_llr_objective" in enriched.columns
    assert "norm_llr_objective" in enriched.columns
    assert enriched["raw_llr_objective"].notna().all()
    assert enriched["norm_llr_objective"].notna().all()


def test_add_raw_llr_objective_deduplicates_sequence_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence": ["AACCAA", "AACCAA", "TTGGTT", "TTGGTT"],
            "chain": [0, 1, 0, 1],
            "sweep_idx": [0, 0, 1, 1],
        }
    )
    pwms = {
        "lexA": PWM(name="lexA", matrix=np.asarray([[0.90, 0.03, 0.03, 0.04]], dtype=float)),
        "cpxR": PWM(name="cpxR", matrix=np.asarray([[0.03, 0.03, 0.90, 0.04]], dtype=float)),
    }
    calls: list[tuple[str, tuple[int, ...]]] = []
    scaled_calls: list[tuple[str, tuple[int, ...]]] = []

    class FakeScorer:
        def __init__(self, _pwms, *, bidirectional, scale, pseudocounts, log_odds_clip):
            del bidirectional, pseudocounts, log_odds_clip
            self._scale = str(scale)

        def compute_all_per_pwm(self, seq_arr: np.ndarray, _seq_len: int) -> dict[str, float]:
            key = tuple(int(v) for v in np.asarray(seq_arr, dtype=int))
            calls.append((self._scale, key))
            base = float(sum(key))
            if self._scale != "llr":
                raise AssertionError("compute_all_per_pwm should only be called for raw-llr scoring.")
            return {"lexA": base + 1.0, "cpxR": base + 2.0}

        def scaled_from_raw_llr(self, raw_llr_by_tf: dict[str, float], _seq_len: int) -> dict[str, float]:
            key = tuple(sorted((str(tf), float(val)) for tf, val in raw_llr_by_tf.items()))
            scaled_calls.append((self._scale, key))
            if self._scale != "normalized-llr":
                raise AssertionError("scaled_from_raw_llr should only be called for normalized-llr scoring.")
            return {tf: float(val) / 10.0 for tf, val in raw_llr_by_tf.items()}

    monkeypatch.setattr("dnadesign.cruncher.analysis.trajectory.Scorer", FakeScorer)

    enriched = add_raw_llr_objective(
        trajectory_df,
        ["lexA", "cpxR"],
        pwms=pwms,
        objective_config={"combine": "min", "softmin": {"enabled": False}},
        bidirectional=True,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
    )

    assert len(calls) == 2
    assert len(scaled_calls) == 2
    assert calls.count(("llr", calls[0][1])) == 1
    assert float(enriched.loc[0, "raw_llr_lexA"]) == float(enriched.loc[1, "raw_llr_lexA"])
    assert float(enriched.loc[2, "norm_llr_cpxR"]) == float(enriched.loc[3, "norm_llr_cpxR"])


def test_add_raw_llr_objective_reuses_shared_score_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence": ["AACCAA", "AACCAA", "TTGGTT", "TTGGTT"],
            "chain": [0, 1, 0, 1],
            "sweep_idx": [0, 0, 1, 1],
        }
    )
    pwms = {
        "lexA": PWM(name="lexA", matrix=np.asarray([[0.90, 0.03, 0.03, 0.04]], dtype=float)),
        "cpxR": PWM(name="cpxR", matrix=np.asarray([[0.03, 0.03, 0.90, 0.04]], dtype=float)),
    }
    calls: list[tuple[str, tuple[int, ...]]] = []
    scaled_calls: list[tuple[str, tuple[int, ...]]] = []

    class FakeScorer:
        def __init__(self, _pwms, *, bidirectional, scale, pseudocounts, log_odds_clip):
            del bidirectional, pseudocounts, log_odds_clip
            self._scale = str(scale)

        def compute_all_per_pwm(self, seq_arr: np.ndarray, _seq_len: int) -> dict[str, float]:
            key = tuple(int(v) for v in np.asarray(seq_arr, dtype=int))
            calls.append((self._scale, key))
            base = float(sum(key))
            if self._scale != "llr":
                raise AssertionError("compute_all_per_pwm should only be called for raw-llr scoring.")
            return {"lexA": base + 1.0, "cpxR": base + 2.0}

        def scaled_from_raw_llr(self, raw_llr_by_tf: dict[str, float], _seq_len: int) -> dict[str, float]:
            key = tuple(sorted((str(tf), float(val)) for tf, val in raw_llr_by_tf.items()))
            scaled_calls.append((self._scale, key))
            if self._scale != "normalized-llr":
                raise AssertionError("scaled_from_raw_llr should only be called for normalized-llr scoring.")
            return {tf: float(val) / 10.0 for tf, val in raw_llr_by_tf.items()}

    monkeypatch.setattr("dnadesign.cruncher.analysis.trajectory.Scorer", FakeScorer)

    score_cache: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    add_raw_llr_objective(
        trajectory_df,
        ["lexA", "cpxR"],
        pwms=pwms,
        objective_config={"combine": "min", "softmin": {"enabled": False}},
        bidirectional=True,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
        score_cache=score_cache,
    )
    add_raw_llr_objective(
        trajectory_df,
        ["lexA", "cpxR"],
        pwms=pwms,
        objective_config={"combine": "min", "softmin": {"enabled": False}},
        bidirectional=True,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
        score_cache=score_cache,
    )

    assert len(calls) == 2
    assert len(scaled_calls) == 2


def test_add_raw_llr_objective_requires_both_custom_scorers() -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence": ["AACCAA"],
            "chain": [0],
            "sweep_idx": [0],
        }
    )
    pwms = {
        "lexA": PWM(name="lexA", matrix=np.asarray([[0.90, 0.03, 0.03, 0.04]], dtype=float)),
        "cpxR": PWM(name="cpxR", matrix=np.asarray([[0.03, 0.03, 0.90, 0.04]], dtype=float)),
    }
    with pytest.raises(ValueError, match="both scorer_raw and scorer_norm"):
        add_raw_llr_objective(
            trajectory_df,
            ["lexA", "cpxR"],
            pwms=pwms,
            objective_config={"combine": "min", "softmin": {"enabled": False}},
            bidirectional=True,
            pwm_pseudocounts=0.10,
            log_odds_clip=None,
            scorer_raw=object(),
        )


def test_build_trajectory_points_replays_softmin_schedule_per_sweep() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 9],
            "phase": ["draw", "draw"],
            "score_lexA": [0.4, 0.4],
            "score_cpxR": [0.8, 0.8],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=0,
        objective_config={
            "combine": "min",
            "total_sweeps": 10,
            "softmin": {
                "enabled": True,
                "schedule": "linear",
                "beta_start": 0.5,
                "beta_end": 5.0,
            },
        },
    )

    assert len(trajectory_df) == 2
    first = float(trajectory_df.iloc[0]["objective_scalar"])
    last = float(trajectory_df.iloc[1]["objective_scalar"])
    assert last > first


def test_build_trajectory_points_softmin_requires_total_sweeps() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0],
            "sweep_idx": [0],
            "phase": ["draw"],
            "score_lexA": [0.4],
            "score_cpxR": [0.8],
        }
    )

    with pytest.raises(ValueError, match="objective.total_sweeps"):
        build_trajectory_points(
            sequences_df,
            ["lexA", "cpxR"],
            max_points=0,
            objective_config={
                "combine": "min",
                "softmin": {
                    "enabled": True,
                    "schedule": "linear",
                    "beta_start": 0.5,
                    "beta_end": 5.0,
                },
            },
        )


def test_build_trajectory_points_rejects_non_finite_score_values() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
            "phase": ["draw", "draw"],
            "score_lexA": [0.4, float("nan")],
            "score_cpxR": [0.8, 0.7],
        }
    )

    with pytest.raises(ValueError, match="non-finite values"):
        build_trajectory_points(
            sequences_df,
            ["lexA", "cpxR"],
            max_points=0,
            objective_config={
                "combine": "min",
                "softmin": {"enabled": False},
            },
        )
