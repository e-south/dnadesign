"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_incremental_scoring.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.optimizers.pt import PTGibbsOptimizer
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import LocalScanCache, Scorer
from dnadesign.cruncher.core.state import SequenceState


def _random_pwm(rng: np.random.Generator, name: str, width: int = 6) -> PWM:
    mat = rng.random((width, 4))
    mat = mat / mat.sum(axis=1, keepdims=True)
    return PWM(name=name, matrix=mat)


def _brute_best_llr(seq: np.ndarray, lom: np.ndarray, bidirectional: bool) -> float:
    w = lom.shape[0]
    L = seq.size
    if L < w:
        return float("-inf")

    def _scan(arr: np.ndarray) -> float:
        windows = np.lib.stride_tricks.sliding_window_view(arr, w)
        llrs = lom[np.arange(w)[:, None], windows.T].sum(axis=0)
        return float(np.max(llrs))

    best = _scan(seq)
    if bidirectional:
        rev = (3 - seq)[::-1]
        best = max(best, _scan(rev))
    return best


def test_compute_all_per_pwm_matches_bruteforce_llr() -> None:
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=12, dtype=np.int8)
    pwms = {"tfA": _random_pwm(rng, "tfA"), "tfB": _random_pwm(rng, "tfB")}

    for bidirectional in (False, True):
        scorer = Scorer(pwms, scale="llr", bidirectional=bidirectional)
        per_tf = scorer.compute_all_per_pwm(seq, seq.size)
        for tf, pwm in pwms.items():
            lom = scorer._info(tf).lom
            expected = _brute_best_llr(seq, lom, bidirectional)
            assert np.isclose(per_tf[tf], expected, atol=1e-9)


def test_fast_s_move_matches_bruteforce() -> None:
    rng = np.random.default_rng(1)
    seq = rng.integers(0, 4, size=15, dtype=np.int8)
    pwms = {"tfA": _random_pwm(rng, "tfA"), "tfB": _random_pwm(rng, "tfB")}
    scorer = Scorer(pwms, scale="normalized-llr", bidirectional=True)
    evaluator = SequenceEvaluator(pwms, scale="normalized-llr", scorer=scorer, bidirectional=True)
    state = SequenceState(seq)
    cache = LocalScanCache(scorer, seq)

    pos = 4
    old_base = int(seq[pos])
    raw_candidates = cache.candidate_raw_llr_maps(pos, old_base)
    fast_per_tf = [scorer.scaled_from_raw_llr(raw, seq.size) for raw in raw_candidates]
    beta_softmin = 0.7
    fast_combined = [
        evaluator.combined_from_scores(per_tf, beta=beta_softmin, length=seq.size) for per_tf in fast_per_tf
    ]

    brute_per_tf = []
    brute_combined = []
    for b in range(4):
        seq[pos] = b
        per_tf_b, comb_b = evaluator.evaluate(state, beta=beta_softmin, length=seq.size)
        brute_per_tf.append(per_tf_b)
        brute_combined.append(comb_b)
    seq[pos] = old_base

    for b in range(4):
        for tf in fast_per_tf[b]:
            assert np.isclose(fast_per_tf[b][tf], brute_per_tf[b][tf], atol=1e-9)
        assert np.isclose(fast_combined[b], brute_combined[b], atol=1e-9)

    beta = 2.5
    lods_fast = beta * np.asarray(fast_combined, dtype=float)
    lods_fast -= lods_fast.max()
    probs_fast = np.exp(lods_fast) / np.exp(lods_fast).sum()

    lods_brute = beta * np.asarray(brute_combined, dtype=float)
    lods_brute -= lods_brute.max()
    probs_brute = np.exp(lods_brute) / np.exp(lods_brute).sum()

    rng_fast = np.random.default_rng(42)
    rng_brute = np.random.default_rng(42)
    new_fast = int(rng_fast.choice(4, p=probs_fast))
    new_brute = int(rng_brute.choice(4, p=probs_brute))
    assert new_fast == new_brute


def test_pt_beta_ladder_anchor_scale() -> None:
    class _Eval:
        scorer = SimpleNamespace(scale="llr")

        def __call__(self, state: SequenceState) -> dict[str, float]:
            return {"tf": float(state.seq.sum())}

        def combined_from_scores(self, per_tf, beta=None, length=None):
            return float(next(iter(per_tf.values())))

        def evaluate(self, state: SequenceState, beta=None, length=None):
            return {"tf": float(state.seq.sum())}, float(state.seq.sum())

    cfg = {
        "draws": 1,
        "tune": 0,
        "chains": 4,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 0.0,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "kind": "geometric",
        "beta": [0.2, 0.4, 0.8, 1.6],
        "softmin": {"enabled": False},
        "adaptive_swap": {"enabled": False},
    }

    opt = PTGibbsOptimizer(
        evaluator=_Eval(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None),
    )
    ladder = opt._scaled_beta_ladder(10.0)
    assert np.isclose(ladder[0], 0.2)
    assert np.isclose(ladder[-1], 16.0)


def test_pt_stats_report_effective_beta_ladder() -> None:
    class _Eval:
        scorer = SimpleNamespace(scale="llr")

        def __call__(self, state: SequenceState) -> dict[str, float]:
            return {"tf": float(state.seq.sum())}

        def combined_from_scores(self, per_tf, beta=None, length=None):
            return float(next(iter(per_tf.values())))

        def evaluate(self, state: SequenceState, beta=None, length=None):
            return {"tf": float(state.seq.sum())}, float(state.seq.sum())

    cfg = {
        "draws": 1,
        "tune": 0,
        "chains": 4,
        "min_dist": 0,
        "top_k": 1,
        "sequence_length": 4,
        "swap_prob": 0.0,
        "record_tune": False,
        "progress_bar": False,
        "progress_every": 0,
        "early_stop": {},
        "block_len_range": (1, 1),
        "multi_k_range": (1, 1),
        "slide_max_shift": 1,
        "swap_len_range": (1, 1),
        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0},
        "kind": "geometric",
        "beta": [0.2, 0.4, 0.8, 1.6],
        "softmin": {"enabled": False},
        "adaptive_swap": {"enabled": False},
    }

    opt = PTGibbsOptimizer(
        evaluator=_Eval(),
        cfg=cfg,
        rng=np.random.default_rng(0),
        pwms={},
        init_cfg=SimpleNamespace(kind="random", length=4, pad_with="background", regulator=None),
    )
    opt.beta_ladder = opt._scaled_beta_ladder(5.0)
    stats = opt.stats()
    assert stats["beta_ladder_final"] == opt.beta_ladder
    assert stats["beta_max_final"] == max(opt.beta_ladder)
    assert stats["final_mcmc_beta"] == max(opt.beta_ladder)
