"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_selection.py

Tests elite-selection helper behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.app.sample.diagnostics import _EliteCandidate, dsdna_equivalence_enabled, resolve_dsdna_mode
from dnadesign.cruncher.app.sample.elites_mmr import (
    build_elite_pool,
    hydrate_candidate_hits,
    select_elites_mmr,
)
from dnadesign.cruncher.app.sample.elites_stage import (
    _candidate_payload,
    _hits_match_polish_contract,
    _postprocess_elite_candidates,
    _remaining_single_owner_polish_improvements,
)
from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def _seq_arr(seq: str) -> np.ndarray:
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.asarray([table[ch] for ch in seq], dtype=np.int8)


def _make_pwm(name: str, consensus: str) -> PWM:
    rows: list[list[float]] = []
    for base in consensus:
        row = [0.01, 0.01, 0.01, 0.01]
        row["ACGT".index(base)] = 0.97
        rows.append(row)
    return PWM(name=name, matrix=np.asarray(rows, dtype=float))


def _elite_candidate(seq: str, scorer: Scorer, *, chain_id: int = 0, draw_idx: int = 1) -> _EliteCandidate:
    arr = _seq_arr(seq)
    per_tf_map, hits = scorer.compute_all_per_pwm_and_hits(arr, arr.size)
    norm_map = scorer.normalized_llr_map(arr)
    return _EliteCandidate(
        seq_arr=arr,
        chain_id=chain_id,
        draw_idx=draw_idx,
        combined_score=float(min(per_tf_map.values())),
        min_norm=float(min(norm_map.values())),
        sum_norm=float(sum(norm_map.values())),
        per_tf_map=per_tf_map,
        norm_map=norm_map,
        per_tf_hits=hits,
    )


def _seq_str(arr: np.ndarray) -> str:
    return "".join("ACGT"[int(v)] for v in arr.tolist())


def test_resolve_dsdna_mode_tracks_bidirectional_flag() -> None:
    assert resolve_dsdna_mode(elites_cfg=object(), bidirectional=True) is True
    assert resolve_dsdna_mode(elites_cfg=object(), bidirectional=False) is False


def test_dsdna_equivalence_enabled_uses_sample_objective_flag() -> None:
    cfg = SampleConfig(
        seed=7,
        sequence_length=6,
        budget={"tune": 0, "draws": 1},
        objective={"bidirectional": True, "score_scale": "normalized-llr"},
    )
    assert dsdna_equivalence_enabled(cfg) is True


def test_select_elites_mmr_with_zero_diversity_uses_score_only_selection() -> None:
    raw_elites = [
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 0, 0], dtype=np.int8),
            chain_id=0,
            draw_idx=1,
            combined_score=0.90,
            min_norm=0.20,
            sum_norm=0.20,
            per_tf_map={"tf": 0.20},
            norm_map={"tf": 0.20},
            per_tf_hits={},
        ),
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 0, 1], dtype=np.int8),
            chain_id=0,
            draw_idx=2,
            combined_score=0.85,
            min_norm=0.95,
            sum_norm=0.95,
            per_tf_map={"tf": 0.95},
            norm_map={"tf": 0.95},
            per_tf_hits={},
        ),
        _EliteCandidate(
            seq_arr=np.asarray([0, 0, 1, 1], dtype=np.int8),
            chain_id=0,
            draw_idx=3,
            combined_score=0.80,
            min_norm=0.99,
            sum_norm=0.99,
            per_tf_map={"tf": 0.99},
            norm_map={"tf": 0.99},
            per_tf_hits={},
        ),
    ]

    result = select_elites_mmr(
        raw_elites=raw_elites,
        elite_k=2,
        pool_size=3,
        scorer=object(),
        pwms={},
        dsdna_mode=False,
        diversity=0.0,
        sample_sequence_length=4,
        cooling_config={},
    )

    assert [f"{row.chain_id}:{row.draw_idx}" for row in result.kept_elites] == ["0:1", "0:2"]
    assert result.mmr_summary is not None
    assert result.mmr_summary["selection_policy"] == "score_topk"
    assert result.mmr_summary["score_weight"] == pytest.approx(1.0)
    assert result.mmr_summary["diversity_weight"] == pytest.approx(0.0)


def test_build_elite_pool_does_not_eagerly_compute_tf_hits() -> None:
    class _DummyOptimizer:
        all_meta = [(0, 1), (1, 2)]
        all_samples = [
            np.asarray([0, 1, 2, 3], dtype=np.int8),
            np.asarray([3, 2, 1, 0], dtype=np.int8),
        ]
        all_scores = [
            {"lexA": 0.5, "cpxR": 0.3},
            {"lexA": 0.4, "cpxR": 0.2},
        ]

    class _DummyEvaluator:
        @staticmethod
        def combined_from_scores(per_tf_map, beta, length):
            _ = (beta, length)
            return float(min(per_tf_map.values()))

    class _DummyScorer:
        tf_names = ["lexA", "cpxR"]

        def __init__(self) -> None:
            self.best_hit_calls = 0

        def best_hit(self, seq_arr, tf_name):
            _ = (seq_arr, tf_name)
            self.best_hit_calls += 1
            return {
                "offset": 0,
                "width": 1,
                "strand": "+",
                "best_start": 0,
                "best_window_seq": "A",
                "best_core_seq": "A",
                "best_score_raw": 0.0,
                "best_hit_tiebreak": "leftmost",
            }

    scorer = _DummyScorer()
    sample_cfg = SampleConfig(
        seed=11,
        sequence_length=4,
        budget={"tune": 0, "draws": 2},
        objective={"bidirectional": True, "score_scale": "normalized-llr"},
    )

    result = build_elite_pool(
        optimizer=_DummyOptimizer(),
        evaluator=_DummyEvaluator(),
        scorer=scorer,
        sample_cfg=sample_cfg,
        beta_softmin_final=1.0,
    )

    assert scorer.best_hit_calls == 0
    assert len(result.raw_elites) == 2
    assert all(candidate.per_tf_hits is None for candidate in result.raw_elites)


def test_hydrate_candidate_hits_populates_missing_hits_once() -> None:
    class _DummyScorer:
        tf_names = ["lexA", "cpxR"]

        def __init__(self) -> None:
            self.best_hit_calls = 0

        def best_hit(self, seq_arr, tf_name):
            _ = seq_arr
            self.best_hit_calls += 1
            return {
                "offset": 0,
                "width": 1,
                "strand": "+",
                "best_start": 0,
                "best_window_seq": "A",
                "best_core_seq": "A",
                "best_score_raw": 0.0,
                "best_hit_tiebreak": f"tf={tf_name}",
            }

    candidates = [
        _EliteCandidate(
            seq_arr=np.asarray([0, 1, 2, 3], dtype=np.int8),
            chain_id=0,
            draw_idx=1,
            combined_score=0.5,
            min_norm=0.2,
            sum_norm=0.7,
            per_tf_map={"lexA": 0.5, "cpxR": 0.2},
            norm_map={"lexA": 0.5, "cpxR": 0.2},
            per_tf_hits=None,
        )
    ]
    scorer = _DummyScorer()

    hydrate_candidate_hits(candidates, scorer=scorer)
    assert scorer.best_hit_calls == 2
    assert candidates[0].per_tf_hits is not None
    assert sorted(candidates[0].per_tf_hits.keys()) == ["cpxR", "lexA"]

    hydrate_candidate_hits(candidates, scorer=scorer)
    assert scorer.best_hit_calls == 2


def test_postprocess_polishes_single_owner_positions_without_moving_hits() -> None:
    scorer = Scorer(
        {
            "tfA": _make_pwm("tfA", "ATGC"),
            "tfB": _make_pwm("tfB", "CGTA"),
        },
        bidirectional=False,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("ATGTCGTT", scorer)

    processed, stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert stats["polish_edits"] >= 2
    assert stats["trim_left"] == 0
    assert stats["trim_right"] == 0
    assert len(processed) == 1
    assert _seq_str(processed[0].seq_arr) == "ATGCCGTA"
    assert int(processed[0].per_tf_hits["tfA"]["best_start"]) == 0
    assert int(processed[0].per_tf_hits["tfB"]["best_start"]) == 4
    for tf_name in ("tfA", "tfB"):
        hit = processed[0].per_tf_hits[tf_name]
        assert isinstance(hit.get("best_score_scaled"), float)
        assert isinstance(hit.get("best_score_norm"), float)


def test_postprocess_trims_only_uncovered_prefix_suffix_offsets() -> None:
    scorer = Scorer(
        {
            "tfA": _make_pwm("tfA", "ATGC"),
            "tfB": _make_pwm("tfB", "CGTA"),
        },
        bidirectional=False,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("TATGTCGTAT", scorer)

    processed, stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert stats["trim_left"] == 1
    assert stats["trim_right"] == 1
    assert len(processed) == 1
    assert _seq_str(processed[0].seq_arr) == "ATGCCGTA"
    assert int(processed[0].per_tf_hits["tfA"]["best_start"]) == 0
    assert int(processed[0].per_tf_hits["tfB"]["best_start"]) == 4


def test_postprocess_trims_edges_per_elite_not_global_minimum() -> None:
    scorer = Scorer(
        {
            "tfA": _make_pwm("tfA", "ATGC"),
            "tfB": _make_pwm("tfB", "CGTA"),
        },
        bidirectional=False,
        scale="normalized-llr",
    )
    # First elite has uncovered suffix, second has uncovered prefix.
    # Per-elite trimming should trim both; global-min trimming trims neither.
    cand_a = _elite_candidate("ATGCCGTAT", scorer, chain_id=0, draw_idx=1)
    cand_b = _elite_candidate("TATGCCGTA", scorer, chain_id=1, draw_idx=2)

    processed, stats = _postprocess_elite_candidates(
        candidates=[cand_a, cand_b],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert stats["trim_left"] == 1
    assert stats["trim_right"] == 1
    assert stats["dedup_dropped"] == 1
    assert len(processed) == 1
    assert _seq_str(processed[0].seq_arr) == "ATGCCGTA"


def test_postprocess_dedup_can_reduce_elite_count() -> None:
    scorer = Scorer(
        {
            "tfA": _make_pwm("tfA", "ATGC"),
            "tfB": _make_pwm("tfB", "CGTA"),
        },
        bidirectional=False,
        scale="normalized-llr",
    )
    cand_a = _elite_candidate("ATGTCGTT", scorer, chain_id=0, draw_idx=1)
    cand_b = _elite_candidate("ATGCCGTT", scorer, chain_id=1, draw_idx=2)

    processed, stats = _postprocess_elite_candidates(
        candidates=[cand_a, cand_b],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert len(processed) == 1
    assert stats["dedup_dropped"] == 1
    assert _seq_str(processed[0].seq_arr) == "ATGCCGTA"


def test_postprocess_preserves_hit_width_contract_for_bidirectional_mixed_strands() -> None:
    scorer = Scorer(
        {
            "tfA": _make_pwm("tfA", "ATGC"),
            "tfB": _make_pwm("tfB", "CGTA"),
        },
        bidirectional=True,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("AAAAAAAG", scorer)
    initial_widths = {tf_name: int(details["width"]) for tf_name, details in candidate.per_tf_hits.items()}
    seq_len = int(candidate.seq_arr.size)

    processed, _stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert len(processed) == 1
    for tf_name, width in initial_widths.items():
        hit = processed[0].per_tf_hits[tf_name]
        assert int(hit["width"]) == width
        start = int(hit["best_start"])
        assert start >= 0
        assert start + width <= seq_len
        assert str(hit["strand"]) in {"+", "-"}


def test_postprocess_polishes_using_raw_llr_when_scaled_value_is_saturated() -> None:
    rows: list[list[float]] = []
    for _ in range(20):
        row = [1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10]
        row[0] = 1.0 - 3.0e-10
        rows.append(row)
    scorer = Scorer(
        {"tfA": PWM(name="tfA", matrix=np.asarray(rows, dtype=float))},
        bidirectional=False,
        scale="logp",
    )
    candidate = _elite_candidate("C" + ("A" * 19), scorer)
    before_scaled = float(candidate.per_tf_map["tfA"])
    before_raw = float(candidate.per_tf_hits["tfA"]["best_score_raw"])

    processed, stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert len(processed) == 1
    assert stats["polish_edits"] >= 1
    assert _seq_str(processed[0].seq_arr) == ("A" * 20)
    assert float(processed[0].per_tf_hits["tfA"]["best_score_raw"]) > before_raw
    assert float(processed[0].per_tf_map["tfA"]) == pytest.approx(before_scaled)


def test_hits_match_polish_contract_rejects_owner_start_or_strand_drift() -> None:
    expected = {
        "tfA": (5, 4, "-"),
        "tfB": (12, 3, "+"),
    }
    per_tf_hits = {
        "tfA": {
            "best_start": 4,
            "width": 4,
            "strand": "+",
        },
        "tfB": {
            "best_start": 12,
            "width": 3,
            "strand": "+",
        },
    }
    assert not _hits_match_polish_contract(
        per_tf_hits=per_tf_hits,
        expected_windows=expected,
        owner_tf="tfA",
        owner_position=6,
    )


def test_hits_match_polish_contract_rejects_owner_move_if_position_leaves_owner_window() -> None:
    expected = {
        "tfA": (5, 4, "-"),
        "tfB": (12, 3, "+"),
    }
    per_tf_hits = {
        "tfA": {
            "best_start": 0,
            "width": 4,
            "strand": "+",
        },
        "tfB": {
            "best_start": 12,
            "width": 3,
            "strand": "+",
        },
    }
    assert not _hits_match_polish_contract(
        per_tf_hits=per_tf_hits,
        expected_windows=expected,
        owner_tf="tfA",
        owner_position=6,
    )


def test_hits_match_polish_contract_rejects_non_owner_window_drift() -> None:
    expected = {
        "tfA": (5, 4, "-"),
        "tfB": (12, 3, "+"),
    }
    per_tf_hits = {
        "tfA": {
            "best_start": 4,
            "width": 4,
            "strand": "+",
        },
        "tfB": {
            "best_start": 11,
            "width": 3,
            "strand": "+",
        },
    }
    assert not _hits_match_polish_contract(
        per_tf_hits=per_tf_hits,
        expected_windows=expected,
        owner_tf="tfA",
        owner_position=6,
    )


def test_postprocess_polishes_reverse_strand_owner_to_consensus_core() -> None:
    rows: list[list[float]] = []
    consensus = "ATTCTGCATAG"
    for base in consensus:
        row = [0.01, 0.01, 0.01, 0.01]
        row["ACGT".index(base)] = 0.97
        rows.append(row)
    scorer = Scorer(
        {"tfA": PWM(name="tfA", matrix=np.asarray(rows, dtype=float))},
        bidirectional=True,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("AAAAAAAAAAAAAA", scorer)
    before_raw = float(candidate.per_tf_hits["tfA"]["best_score_raw"])
    assert str(candidate.per_tf_hits["tfA"]["strand"]) == "-"

    processed, stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert len(processed) == 1
    assert stats["polish_edits"] >= 1
    hit = processed[0].per_tf_hits["tfA"]
    assert str(hit["strand"]) == "-"
    assert str(hit["best_core_seq"]) == consensus
    assert float(hit["best_score_raw"]) > before_raw


def test_postprocess_converges_without_remaining_reverse_single_owner_improvements() -> None:
    rows: list[list[float]] = []
    consensus = "ATTCTGCATAG"
    for base in consensus:
        row = [0.01, 0.01, 0.01, 0.01]
        row["ACGT".index(base)] = 0.97
        rows.append(row)
    scorer = Scorer(
        {"tfA": PWM(name="tfA", matrix=np.asarray(rows, dtype=float))},
        bidirectional=True,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("AAAAAAAAAAAAAA", scorer)

    processed, _stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=False,
    )

    assert len(processed) == 1
    remaining = _remaining_single_owner_polish_improvements(
        seq_arr=np.asarray(processed[0].seq_arr, dtype=np.int8),
        per_tf_hits=processed[0].per_tf_hits,
        scorer=scorer,
    )
    assert remaining == []


def test_postprocess_converges_after_edge_trim_reexposes_single_owner_edits() -> None:
    pwms = {
        "tf0": PWM(
            name="tf0",
            matrix=np.asarray(
                [
                    [0.765833, 0.002346, 0.068300, 0.163520],
                    [0.014267, 0.961906, 0.002811, 0.021016],
                    [0.029166, 0.053261, 0.000002, 0.917572],
                    [0.155809, 0.174179, 0.640179, 0.029833],
                    [0.035700, 0.000002, 0.894036, 0.070263],
                    [0.007087, 0.059334, 0.933049, 0.000530],
                    [0.007785, 0.001857, 0.002240, 0.988118],
                ],
                dtype=float,
            ),
        ),
        "tf1": PWM(
            name="tf1",
            matrix=np.asarray(
                [
                    [0.011720, 0.834544, 0.151271, 0.002465],
                    [0.047050, 0.874755, 0.002439, 0.075756],
                    [0.150964, 0.844694, 0.003927, 0.000416],
                    [0.250183, 0.093424, 0.202288, 0.454105],
                    [0.002846, 0.000091, 0.019627, 0.977436],
                    [0.005294, 0.862562, 0.129876, 0.002268],
                    [0.041832, 0.827451, 0.119161, 0.011555],
                ],
                dtype=float,
            ),
        ),
        "tf2": PWM(
            name="tf2",
            matrix=np.asarray(
                [
                    [0.019983, 0.002283, 0.012203, 0.965531],
                    [0.002179, 0.000512, 0.338616, 0.658693],
                    [0.011476, 0.163230, 0.821147, 0.004147],
                    [0.001099, 0.015054, 0.983845, 0.000002],
                    [0.039227, 0.008440, 0.175476, 0.776856],
                    [0.867125, 0.057768, 0.013870, 0.061236],
                    [0.083110, 0.784033, 0.057082, 0.075775],
                ],
                dtype=float,
            ),
        ),
        "tf3": PWM(
            name="tf3",
            matrix=np.asarray(
                [
                    [0.162227, 0.073212, 0.693156, 0.071405],
                    [0.035862, 0.085275, 0.037986, 0.840877],
                    [0.006744, 0.001492, 0.932227, 0.059537],
                    [0.047220, 0.026836, 0.924591, 0.001353],
                    [0.019446, 0.834570, 0.037580, 0.108405],
                    [0.000186, 0.903698, 0.092536, 0.003580],
                    [0.000739, 0.858907, 0.004799, 0.135555],
                ],
                dtype=float,
            ),
        ),
    }
    scorer = Scorer(
        pwms,
        bidirectional=True,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("TTGGGGGTAAGTGCAATTACG", scorer)

    processed, stats = _postprocess_elite_candidates(
        candidates=[candidate],
        scorer=scorer,
        dsdna_mode=True,
    )

    assert len(processed) == 1
    assert stats["trim_left"] > 0
    assert stats["trim_right"] > 0
    remaining = _remaining_single_owner_polish_improvements(
        seq_arr=np.asarray(processed[0].seq_arr, dtype=np.int8),
        per_tf_hits=processed[0].per_tf_hits,
        scorer=scorer,
    )
    assert remaining == []


def test_hits_match_polish_contract_rejects_reverse_flip_even_when_owner_score_improves() -> None:
    rows: list[list[float]] = []
    for base in "ATGC":
        row = [0.01, 0.01, 0.01, 0.01]
        row["ACGT".index(base)] = 0.97
        rows.append(row)
    scorer = Scorer(
        {"tfA": PWM(name="tfA", matrix=np.asarray(rows, dtype=float))},
        bidirectional=True,
        scale="normalized-llr",
    )
    candidate = _elite_candidate("AAAAAAAA", scorer)
    expected_windows = {"tfA": (0, 4, "+")}

    trial = np.asarray(candidate.seq_arr, dtype=np.int8).copy()
    trial[0] = 2  # A -> G creates a stronger reverse hit at the same start
    _, trial_hits, _, _, _ = _candidate_payload(seq_arr=trial, scorer=scorer)

    assert str(candidate.per_tf_hits["tfA"]["strand"]) == "+"
    assert str(trial_hits["tfA"]["strand"]) == "-"
    assert not _hits_match_polish_contract(
        per_tf_hits=trial_hits,
        expected_windows=expected_windows,
        owner_tf="tfA",
        owner_position=0,
    )
