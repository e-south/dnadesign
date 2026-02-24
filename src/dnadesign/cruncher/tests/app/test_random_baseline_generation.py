"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_random_baseline_generation.py

Contracts for random-baseline artifact generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dnadesign.cruncher.app.sample.run_set_execution import _write_random_baseline_artifacts
from dnadesign.cruncher.artifacts.layout import random_baseline_hits_path, random_baseline_path


class _DummyPWM:
    def __init__(self, length: int) -> None:
        self.length = int(length)


class _SinglePassScorer:
    tf_names = ["tfA", "tfB"]

    def __init__(self) -> None:
        self.calls = 0

    def compute_all_per_pwm_and_hits(self, seq_arr, seq_length: int):
        self.calls += 1
        per_tf = {"tfA": 0.8, "tfB": 0.6}
        hits = {
            "tfA": {
                "best_start": 0,
                "strand": "+",
                "best_window_seq": "AAAA",
                "best_core_seq": "AAAA",
                "best_score_raw": 1.0,
                "best_hit_tiebreak": "max_forward",
                "width": 4,
            },
            "tfB": {
                "best_start": 1,
                "strand": "-",
                "best_window_seq": "CCCC",
                "best_core_seq": "GGGG",
                "best_score_raw": 0.7,
                "best_hit_tiebreak": "max_reverse",
                "width": 4,
            },
        }
        return per_tf, hits

    def compute_all_per_pwm(self, seq_arr, seq_length: int):
        raise AssertionError("random baseline path must use single-pass scorer API")

    def best_hit(self, seq_arr, tf: str):
        raise AssertionError("random baseline path must not rescan per TF best-hit")


def test_random_baseline_generation_uses_single_pass_scoring_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True)
    scorer = _SinglePassScorer()
    sample_cfg = SimpleNamespace(
        seed=7,
        sequence_length=12,
        objective=SimpleNamespace(bidirectional=True, score_scale="normalized-llr"),
        output=SimpleNamespace(save_random_baseline=True, random_baseline_n=3),
    )
    tfs = ["tfA", "tfB"]
    pwms = {"tfA": _DummyPWM(4), "tfB": _DummyPWM(4)}
    pwm_ref_by_tf = {"tfA": "src:A", "tfB": "src:B"}
    pwm_hash_by_tf = {"tfA": "ha", "tfB": "hb"}
    core_def_by_tf = {"tfA": "ca", "tfB": "cb"}
    artifacts: list[dict[str, object]] = []

    _write_random_baseline_artifacts(
        out_dir=out_dir,
        sample_cfg=sample_cfg,
        set_index=1,
        tfs=tfs,
        scorer=scorer,
        pwms=pwms,
        pwm_ref_by_tf=pwm_ref_by_tf,
        pwm_hash_by_tf=pwm_hash_by_tf,
        core_def_by_tf=core_def_by_tf,
        stage="sample",
        artifacts=artifacts,
    )

    baseline_df = pd.read_parquet(random_baseline_path(out_dir))
    baseline_hits_df = pd.read_parquet(random_baseline_hits_path(out_dir))
    assert len(baseline_df) == 3
    assert len(baseline_hits_df) == 6
    assert scorer.calls == 3
