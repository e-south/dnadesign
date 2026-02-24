"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_consensus_anchors.py

Validate consensus-anchor contracts for score-space maxima references.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.analysis.consensus import compute_consensus_anchors
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def _encode(seq: str) -> np.ndarray:
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.asarray([lut[ch] for ch in seq], dtype=np.int8)


def _toy_pwms() -> dict[str, PWM]:
    return {
        "tfA": PWM(
            name="tfA",
            matrix=np.asarray(
                [
                    [0.90, 0.05, 0.03, 0.02],
                    [0.05, 0.90, 0.03, 0.02],
                ],
                dtype=float,
            ),
        ),
        "tfB": PWM(
            name="tfB",
            matrix=np.asarray(
                [
                    [0.05, 0.03, 0.90, 0.02],
                    [0.05, 0.03, 0.02, 0.90],
                ],
                dtype=float,
            ),
        ),
        "tfC": PWM(
            name="tfC",
            matrix=np.asarray(
                [
                    [0.02, 0.90, 0.05, 0.03],
                    [0.03, 0.02, 0.90, 0.05],
                ],
                dtype=float,
            ),
        ),
    }


def test_consensus_anchors_use_tf_consensus_sequences_for_pair_axes() -> None:
    pwms = _toy_pwms()
    anchors = compute_consensus_anchors(
        pwms=pwms,
        tf_names=["tfA", "tfB"],
        sequence_length=2,
        objective_config={"score_scale": "llr", "bidirectional": False},
        x_metric="score_tfA",
        y_metric="score_tfB",
    )

    assert len(anchors) == 2
    anchors_by_tf = {str(anchor["tf"]): anchor for anchor in anchors}
    assert set(anchors_by_tf) == {"tfA", "tfB"}
    assert anchors_by_tf["tfA"]["label"] == "tfA consensus"
    assert anchors_by_tf["tfB"]["label"] == "tfB consensus"

    scorer = Scorer(pwms, scale="llr", bidirectional=False, pseudocounts=0.10)
    for tf_name in ("tfA", "tfB"):
        consensus_seq = scorer.consensus_sequence(tf_name)
        per_tf = scorer.compute_all_per_pwm(_encode(consensus_seq), int(len(consensus_seq)))
        anchor = anchors_by_tf[tf_name]
        assert str(anchor["sequence"]) == consensus_seq
        assert float(anchor["x"]) == float(per_tf["tfA"])
        assert float(anchor["y"]) == float(per_tf["tfB"])


def test_consensus_anchors_project_worst_second_from_each_tf_consensus() -> None:
    pwms = _toy_pwms()
    anchors = compute_consensus_anchors(
        pwms=pwms,
        tf_names=["tfA", "tfB", "tfC"],
        sequence_length=2,
        objective_config={"score_scale": "llr", "bidirectional": False},
        x_metric="worst_tf_score",
        y_metric="second_worst_tf_score",
    )

    assert len(anchors) == 3
    anchors_by_tf = {str(anchor["tf"]): anchor for anchor in anchors}
    assert set(anchors_by_tf) == {"tfA", "tfB", "tfC"}
    assert all(str(anchor["label"]).endswith("consensus") for anchor in anchors)

    scorer = Scorer(pwms, scale="llr", bidirectional=False, pseudocounts=0.10)
    for tf_name, anchor in anchors_by_tf.items():
        consensus_seq = scorer.consensus_sequence(tf_name)
        per_tf = scorer.compute_all_per_pwm(_encode(consensus_seq), int(len(consensus_seq)))
        ordered = sorted(float(value) for value in per_tf.values())
        assert float(anchor["x"]) == ordered[0]
        assert float(anchor["y"]) == ordered[1]
