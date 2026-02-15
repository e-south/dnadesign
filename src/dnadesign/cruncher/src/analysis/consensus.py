"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/consensus.py

Compute consensus-anchor coordinates for trajectory plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def pwm_consensus(pwm: PWM) -> str:
    indices = np.argmax(np.asarray(pwm.matrix, dtype=float), axis=1)
    return "".join("ACGT"[int(idx)] for idx in indices)


def _embed_consensus(consensus: str, sequence_length: int) -> str:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1 for consensus anchors.")
    if not consensus:
        raise ValueError("Consensus sequence must be non-empty.")
    if len(consensus) >= sequence_length:
        start = (len(consensus) - sequence_length) // 2
        return consensus[start : start + sequence_length]
    left = (sequence_length - len(consensus)) // 2
    right = sequence_length - len(consensus) - left
    return ("A" * left) + consensus + ("A" * right)


def _encode_sequence(seq: str) -> np.ndarray:
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    try:
        return np.asarray([lut[ch] for ch in seq], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Consensus anchor sequence contains non-ACGT symbol: {seq!r}") from exc


def _project_from_per_tf(per_tf: dict[str, float], *, x_metric: str, y_metric: str) -> tuple[float, float]:
    ordered = sorted(float(v) for v in per_tf.values())
    if x_metric.startswith("score_"):
        tf = x_metric.removeprefix("score_")
        if tf not in per_tf:
            raise ValueError(f"Missing TF score for x-axis metric '{x_metric}'.")
        x = float(per_tf[tf])
    elif x_metric == "worst_tf_score":
        x = float(ordered[0])
    elif x_metric == "second_worst_tf_score":
        x = float(ordered[1] if len(ordered) > 1 else ordered[0])
    else:
        raise ValueError(f"Unsupported x-axis metric for consensus anchors: {x_metric}")

    if y_metric.startswith("score_"):
        tf = y_metric.removeprefix("score_")
        if tf not in per_tf:
            raise ValueError(f"Missing TF score for y-axis metric '{y_metric}'.")
        y = float(per_tf[tf])
    elif y_metric == "worst_tf_score":
        y = float(ordered[0])
    elif y_metric == "second_worst_tf_score":
        y = float(ordered[1] if len(ordered) > 1 else ordered[0])
    else:
        raise ValueError(f"Unsupported y-axis metric for consensus anchors: {y_metric}")
    return x, y


def compute_consensus_anchors(
    *,
    pwms: dict[str, PWM],
    tf_names: Iterable[str],
    sequence_length: int,
    objective_config: dict[str, object] | None,
    x_metric: str,
    y_metric: str,
) -> list[dict[str, object]]:
    if not pwms:
        raise ValueError("Cannot compute consensus anchors without PWM inputs.")
    objective_cfg = objective_config if isinstance(objective_config, dict) else {}
    scoring_cfg = objective_cfg.get("scoring")
    scoring_cfg = scoring_cfg if isinstance(scoring_cfg, dict) else {}
    scale = str(objective_cfg.get("score_scale") or "normalized-llr")
    bidirectional = bool(objective_cfg.get("bidirectional", True))
    pseudocounts = float(scoring_cfg.get("pwm_pseudocounts", 0.10))
    log_odds_clip = scoring_cfg.get("log_odds_clip")
    if log_odds_clip is not None:
        log_odds_clip = float(log_odds_clip)

    scorer = Scorer(
        pwms,
        scale=scale,
        bidirectional=bidirectional,
        pseudocounts=pseudocounts,
        log_odds_clip=log_odds_clip,
    )

    anchors: list[dict[str, object]] = []
    for tf_name in tf_names:
        if tf_name not in pwms:
            raise ValueError(f"Cannot compute consensus anchor; missing PWM for TF '{tf_name}'.")
        consensus = pwm_consensus(pwms[tf_name])
        seq = _embed_consensus(consensus, int(sequence_length))
        seq_arr = _encode_sequence(seq)
        per_tf = scorer.compute_all_per_pwm(seq_arr, int(sequence_length))
        x, y = _project_from_per_tf(per_tf, x_metric=x_metric, y_metric=y_metric)
        anchors.append(
            {
                "tf": str(tf_name),
                "label": f"{tf_name} consensus (max)",
                "x": float(x),
                "y": float(y),
                "sequence": seq,
            }
        )
    return anchors
