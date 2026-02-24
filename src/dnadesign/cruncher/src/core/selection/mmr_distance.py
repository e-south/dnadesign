"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/selection/mmr_distance.py

Distance and TFBS-core extraction helpers used by MMR selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.sequence import revcomp_int


def compute_position_weights(pwm: PWM) -> np.ndarray:
    matrix = np.asarray(pwm.matrix, dtype=float)
    p = matrix + 1.0e-9
    info = 2.0 + np.sum(p * np.log2(p), axis=1)
    min_info = float(np.min(info))
    max_info = float(np.max(info))
    if max_info - min_info <= 0:
        info_norm = np.zeros_like(info, dtype=float)
    else:
        info_norm = (info - min_info) / (max_info - min_info)
    return 1.0 - info_norm


def compute_core_distance(
    cores_a: dict[str, np.ndarray],
    cores_b: dict[str, np.ndarray],
    *,
    weights: dict[str, np.ndarray],
    tf_names: Sequence[str],
) -> float:
    if not tf_names:
        return 0.0
    distances: list[float] = []
    for tf in tf_names:
        core_a = cores_a[tf]
        core_b = cores_b[tf]
        w = weights[tf]
        if core_a.shape != core_b.shape or core_a.shape != w.shape:
            raise ValueError(f"Core/weight shape mismatch for TF '{tf}'.")
        mismatches = (core_a != core_b).astype(float)
        denom = float(np.sum(w))
        if denom <= 0:
            raise ValueError(f"Non-positive weight sum for TF '{tf}'.")
        tf_distance = float(np.sum(w * mismatches) / denom)
        if tf_distance < -1.0e-12 or tf_distance > 1.0 + 1.0e-12:
            raise ValueError(f"TF core distance for '{tf}' must be in [0, 1], got {tf_distance}.")
        distances.append(float(np.clip(tf_distance, 0.0, 1.0)))
    value = float(np.mean(distances))
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"Core distance must be in [0, 1], got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def core_from_hit(seq_arr: np.ndarray, *, offset: int, width: int, strand: str) -> np.ndarray:
    if width < 1:
        raise ValueError("core_from_hit requires width >= 1")
    if offset < 0:
        raise ValueError("core_from_hit requires offset >= 0")
    window = np.asarray(seq_arr, dtype=np.int8)[offset : offset + width]
    if window.size != width:
        raise ValueError("core_from_hit window is out of bounds for sequence length")
    if strand == "-":
        return revcomp_int(window)
    return window


def tfbs_cores_from_hits(
    seq_arr: np.ndarray,
    *,
    per_tf_hits: dict[str, dict[str, object]],
    tf_names: Sequence[str],
) -> dict[str, np.ndarray]:
    cores: dict[str, np.ndarray] = {}
    for tf in tf_names:
        hit = per_tf_hits.get(tf)
        if not isinstance(hit, dict):
            raise ValueError(f"Missing TF hit data for '{tf}'.")
        offset = hit.get("offset")
        width = hit.get("width")
        strand = hit.get("strand")
        if not isinstance(offset, int) or not isinstance(width, int) or not isinstance(strand, str):
            raise ValueError(f"Invalid TF hit data for '{tf}'.")
        cores[tf] = core_from_hit(seq_arr, offset=offset, width=width, strand=strand)
    return cores


def tfbs_cores_from_scorer(
    seq_arr: np.ndarray,
    *,
    scorer: object,
    tf_names: Sequence[str],
) -> dict[str, np.ndarray]:
    cores: dict[str, np.ndarray] = {}
    seq_length = int(np.asarray(seq_arr).size)
    if seq_length < 1:
        raise ValueError("Core extraction requires non-empty sequences.")
    for tf in tf_names:
        raw_llr, offset, strand = scorer.best_llr(seq_arr, tf)
        _ = raw_llr
        width = int(scorer.pwm_width(tf))
        if width > seq_length:
            width = seq_length
            offset = 0
        if strand == "-":
            rev = revcomp_int(seq_arr)
            core = rev[offset : offset + width]
        else:
            core = seq_arr[offset : offset + width]
        if core.size != width:
            raise ValueError(f"Core extraction failed for '{tf}'.")
        cores[tf] = core
    return cores


def full_sequence_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.int8)
    b_arr = np.asarray(b, dtype=np.int8)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Full-sequence distance requires equal-length sequence arrays.")
    if a_arr.size == 0:
        raise ValueError("Full-sequence distance requires non-empty sequence arrays.")
    value = float(np.count_nonzero(a_arr != b_arr)) / float(a_arr.size)
    if value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"Full-sequence distance must be in [0, 1], got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def full_sequence_distance_bp(a: np.ndarray, b: np.ndarray) -> int:
    a_arr = np.asarray(a, dtype=np.int8)
    b_arr = np.asarray(b, dtype=np.int8)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Full-sequence distance requires equal-length sequence arrays.")
    if a_arr.size == 0:
        raise ValueError("Full-sequence distance requires non-empty sequence arrays.")
    return int(np.count_nonzero(a_arr != b_arr))


def core_hamming_bp(
    cores_a: dict[str, np.ndarray],
    cores_b: dict[str, np.ndarray],
    *,
    tf_names: Sequence[str],
) -> int:
    total = 0
    for tf in tf_names:
        core_a = np.asarray(cores_a[tf], dtype=np.int8)
        core_b = np.asarray(cores_b[tf], dtype=np.int8)
        if core_a.shape != core_b.shape:
            raise ValueError(f"Core shape mismatch for TF '{tf}'.")
        total += int(np.count_nonzero(core_a != core_b))
    return total
