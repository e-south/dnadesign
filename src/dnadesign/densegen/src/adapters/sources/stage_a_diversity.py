"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_diversity.py

Stage-A core diversity metrics for PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def _score_quantiles(scores: Sequence[float]) -> dict[str, float] | None:
    if not scores:
        return None
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return None
    p10 = np.percentile(arr, 10)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "mean": float(arr.mean()),
    }


def _assert_uniform_core_length(cores: Sequence[str], *, label: str) -> int:
    if not cores:
        return 0
    lengths = {len(core) for core in cores}
    if len(lengths) > 1:
        preview = ", ".join(sorted({str(len(core)) for core in cores[:10]}))
        raise ValueError(f"Non-uniform core lengths for {label} (saw {preview}).")
    return int(next(iter(lengths)))


def _stable_subsample(values: Sequence[str], max_size: int) -> tuple[list[str], bool]:
    if len(values) <= max_size:
        return list(values), False
    keyed = []
    for idx, val in enumerate(values):
        digest = hashlib.md5(str(val).encode("utf-8")).hexdigest()
        keyed.append((digest, idx, val))
    keyed.sort()
    sampled = [val for _, _, val in keyed[: int(max_size)]]
    return sampled, True


def _core_entropy(cores: Sequence[str]) -> list[float]:
    if not cores:
        return []
    length = len(cores[0])
    if any(len(core) != length for core in cores):
        return []
    counts = np.zeros((length, 4), dtype=float)
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    for core in cores:
        for pos, base in enumerate(core):
            idx = idx_map.get(base)
            if idx is None:
                continue
            counts[pos, idx] += 1.0
    entropies: list[float] = []
    for row in counts:
        total = float(row.sum())
        if total <= 0:
            entropies.append(0.0)
            continue
        probs = row / total
        mask = probs > 0
        entropies.append(float(-(probs[mask] * np.log2(probs[mask])).sum()))
    return entropies


def _core_hamming_knn(
    cores: Sequence[str],
    *,
    k: int,
    max_n: int = 2500,
) -> dict[str, object] | None:
    if not cores:
        return None
    if int(k) <= 0:
        raise ValueError("k must be >= 1 for k-nearest neighbor distances.")
    length = len(cores[0])
    if length == 0 or any(len(core) != length for core in cores):
        return None
    sample, subsampled = _stable_subsample(cores, max_n)
    n = len(sample)
    if n == 0:
        return None
    if n == 1:
        if int(k) > 1:
            return None
        distances = np.array([0], dtype=int)
    else:
        if int(k) >= n:
            return None
        idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        encoded = np.full((n, length), 4, dtype=np.int8)
        for i, core in enumerate(sample):
            encoded[i] = np.array([idx_map.get(base, 4) for base in core], dtype=np.int8)
        distances = np.zeros(n, dtype=int)
        kth = int(k) - 1
        for i in range(n):
            diff = (encoded[i] != encoded).sum(axis=1)
            diff[i] = length + 1
            distances[i] = int(np.partition(diff, kth)[kth])
    max_dist = int(length)
    counts = np.bincount(distances, minlength=max_dist + 1)
    frac_le_1 = float(np.mean(distances <= 1)) if distances.size else 0.0
    p05 = float(np.percentile(distances, 5)) if distances.size else 0.0
    p95 = float(np.percentile(distances, 95)) if distances.size else 0.0
    return {
        "bins": list(range(max_dist + 1)),
        "counts": [int(v) for v in counts.tolist()],
        "median": float(np.median(distances)) if distances.size else 0.0,
        "p05": p05,
        "p95": p95,
        "frac_le_1": float(frac_le_1),
        "n": int(n),
        "subsampled": bool(subsampled),
        "k": int(k),
    }


def _core_hamming_nnd(cores: Sequence[str], *, max_n: int = 2500) -> dict[str, object] | None:
    return _core_hamming_knn(cores, k=1, max_n=max_n)


def _stable_seed_from_sequences(values: Sequence[str]) -> int:
    hashed = [hashlib.md5(str(val).encode("utf-8")).hexdigest() for val in values]
    joined = "|".join(sorted(hashed))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pairwise_hamming_summary(cores: Sequence[str], *, max_pairs: int = 10000) -> dict[str, object] | None:
    if len(cores) < 2:
        return None
    length = len(cores[0])
    if length == 0 or any(len(core) != length for core in cores):
        return None
    n = len(cores)
    total_pairs = n * (n - 1) // 2
    sample_pairs = int(min(max_pairs, total_pairs))
    rng = np.random.default_rng(_stable_seed_from_sequences(cores))
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.full((n, length), 4, dtype=np.int8)
    for i, core in enumerate(cores):
        encoded[i] = np.array([idx_map.get(base, 4) for base in core], dtype=np.int8)
    distances: list[int] = []
    while len(distances) < sample_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        dist = int((encoded[i] != encoded[j]).sum())
        distances.append(dist)
    arr = np.asarray(distances, dtype=float)
    return {
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_pairs": int(sample_pairs),
        "total_pairs": int(total_pairs),
    }


def _diversity_summary(
    *,
    baseline_cores: Sequence[str],
    actual_cores: Sequence[str],
    baseline_scores: Sequence[float],
    actual_scores: Sequence[float],
    baseline_global_cores: Sequence[str] | None = None,
    baseline_global_scores: Sequence[float] | None = None,
    uniqueness_key: str | None = None,
    candidate_pool_size: int | None = None,
    shortlist_target: int | None = None,
    label: str | None = None,
    max_n: int = 2500,
) -> dict[str, object] | None:
    label = label or "diversity"
    base_len = _assert_uniform_core_length(baseline_cores, label=f"{label} baseline")
    actual_len = _assert_uniform_core_length(actual_cores, label=f"{label} actual")
    if base_len and actual_len and base_len != actual_len:
        raise ValueError(f"Core length mismatch for {label} (baseline {base_len} vs actual {actual_len}).")
    global_len = 0
    if baseline_global_cores:
        global_len = _assert_uniform_core_length(baseline_global_cores, label=f"{label} baseline global")
    if base_len and global_len and base_len != global_len:
        raise ValueError(f"Core length mismatch for {label} (baseline {base_len} vs global {global_len}).")
    if uniqueness_key == "core" and actual_cores:
        if len(set(actual_cores)) != len(actual_cores):
            raise ValueError(f"Duplicate retained cores detected for {label} with uniqueness.key=core.")
    baseline_k1 = _core_hamming_knn(baseline_cores, k=1, max_n=max_n)
    actual_k1 = _core_hamming_knn(actual_cores, k=1, max_n=max_n)
    if baseline_k1 is None or actual_k1 is None:
        return None
    baseline_k5 = _core_hamming_knn(baseline_cores, k=5, max_n=max_n)
    actual_k5 = _core_hamming_knn(actual_cores, k=5, max_n=max_n)
    baseline_pairwise = _pairwise_hamming_summary(baseline_cores, max_pairs=10000)
    actual_pairwise = _pairwise_hamming_summary(actual_cores, max_pairs=10000)
    baseline_entropy = _core_entropy(baseline_cores)
    actual_entropy = _core_entropy(actual_cores)
    baseline_quantiles = _score_quantiles(baseline_scores)
    actual_quantiles = _score_quantiles(actual_scores)
    baseline_global_quantiles = _score_quantiles(baseline_global_scores or [])
    overlap_fraction = None
    overlap_swaps = None
    if actual_cores:
        overlap = len(set(baseline_cores) & set(actual_cores))
        overlap_fraction = float(overlap) / float(len(actual_cores))
        overlap_swaps = int(len(actual_cores) - overlap)
    core_hamming = {
        "nnd_k1": {"baseline": baseline_k1, "actual": actual_k1},
        "nnd_k5": {"baseline": baseline_k5, "actual": actual_k5}
        if baseline_k5 is not None and actual_k5 is not None
        else None,
        "pairwise": {"baseline": baseline_pairwise, "actual": actual_pairwise}
        if baseline_pairwise is not None and actual_pairwise is not None
        else None,
    }
    return {
        "candidate_pool_size": candidate_pool_size,
        "shortlist_target": shortlist_target,
        "core_hamming": core_hamming,
        "set_overlap_fraction": overlap_fraction,
        "set_overlap_swaps": overlap_swaps,
        "core_entropy": {
            "baseline": {"values": baseline_entropy, "n": int(len(baseline_cores))},
            "actual": {"values": actual_entropy, "n": int(len(actual_cores))},
        },
        "score_quantiles": {
            "baseline": baseline_quantiles,
            "actual": actual_quantiles,
            "baseline_global": baseline_global_quantiles,
        },
    }
