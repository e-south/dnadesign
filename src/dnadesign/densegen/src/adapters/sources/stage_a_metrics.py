"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_metrics.py

Stage-A metrics (diversity, mining audits, padding audits).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KnnSummary:
    bins: list[float]
    counts: list[int]
    median: float
    p05: float
    p95: float
    frac_le_1: float
    n: int
    subsampled: bool
    k: int

    def to_dict(self) -> dict[str, object]:
        return {
            "bins": [float(v) for v in self.bins],
            "counts": [int(v) for v in self.counts],
            "median": float(self.median),
            "p05": float(self.p05),
            "p95": float(self.p95),
            "frac_le_1": float(self.frac_le_1),
            "n": int(self.n),
            "subsampled": bool(self.subsampled),
            "k": int(self.k),
        }


@dataclass(frozen=True)
class PairwiseSummary:
    bins: list[float]
    counts: list[int]
    median: float
    mean: float
    p10: float
    p90: float
    n_pairs: int
    total_pairs: int
    subsampled: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "bins": [float(v) for v in self.bins],
            "counts": [int(v) for v in self.counts],
            "median": float(self.median),
            "mean": float(self.mean),
            "p10": float(self.p10),
            "p90": float(self.p90),
            "n_pairs": int(self.n_pairs),
            "total_pairs": int(self.total_pairs),
            "subsampled": bool(self.subsampled),
        }


@dataclass(frozen=True)
class PairwiseBlock:
    baseline: PairwiseSummary
    actual: PairwiseSummary
    upper_bound: PairwiseSummary | None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "actual": self.actual.to_dict(),
            "upper_bound": self.upper_bound.to_dict() if self.upper_bound is not None else None,
        }


@dataclass(frozen=True)
class KnnBlock:
    baseline: KnnSummary
    actual: KnnSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "actual": self.actual.to_dict(),
        }


@dataclass(frozen=True)
class CoreHammingSummary:
    metric: str
    nnd_k1: KnnBlock
    nnd_k5: KnnBlock | None
    pairwise: PairwiseBlock | None

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": str(self.metric),
            "nnd_k1": self.nnd_k1.to_dict(),
            "nnd_k5": self.nnd_k5.to_dict() if self.nnd_k5 is not None else None,
            "pairwise": self.pairwise.to_dict() if self.pairwise is not None else None,
        }


@dataclass(frozen=True)
class EntropySummary:
    values: list[float]
    n: int

    def to_dict(self) -> dict[str, object]:
        return {
            "values": [float(v) for v in self.values],
            "n": int(self.n),
        }


@dataclass(frozen=True)
class EntropyBlock:
    baseline: EntropySummary
    actual: EntropySummary

    def to_dict(self) -> dict[str, object]:
        return {"baseline": self.baseline.to_dict(), "actual": self.actual.to_dict()}


@dataclass(frozen=True)
class ScoreQuantiles:
    p10: float
    p50: float
    p90: float
    mean: float

    def to_dict(self) -> dict[str, object]:
        return {
            "p10": float(self.p10),
            "p50": float(self.p50),
            "p90": float(self.p90),
            "mean": float(self.mean),
        }


@dataclass(frozen=True)
class ScoreQuantilesBlock:
    baseline: ScoreQuantiles | None
    actual: ScoreQuantiles | None
    baseline_global: ScoreQuantiles | None
    upper_bound: ScoreQuantiles | None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "actual": self.actual.to_dict() if self.actual is not None else None,
            "baseline_global": self.baseline_global.to_dict() if self.baseline_global is not None else None,
            "upper_bound": self.upper_bound.to_dict() if self.upper_bound is not None else None,
        }


@dataclass(frozen=True)
class DiversitySummary:
    candidate_pool_size: int | None
    shortlist_target: int | None
    core_hamming: CoreHammingSummary
    set_overlap_fraction: float | None
    set_overlap_swaps: int | None
    core_entropy: EntropyBlock
    score_quantiles: ScoreQuantilesBlock

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_pool_size": self.candidate_pool_size,
            "shortlist_target": self.shortlist_target,
            "core_hamming": self.core_hamming.to_dict(),
            "set_overlap_fraction": self.set_overlap_fraction,
            "set_overlap_swaps": self.set_overlap_swaps,
            "core_entropy": self.core_entropy.to_dict(),
            "score_quantiles": self.score_quantiles.to_dict(),
        }


def _score_quantiles(scores: Sequence[float]) -> ScoreQuantiles | None:
    if not scores:
        return None
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return None
    p10 = np.percentile(arr, 10)
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    return ScoreQuantiles(p10=float(p10), p50=float(p50), p90=float(p90), mean=float(arr.mean()))


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
    weights: Sequence[float] | None = None,
) -> KnnSummary | None:
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
        distances = np.array([0.0], dtype=float)
    else:
        if int(k) >= n:
            return None
        idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        encoded = np.full((n, length), 4, dtype=np.int8)
        for i, core in enumerate(sample):
            encoded[i] = np.array([idx_map.get(base, 4) for base in core], dtype=np.int8)
        weights_arr = None
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape[0] != length:
                raise ValueError("Weighted Hamming requires weights matching core length.")
        distances = np.zeros(n, dtype=float)
        kth = int(k) - 1
        for i in range(n):
            diff = encoded[i] != encoded
            if weights_arr is None:
                dist = diff.sum(axis=1).astype(float)
                dist[i] = float(length) + 1.0
            else:
                dist = (diff * weights_arr).sum(axis=1)
                dist[i] = float(weights_arr.sum()) + 1.0
            distances[i] = float(np.partition(dist, kth)[kth])
    if weights is None:
        max_dist = int(length)
        counts = np.bincount(distances.astype(int), minlength=max_dist + 1)
        centers = list(range(max_dist + 1))
    else:
        max_dist = float(np.asarray(weights, dtype=float).sum())
        bin_count = max(6, min(20, int(length) + 1))
        edges = np.linspace(0.0, max_dist, num=bin_count + 1)
        counts, _ = np.histogram(distances, bins=edges)
        centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    frac_le_1 = float(np.mean(distances <= 1.0)) if distances.size else 0.0
    p05 = float(np.percentile(distances, 5)) if distances.size else 0.0
    p95 = float(np.percentile(distances, 95)) if distances.size else 0.0
    return KnnSummary(
        bins=[float(v) for v in centers],
        counts=[int(v) for v in counts.tolist()],
        median=float(np.median(distances)) if distances.size else 0.0,
        p05=p05,
        p95=p95,
        frac_le_1=float(frac_le_1),
        n=int(n),
        subsampled=bool(subsampled),
        k=int(k),
    )


def _core_hamming_nnd(
    cores: Sequence[str],
    *,
    max_n: int = 2500,
    weights: Sequence[float] | None = None,
) -> KnnSummary | None:
    return _core_hamming_knn(cores, k=1, max_n=max_n, weights=weights)


def _stable_seed_from_sequences(values: Sequence[str]) -> int:
    hashed = [hashlib.md5(str(val).encode("utf-8")).hexdigest() for val in values]
    joined = "|".join(sorted(hashed))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pairwise_hamming_summary(
    cores: Sequence[str],
    *,
    max_pairs: int = 10000,
    weights: Sequence[float] | None = None,
) -> PairwiseSummary | None:
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
    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape[0] != length:
            raise ValueError("Weighted Hamming requires weights matching core length.")
    pairs = []
    while len(pairs) < sample_pairs:
        draw = rng.integers(0, n, size=(sample_pairs * 2, 2))
        mask = draw[:, 0] != draw[:, 1]
        if not np.any(mask):
            continue
        for i, j in draw[mask]:
            pairs.append((int(i), int(j)))
            if len(pairs) >= sample_pairs:
                break
    idx_i = np.fromiter((pair[0] for pair in pairs), dtype=int, count=sample_pairs)
    idx_j = np.fromiter((pair[1] for pair in pairs), dtype=int, count=sample_pairs)
    diff = encoded[idx_i] != encoded[idx_j]
    if weights_arr is None:
        distances = diff.sum(axis=1).astype(float)
    else:
        distances = (diff * weights_arr).sum(axis=1)
    if weights_arr is None:
        max_dist = int(length)
        counts = np.bincount(distances.astype(int), minlength=max_dist + 1)
        centers = list(range(max_dist + 1))
    else:
        max_dist = float(weights_arr.sum())
        bin_count = max(6, min(20, int(length) + 1))
        edges = np.linspace(0.0, max_dist, num=bin_count + 1)
        counts, _ = np.histogram(distances, bins=edges)
        centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    return PairwiseSummary(
        bins=[float(v) for v in centers],
        counts=[int(v) for v in counts.tolist()],
        median=float(np.median(distances)),
        mean=float(distances.mean()),
        p10=float(np.percentile(distances, 10)),
        p90=float(np.percentile(distances, 90)),
        n_pairs=int(sample_pairs),
        total_pairs=int(total_pairs),
        subsampled=bool(sample_pairs < total_pairs),
    )


def _tail_unique_slope(
    generated_by_batch: Sequence[int],
    unique_by_batch: Sequence[int],
    *,
    window: int = 5,
) -> dict[str, object] | None:
    if not generated_by_batch or len(generated_by_batch) < 2:
        return None
    if len(generated_by_batch) != len(unique_by_batch):
        raise ValueError("Generated/unique batch lengths must match for tail slope.")
    window = min(int(window), len(generated_by_batch))
    if window < 2:
        return None
    start_idx = max(0, len(generated_by_batch) - window)
    delta_gen = int(generated_by_batch[-1]) - int(generated_by_batch[start_idx])
    delta_unique = int(unique_by_batch[-1]) - int(unique_by_batch[start_idx])
    if delta_gen <= 0:
        return None
    return {
        "unique_slope": float(delta_unique) / float(delta_gen),
        "unique_slope_window": int(window),
        "unique_slope_generated": int(delta_gen),
    }


def _diversity_summary(
    *,
    baseline_cores: Sequence[str],
    actual_cores: Sequence[str],
    baseline_scores: Sequence[float],
    actual_scores: Sequence[float],
    baseline_global_cores: Sequence[str] | None = None,
    baseline_global_scores: Sequence[float] | None = None,
    upper_bound_cores: Sequence[str] | None = None,
    upper_bound_scores: Sequence[float] | None = None,
    uniqueness_key: str | None = None,
    candidate_pool_size: int | None = None,
    shortlist_target: int | None = None,
    label: str | None = None,
    max_n: int = 2500,
    distance_weights: Sequence[float] | None = None,
) -> DiversitySummary | None:
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
    upper_len = 0
    if upper_bound_cores:
        upper_len = _assert_uniform_core_length(upper_bound_cores, label=f"{label} upper bound")
    if base_len and upper_len and base_len != upper_len:
        raise ValueError(f"Core length mismatch for {label} (baseline {base_len} vs upper bound {upper_len}).")
    if uniqueness_key == "core" and actual_cores:
        if len(set(actual_cores)) != len(actual_cores):
            raise ValueError(f"Duplicate retained cores detected for {label} with uniqueness.key=core.")
    baseline_k1 = _core_hamming_knn(baseline_cores, k=1, max_n=max_n, weights=distance_weights)
    actual_k1 = _core_hamming_knn(actual_cores, k=1, max_n=max_n, weights=distance_weights)
    if baseline_k1 is None or actual_k1 is None:
        return None
    baseline_k5 = _core_hamming_knn(baseline_cores, k=5, max_n=max_n, weights=distance_weights)
    actual_k5 = _core_hamming_knn(actual_cores, k=5, max_n=max_n, weights=distance_weights)
    baseline_pairwise = _pairwise_hamming_summary(baseline_cores, max_pairs=10000, weights=distance_weights)
    actual_pairwise = _pairwise_hamming_summary(actual_cores, max_pairs=10000, weights=distance_weights)
    upper_pairwise = None
    if upper_bound_cores:
        upper_pairwise = _pairwise_hamming_summary(upper_bound_cores, max_pairs=10000, weights=distance_weights)
    baseline_entropy = _core_entropy(baseline_cores)
    actual_entropy = _core_entropy(actual_cores)
    baseline_quantiles = _score_quantiles(baseline_scores)
    actual_quantiles = _score_quantiles(actual_scores)
    baseline_global_quantiles = _score_quantiles(baseline_global_scores or [])
    upper_bound_quantiles = _score_quantiles(upper_bound_scores or [])
    overlap_fraction = None
    overlap_swaps = None
    if actual_cores:
        overlap = len(set(baseline_cores) & set(actual_cores))
        overlap_fraction = float(overlap) / float(len(actual_cores))
        overlap_swaps = int(len(actual_cores) - overlap)
    core_hamming = CoreHammingSummary(
        metric="weighted_hamming_tolerant" if distance_weights is not None else "hamming",
        nnd_k1=KnnBlock(baseline=baseline_k1, actual=actual_k1),
        nnd_k5=KnnBlock(baseline=baseline_k5, actual=actual_k5) if baseline_k5 and actual_k5 else None,
        pairwise=PairwiseBlock(
            baseline=baseline_pairwise,
            actual=actual_pairwise,
            upper_bound=upper_pairwise,
        )
        if baseline_pairwise and actual_pairwise
        else None,
    )
    entropy_block = EntropyBlock(
        baseline=EntropySummary(values=baseline_entropy, n=int(len(baseline_cores))),
        actual=EntropySummary(values=actual_entropy, n=int(len(actual_cores))),
    )
    score_block = ScoreQuantilesBlock(
        baseline=baseline_quantiles,
        actual=actual_quantiles,
        baseline_global=baseline_global_quantiles,
        upper_bound=upper_bound_quantiles,
    )
    return DiversitySummary(
        candidate_pool_size=int(candidate_pool_size) if candidate_pool_size is not None else None,
        shortlist_target=int(shortlist_target) if shortlist_target is not None else None,
        core_hamming=core_hamming,
        set_overlap_fraction=overlap_fraction,
        set_overlap_swaps=overlap_swaps,
        core_entropy=entropy_block,
        score_quantiles=score_block,
    )
