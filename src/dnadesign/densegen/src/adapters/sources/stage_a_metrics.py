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

from .stage_a_encoding import CoreEncodingStore, encode_cores


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

    def __post_init__(self) -> None:
        if len(self.bins) != len(self.counts):
            raise ValueError("KNN summary bins/counts length mismatch.")
        if int(self.n) < 0:
            raise ValueError("KNN summary n must be >= 0.")
        if int(self.k) <= 0:
            raise ValueError("KNN summary k must be >= 1.")

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

    def __post_init__(self) -> None:
        if len(self.bins) != len(self.counts):
            raise ValueError("Pairwise summary bins/counts length mismatch.")
        if int(self.n_pairs) < 0 or int(self.total_pairs) < 0:
            raise ValueError("Pairwise summary pair counts must be >= 0.")
        if int(self.n_pairs) > int(self.total_pairs):
            raise ValueError("Pairwise summary n_pairs must be <= total_pairs.")

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
    top_candidates: PairwiseSummary
    diversified_candidates: PairwiseSummary
    max_diversity_upper_bound: PairwiseSummary | None

    def to_dict(self) -> dict[str, object]:
        return {
            "top_candidates": self.top_candidates.to_dict(),
            "diversified_candidates": self.diversified_candidates.to_dict(),
            "max_diversity_upper_bound": (
                self.max_diversity_upper_bound.to_dict() if self.max_diversity_upper_bound is not None else None
            ),
        }


@dataclass(frozen=True)
class KnnBlock:
    top_candidates: KnnSummary
    diversified_candidates: KnnSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "top_candidates": self.top_candidates.to_dict(),
            "diversified_candidates": self.diversified_candidates.to_dict(),
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
    top_candidates: EntropySummary
    diversified_candidates: EntropySummary

    def to_dict(self) -> dict[str, object]:
        return {
            "top_candidates": self.top_candidates.to_dict(),
            "diversified_candidates": self.diversified_candidates.to_dict(),
        }


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
    top_candidates: ScoreQuantiles | None
    diversified_candidates: ScoreQuantiles | None
    top_candidates_global: ScoreQuantiles | None
    max_diversity_upper_bound: ScoreQuantiles | None

    def to_dict(self) -> dict[str, object]:
        return {
            "top_candidates": self.top_candidates.to_dict() if self.top_candidates is not None else None,
            "diversified_candidates": (
                self.diversified_candidates.to_dict() if self.diversified_candidates is not None else None
            ),
            "top_candidates_global": (
                self.top_candidates_global.to_dict() if self.top_candidates_global is not None else None
            ),
            "max_diversity_upper_bound": (
                self.max_diversity_upper_bound.to_dict() if self.max_diversity_upper_bound is not None else None
            ),
        }


@dataclass(frozen=True)
class DiversitySummary:
    candidate_pool_size: int | None
    nnd_unweighted_k1: KnnBlock
    nnd_unweighted_median_top: float | None
    nnd_unweighted_median_diversified: float | None
    delta_nnd_unweighted_median: float | None
    core_hamming: CoreHammingSummary
    set_overlap_fraction: float | None
    set_overlap_swaps: int | None
    core_entropy: EntropyBlock
    score_quantiles: ScoreQuantilesBlock
    objective_top_candidates: float | None = None
    objective_diversified_candidates: float | None = None
    objective_delta: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_pool_size": self.candidate_pool_size,
            "nnd_unweighted_k1": self.nnd_unweighted_k1.to_dict(),
            "nnd_unweighted_median_top": self.nnd_unweighted_median_top,
            "nnd_unweighted_median_diversified": self.nnd_unweighted_median_diversified,
            "delta_nnd_unweighted_median": self.delta_nnd_unweighted_median,
            "core_hamming": self.core_hamming.to_dict(),
            "set_overlap_fraction": self.set_overlap_fraction,
            "set_overlap_swaps": self.set_overlap_swaps,
            "core_entropy": self.core_entropy.to_dict(),
            "score_quantiles": self.score_quantiles.to_dict(),
            "objective_top_candidates": self.objective_top_candidates,
            "objective_diversified_candidates": self.objective_diversified_candidates,
            "objective_delta": self.objective_delta,
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
    encoding_store: CoreEncodingStore | None = None,
) -> KnnSummary | None:
    if not cores:
        return KnnSummary(
            bins=[0.0],
            counts=[0],
            median=0.0,
            p05=0.0,
            p95=0.0,
            frac_le_1=0.0,
            n=0,
            subsampled=False,
            k=int(k),
        )
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
        encoded = encoding_store.encode(sample) if encoding_store is not None else encode_cores(sample)
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
    encoding_store: CoreEncodingStore | None = None,
) -> KnnSummary | None:
    return _core_hamming_knn(cores, k=1, max_n=max_n, weights=weights, encoding_store=encoding_store)


def _stable_seed_from_sequences(values: Sequence[str]) -> int:
    hashed = [hashlib.md5(str(val).encode("utf-8")).hexdigest() for val in values]
    joined = "|".join(sorted(hashed))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mmr_objective(
    *,
    cores: Sequence[str],
    scores: Sequence[float],
    scores_norm_map: dict[float, float],
    alpha: float,
    distance_weights: Sequence[float] | None = None,
    encoding_store: CoreEncodingStore | None = None,
) -> float | None:
    if not cores or not scores:
        return None
    if len(cores) != len(scores):
        raise ValueError("MMR objective cores/scores length mismatch.")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("MMR objective alpha must be in (0, 1].")
    length = _assert_uniform_core_length(cores, label="mmr objective")
    if length <= 0:
        return None
    weights_arr = None
    if distance_weights is not None:
        weights_arr = np.asarray(distance_weights, dtype=float)
        if weights_arr.shape[0] != length:
            raise ValueError("MMR objective weights length must match core length.")
    encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
    min_dist = np.full(len(cores), np.inf, dtype=float)
    utilities = np.empty(len(cores), dtype=float)
    score_weight = float(alpha)
    diversity_weight = 1.0 - score_weight
    for idx, score in enumerate(scores):
        score_val = float(score)
        score_norm = scores_norm_map.get(score_val)
        if score_norm is None:
            raise ValueError("MMR objective missing normalized score for candidate.")
        if idx == 0:
            max_sim = 0.0
        else:
            max_sim = 1.0 / (1.0 + float(min_dist[idx]))
        utilities[idx] = score_weight * score_norm - diversity_weight * max_sim
        diff = encoded != encoded[idx]
        if weights_arr is None:
            dists = diff.sum(axis=1).astype(float)
        else:
            dists = diff @ weights_arr
        min_dist = np.minimum(min_dist, dists)
        min_dist[idx] = 0.0
    return float(np.mean(utilities))


def _pairwise_hamming_summary(
    cores: Sequence[str],
    *,
    max_pairs: int | None = 10000,
    weights: Sequence[float] | None = None,
    encoding_store: CoreEncodingStore | None = None,
) -> PairwiseSummary | None:
    if not cores:
        return PairwiseSummary(
            bins=[0.0],
            counts=[0],
            median=0.0,
            mean=0.0,
            p10=0.0,
            p90=0.0,
            n_pairs=0,
            total_pairs=0,
            subsampled=False,
        )
    length = len(cores[0])
    if length == 0:
        raise ValueError("Core length must be > 0 for pairwise distance summary.")
    if any(len(core) != length for core in cores):
        raise ValueError("Core length mismatch in pairwise distance summary.")
    if len(cores) < 2:
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape[0] != length:
                raise ValueError("Weighted Hamming requires weights matching core length.")
        return PairwiseSummary(
            bins=[0.0],
            counts=[0],
            median=0.0,
            mean=0.0,
            p10=0.0,
            p90=0.0,
            n_pairs=0,
            total_pairs=0,
            subsampled=False,
        )
    n = len(cores)
    total_pairs = n * (n - 1) // 2
    encoded = encoding_store.encode(cores) if encoding_store is not None else encode_cores(cores)
    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape[0] != length:
            raise ValueError("Weighted Hamming requires weights matching core length.")
    if max_pairs is None:
        sample_pairs = int(total_pairs)
    else:
        sample_pairs = int(min(max_pairs, total_pairs))
    if sample_pairs == total_pairs:
        distances = np.empty(sample_pairs, dtype=float)
        offset = 0
        for idx in range(n - 1):
            diff = encoded[idx + 1 :] != encoded[idx]
            if weights_arr is None:
                row_distances = diff.sum(axis=1).astype(float)
            else:
                row_distances = diff @ weights_arr
            end = offset + row_distances.shape[0]
            distances[offset:end] = row_distances
            offset = end
    else:
        rng = np.random.default_rng(_stable_seed_from_sequences(cores))
        idx_i = rng.integers(0, n, size=sample_pairs)
        idx_j = rng.integers(0, n - 1, size=sample_pairs)
        idx_j = idx_j + (idx_j >= idx_i)
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
    top_candidates_cores: Sequence[str],
    diversified_candidates_cores: Sequence[str],
    top_candidates_scores: Sequence[float],
    diversified_candidates_scores: Sequence[float],
    top_candidates_global_cores: Sequence[str] | None = None,
    top_candidates_global_scores: Sequence[float] | None = None,
    max_diversity_upper_bound_cores: Sequence[str] | None = None,
    max_diversity_upper_bound_scores: Sequence[float] | None = None,
    pwm_theoretical_max_score: float | None = None,
    objective_top_candidates: float | None = None,
    objective_diversified_candidates: float | None = None,
    uniqueness_key: str | None = None,
    candidate_pool_size: int | None = None,
    label: str | None = None,
    max_n: int = 2500,
    distance_weights: Sequence[float] | None = None,
    encoding_store: CoreEncodingStore | None = None,
) -> DiversitySummary | None:
    label = label or "diversity"
    top_len = _assert_uniform_core_length(top_candidates_cores, label=f"{label} top candidates")
    actual_len = _assert_uniform_core_length(diversified_candidates_cores, label=f"{label} diversified candidates")
    if top_len and actual_len and top_len != actual_len:
        raise ValueError(f"Core length mismatch for {label} (top {top_len} vs diversified {actual_len}).")
    global_len = 0
    if top_candidates_global_cores:
        global_len = _assert_uniform_core_length(top_candidates_global_cores, label=f"{label} top candidates global")
    if top_len and global_len and top_len != global_len:
        raise ValueError(f"Core length mismatch for {label} (top {top_len} vs global {global_len}).")
    upper_len = 0
    if max_diversity_upper_bound_cores:
        upper_len = _assert_uniform_core_length(
            max_diversity_upper_bound_cores, label=f"{label} max diversity upper bound"
        )
    if top_len and upper_len and top_len != upper_len:
        raise ValueError(f"Core length mismatch for {label} (top {top_len} vs upper bound {upper_len}).")
    if uniqueness_key == "core" and diversified_candidates_cores:
        if len(set(diversified_candidates_cores)) != len(diversified_candidates_cores):
            raise ValueError(f"Duplicate retained cores detected for {label} with uniqueness.key=core.")
    top_candidates_unweighted_k1 = _core_hamming_knn(
        top_candidates_cores,
        k=1,
        max_n=max_n,
        weights=None,
        encoding_store=encoding_store,
    )
    diversified_candidates_unweighted_k1 = _core_hamming_knn(
        diversified_candidates_cores,
        k=1,
        max_n=max_n,
        weights=None,
        encoding_store=encoding_store,
    )
    if top_candidates_unweighted_k1 is None or diversified_candidates_unweighted_k1 is None:
        return None
    nnd_unweighted_k1 = KnnBlock(
        top_candidates=top_candidates_unweighted_k1,
        diversified_candidates=diversified_candidates_unweighted_k1,
    )
    nnd_unweighted_median_top = float(top_candidates_unweighted_k1.median)
    nnd_unweighted_median_diversified = float(diversified_candidates_unweighted_k1.median)
    delta_nnd_unweighted_median = nnd_unweighted_median_diversified - nnd_unweighted_median_top

    top_candidates_k1 = _core_hamming_knn(
        top_candidates_cores,
        k=1,
        max_n=max_n,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    diversified_candidates_k1 = _core_hamming_knn(
        diversified_candidates_cores,
        k=1,
        max_n=max_n,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    if top_candidates_k1 is None or diversified_candidates_k1 is None:
        return None
    top_candidates_k5 = _core_hamming_knn(
        top_candidates_cores,
        k=5,
        max_n=max_n,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    diversified_candidates_k5 = _core_hamming_knn(
        diversified_candidates_cores,
        k=5,
        max_n=max_n,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    top_candidates_pairwise = _pairwise_hamming_summary(
        top_candidates_cores,
        max_pairs=None,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    diversified_candidates_pairwise = _pairwise_hamming_summary(
        diversified_candidates_cores,
        max_pairs=None,
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    max_diversity_upper_pairwise = None
    if max_diversity_upper_bound_cores:
        max_diversity_upper_pairwise = _pairwise_hamming_summary(
            max_diversity_upper_bound_cores,
            max_pairs=None,
            weights=distance_weights,
            encoding_store=encoding_store,
        )
    top_candidates_entropy = _core_entropy(top_candidates_cores)
    diversified_candidates_entropy = _core_entropy(diversified_candidates_cores)
    score_denominator = None
    if (
        top_candidates_scores
        or diversified_candidates_scores
        or top_candidates_global_scores
        or max_diversity_upper_bound_scores
    ):
        if pwm_theoretical_max_score is None:
            raise ValueError("pwm_theoretical_max_score is required to normalize Stage-A score quantiles.")
        pwm_theoretical_max_score = float(pwm_theoretical_max_score)
        if pwm_theoretical_max_score < 0.0:
            raise ValueError("pwm_theoretical_max_score must be >= 0 to normalize Stage-A score quantiles.")
        if pwm_theoretical_max_score == 0.0:
            all_scores = list(top_candidates_scores) + list(diversified_candidates_scores)
            all_scores += list(top_candidates_global_scores or []) + list(max_diversity_upper_bound_scores or [])
            if any(abs(float(score)) > 1e-9 for score in all_scores):
                raise ValueError("pwm_theoretical_max_score=0 but nonzero scores found in Stage-A score quantiles.")
            score_denominator = 1.0
        else:
            score_denominator = pwm_theoretical_max_score
    top_candidates_norm = (
        [float(score) / score_denominator for score in top_candidates_scores] if top_candidates_scores else []
    )
    diversified_candidates_norm = (
        [float(score) / score_denominator for score in diversified_candidates_scores]
        if diversified_candidates_scores
        else []
    )
    top_candidates_global_norm = (
        [float(score) / score_denominator for score in top_candidates_global_scores]
        if top_candidates_global_scores
        else []
    )
    max_diversity_upper_bound_norm = (
        [float(score) / score_denominator for score in max_diversity_upper_bound_scores]
        if max_diversity_upper_bound_scores
        else []
    )
    top_candidates_quantiles = _score_quantiles(top_candidates_norm)
    diversified_candidates_quantiles = _score_quantiles(diversified_candidates_norm)
    top_candidates_global_quantiles = _score_quantiles(top_candidates_global_norm)
    max_diversity_upper_bound_quantiles = _score_quantiles(max_diversity_upper_bound_norm)
    overlap_fraction = None
    overlap_swaps = None
    if diversified_candidates_cores:
        overlap = len(set(top_candidates_cores) & set(diversified_candidates_cores))
        overlap_fraction = float(overlap) / float(len(diversified_candidates_cores))
        overlap_swaps = int(len(diversified_candidates_cores) - overlap)
    objective_delta = None
    if objective_top_candidates is not None and objective_diversified_candidates is not None:
        objective_delta = float(objective_diversified_candidates) - float(objective_top_candidates)
    core_hamming = CoreHammingSummary(
        metric="weighted_hamming_tolerant" if distance_weights is not None else "hamming",
        nnd_k1=KnnBlock(top_candidates=top_candidates_k1, diversified_candidates=diversified_candidates_k1),
        nnd_k5=KnnBlock(top_candidates=top_candidates_k5, diversified_candidates=diversified_candidates_k5)
        if top_candidates_k5 and diversified_candidates_k5
        else None,
        pairwise=PairwiseBlock(
            top_candidates=top_candidates_pairwise,
            diversified_candidates=diversified_candidates_pairwise,
            max_diversity_upper_bound=max_diversity_upper_pairwise,
        )
        if top_candidates_pairwise and diversified_candidates_pairwise
        else None,
    )
    entropy_block = EntropyBlock(
        top_candidates=EntropySummary(values=top_candidates_entropy, n=int(len(top_candidates_cores))),
        diversified_candidates=EntropySummary(
            values=diversified_candidates_entropy, n=int(len(diversified_candidates_cores))
        ),
    )
    score_block = ScoreQuantilesBlock(
        top_candidates=top_candidates_quantiles,
        diversified_candidates=diversified_candidates_quantiles,
        top_candidates_global=top_candidates_global_quantiles,
        max_diversity_upper_bound=max_diversity_upper_bound_quantiles,
    )
    return DiversitySummary(
        candidate_pool_size=int(candidate_pool_size) if candidate_pool_size is not None else None,
        nnd_unweighted_k1=nnd_unweighted_k1,
        nnd_unweighted_median_top=nnd_unweighted_median_top,
        nnd_unweighted_median_diversified=nnd_unweighted_median_diversified,
        delta_nnd_unweighted_median=delta_nnd_unweighted_median,
        core_hamming=core_hamming,
        set_overlap_fraction=overlap_fraction,
        set_overlap_swaps=overlap_swaps,
        core_entropy=entropy_block,
        score_quantiles=score_block,
        objective_top_candidates=float(objective_top_candidates) if objective_top_candidates is not None else None,
        objective_diversified_candidates=float(objective_diversified_candidates)
        if objective_diversified_candidates is not None
        else None,
        objective_delta=objective_delta,
    )
