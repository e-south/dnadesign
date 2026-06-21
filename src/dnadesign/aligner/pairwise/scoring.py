"""Needleman-Wunsch style global pairwise scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from Bio.Align import PairwiseAligner

from dnadesign.aligner.pairwise.cache import generate_cache_filename, load_cache, save_cache
from dnadesign.aligner.pairwise.validation import extract_sequence, validate_sequence


def global_alignment(
    seq_a: str,
    seq_b: str,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,
    gap_extend: int = 1,
    return_alignment_str: bool = False,
) -> float | tuple[float, str]:
    """Perform a global alignment between two sequences with affine gap penalties."""

    try:
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = match
        aligner.mismatch_score = mismatch
        aligner.open_gap_score = -abs(gap_open)
        aligner.extend_gap_score = -abs(gap_extend)

        alignments = aligner.align(seq_a, seq_b)
        if not alignments:
            return (0.0, "") if return_alignment_str else 0.0
        best_alignment = alignments[0]
        score = best_alignment.score
        if return_alignment_str:
            return score, str(best_alignment)
        return score
    except Exception as exc:  # pragma: no cover - Biopython error path.
        raise RuntimeError(f"Alignment error: {exc}") from exc


def compute_alignment_scores(
    sequences: Sequence[str | dict[str, Any]],
    sequence_key: str = "sequence",
    output: str = "mean",
    normalize: bool = True,
    normalization: str = "max_score",
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,
    gap_extend: int = 1,
    return_formats: tuple[str, ...] = ("mean", "condensed"),
    parallel: bool = True,
    num_workers: int | None = None,
    return_raw: bool = False,
    return_dissimilarity: bool = False,
    verbose: bool = False,
) -> float | np.ndarray | dict[str, Any]:
    """Compute global pairwise alignment scores for a sequence collection."""

    del output, normalization, parallel, num_workers
    cache_path = Path("./swcache") if cache_dir is None else Path(cache_dir)
    clean_seqs = [extract_sequence(item, sequence_key) for item in sequences]
    n = len(clean_seqs)

    if verbose:
        print(f"Computing global alignment for {n} sequences.")
    if n > 1000:
        est_comparisons = n * (n - 1) // 2
        print(f"Warning: {n} sequences generate ~{est_comparisons} comparisons; performance may be impacted.")

    cache_filename = generate_cache_filename(
        clean_seqs,
        normalize,
        match,
        mismatch,
        gap_open,
        gap_extend,
        matrix_id="nt",
        return_formats=return_formats,
    )
    if use_cache:
        cached_data = load_cache(cache_path, cache_filename)
        if cached_data is not None:
            if verbose:
                print(f"Loaded cache from {cache_path / cache_filename}")
            return cached_data

    from dnadesign.aligner.pairwise.matrix import build_score_matrix, matrix_to_condensed

    full_matrix = build_score_matrix(clean_seqs, match, mismatch, gap_open, gap_extend)
    if normalize:
        norm_matrix = np.zeros_like(full_matrix, dtype=np.float32)
        for i in range(n):
            norm_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                denom = match * max(len(clean_seqs[i]), len(clean_seqs[j]))
                value = full_matrix[i, j] / denom if denom > 0 else 0.0
                norm_matrix[i, j] = value
                norm_matrix[j, i] = value
    else:
        norm_matrix = full_matrix

    outputs: dict[str, Any] = {}
    if "matrix" in return_formats:
        outputs["matrix"] = norm_matrix
    if "condensed" in return_formats:
        outputs["condensed"] = matrix_to_condensed(norm_matrix)
    if "mean" in return_formats:
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += norm_matrix[i, j]
                count += 1
        outputs["mean"] = total / count if count > 0 else 0.0

    if return_dissimilarity and "mean" in outputs:
        outputs["dissimilarity"] = 1 - outputs["mean"]

    result: float | dict[str, Any] = outputs["mean"] if return_formats == ("mean",) else outputs
    if return_raw:
        result = {"normalized": outputs, "raw": full_matrix}

    if use_cache:
        save_cache(cache_path, cache_filename, result)
        if verbose:
            print(f"Saved cache to {cache_path / cache_filename}")

    return result


def mean_pairwise(sequences: Sequence[str | dict[str, Any]], sequence_key: str = "sequence", **kwargs: Any) -> float:
    """Return only the mean normalized pairwise score."""

    result = compute_alignment_scores(
        sequences=sequences,
        sequence_key=sequence_key,
        return_formats=("mean",),
        **kwargs,
    )
    if isinstance(result, dict):
        return float(result.get("mean", 0.0))
    return float(result)


def score_pairwise(
    seq_a: str,
    seq_b: str,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,
    gap_extend: int = 1,
    normalization: str = "max_score",
    return_raw: bool = False,
    return_alignment_str: bool = False,
    return_dissimilarity: bool = False,
) -> float | dict[str, float | str]:
    """Compute a normalized global alignment score for a pair of nucleotide sequences."""

    seq_a = validate_sequence(seq_a)
    seq_b = validate_sequence(seq_b)
    result = global_alignment(
        seq_a,
        seq_b,
        match,
        mismatch,
        gap_open,
        gap_extend,
        return_alignment_str=return_alignment_str,
    )

    if return_alignment_str:
        raw_score, alignment_str = result
    else:
        raw_score = result

    if normalization == "max_score":
        denom = match * min(len(seq_a), len(seq_b))
    elif normalization == "alignment_length":
        denom = min(len(seq_a), len(seq_b))
    else:
        raise ValueError(f"Unknown normalization strategy: {normalization}")

    norm_score = raw_score / denom if denom > 0 else 0.0
    ret: dict[str, float | str] = {"raw": float(raw_score), "normalized": float(norm_score)}
    if return_dissimilarity:
        ret["dissimilarity"] = 1 - float(norm_score)
    if return_alignment_str:
        ret["alignment"] = alignment_str
    if not return_raw and not return_alignment_str:
        return float(norm_score)
    return ret
