"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/metrics.py

Metrics module for high-level swtools functions.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .align import global_alignment
from .cache import generate_cache_filename, load_cache, save_cache
from .matrix import build_score_matrix, matrix_to_condensed
from .utils import extract_sequence


def compute_alignment_scores(
    sequences: List[Union[str, dict]],
    sequence_key: str = "sequence",
    output: str = "mean",  # Options: "mean", "matrix", "condensed"
    normalize: bool = True,
    normalization: str = "max_score",  # "max_score" or "alignment_length"
    use_cache: bool = True,
    cache_dir: Union[str, Path, None] = None,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,  # New affine gap default: 10
    gap_extend: int = 1,  # New affine gap default: 1
    return_formats: Tuple[str, ...] = ("mean", "condensed"),
    parallel: bool = True,
    num_workers: Union[int, None] = None,
    return_raw: bool = False,
    return_dissimilarity: bool = False,
    verbose: bool = False,
) -> Union[float, np.ndarray, Dict[str, Any]]:
    """
    High-level function to compute global alignment scores for a set of sequences.

    This function computes a normalized similarity score based on global alignment.
    It also computes the dissimilarity score (defined as 1 - normalized similarity)
    if requested.

    Parameters:
        sequences: List of sequences (strings or dicts with a sequence key).
        sequence_key: Key to extract the sequence if the item is a dict.
        output: Primary output mode.
        normalize: If True, normalize the scores.
        normalization: "max_score" (default) uses: match * L as max score.
                       "alignment_length" uses the actual alignment length.
        use_cache: Enable disk caching.
        cache_dir: Directory for cache files (default: ./swcache).
        match, mismatch, gap_open, gap_extend: Alignment scoring parameters.
            (Defaults: match=2, mismatch=-1, gap_open=10, gap_extend=1).
        return_formats: Requested outputs (e.g., "mean", "matrix", "condensed").
        parallel: Enables parallel processing (if applicable).
        num_workers: Number of workers (if parallel processing is used).
        return_raw: If True, return both raw and normalized scores.
        return_dissimilarity: If True, include dissimilarity = 1 - normalized score.
        verbose: If True, prints progress messages.

    Returns:
        A float or dictionary with the requested outputs. When return_dissimilarity is True,
        the dictionary includes a "dissimilarity" key.
    """
    if cache_dir is None:
        cache_dir = Path("./swcache")
    else:
        cache_dir = Path(cache_dir)

    clean_seqs = [extract_sequence(item, sequence_key) for item in sequences]
    n = len(clean_seqs)

    if verbose:
        print(f"Computing global alignment for {n} sequences.")
    if n > 1000:
        est_comparisons = n * (n - 1) // 2
        print(f"Warning: {n} sequences generate ~{est_comparisons} comparisons; performance may be impacted.")

    cache_filename = generate_cache_filename(
        n, normalize, gap_open, gap_extend, matrix_id="nt", return_formats=return_formats
    )
    if use_cache:
        cached_data = load_cache(cache_dir, cache_filename)
        if cached_data is not None:
            if verbose:
                print(f"Loaded cache from {cache_dir / cache_filename}")
            return cached_data

    full_matrix = build_score_matrix(clean_seqs, match, mismatch, gap_open, gap_extend)

    if normalize:
        norm_matrix = np.zeros_like(full_matrix, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    norm_matrix[i, j] = 1.0
                else:
                    L = len(clean_seqs[i])
                    denom = match * L
                    norm_matrix[i, j] = full_matrix[i, j] / denom if denom > 0 else 0.0
    else:
        norm_matrix = full_matrix

    outputs: Dict[str, Any] = {}
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

    if return_dissimilarity:
        if "mean" in outputs:
            outputs["dissimilarity"] = 1 - outputs["mean"]

    result: Union[float, Dict[str, Any]] = outputs["mean"] if return_formats == ("mean",) else outputs
    if return_raw:
        result = {"normalized": outputs, "raw": full_matrix}

    if use_cache:
        save_cache(cache_dir, cache_filename, result)
        if verbose:
            print(f"Saved cache to {cache_dir / cache_filename}")

    return result


def mean_pairwise(sequences: List[Union[str, dict]], sequence_key: str = "sequence", **kwargs) -> float:
    """
    Convenience wrapper that returns only the mean normalized score.
    """
    result = compute_alignment_scores(
        sequences=sequences, sequence_key=sequence_key, return_formats=("mean",), **kwargs
    )
    if isinstance(result, dict):
        return result.get("mean", 0.0)
    return result


def score_pairwise(
    seqA: str,
    seqB: str,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 10,  # Updated default affine gap (10 -> -10 internally)
    gap_extend: int = 1,  # Updated default affine gap (1 -> -1 internally)
    normalization: str = "max_score",
    return_raw: bool = False,
    return_alignment_str: bool = False,
    return_dissimilarity: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Compute the global alignment score for a pair of raw sequences using affine gap penalties.

    This function performs a full (global) alignment, then computes a normalized similarity score,
    and (if requested) a dissimilarity score defined as:

        Dissimilarity = 1 - (Normalized Similarity)

    It can also return the full alignment string for visualization.

    Parameters:
        seqA, seqB: Two nucleotide sequences.
        match, mismatch, gap_open, gap_extend: Alignment parameters.
            Defaults: match=2, mismatch=-1, gap_open=10, gap_extend=1.
        normalization: "max_score" or "alignment_length".
        return_raw: If True, return a dict with both raw and normalized scores.
        return_alignment_str: If True, include the full alignment string.
        return_dissimilarity: If True, include a dissimilarity score.

    Returns:
        A float if neither return_raw nor return_alignment_str is True; otherwise, a dict with:
            - "raw": Raw alignment score.
            - "normalized": Normalized similarity score.
            - "dissimilarity": (if requested) 1 - normalized score.
            - "alignment": (if requested) The full alignment string.
    """
    from .utils import validate_sequence

    seqA = validate_sequence(seqA)
    seqB = validate_sequence(seqB)

    result = global_alignment(
        seqA, seqB, match, mismatch, gap_open, gap_extend, return_alignment_str=return_alignment_str
    )

    if return_alignment_str:
        raw_score, alignment_str = result
    else:
        raw_score = result

    if normalization == "max_score":
        denom = match * min(len(seqA), len(seqB))
    elif normalization == "alignment_length":
        denom = min(len(seqA), len(seqB))
    else:
        raise ValueError(f"Unknown normalization strategy: {normalization}")

    norm_score = raw_score / denom if denom > 0 else 0.0

    ret = {"raw": raw_score, "normalized": norm_score}
    if return_dissimilarity:
        ret["dissimilarity"] = 1 - norm_score
    if return_alignment_str:
        ret["alignment"] = alignment_str
    if not return_raw and not return_alignment_str:
        return norm_score
    return ret
