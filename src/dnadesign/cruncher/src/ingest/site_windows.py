"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/site_windows.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping, Optional

from dnadesign.cruncher.ingest.models import GenomicInterval


def resolve_window_length(
    *,
    tf_name: Optional[str],
    dataset_id: Optional[str],
    window_lengths: Mapping[str, int],
) -> Optional[int]:
    if not window_lengths:
        return None
    if dataset_id:
        for key in (dataset_id, f"dataset:{dataset_id}"):
            if key in window_lengths:
                return int(window_lengths[key])
    if tf_name:
        tf_key = tf_name.strip().lower()
        for key, value in window_lengths.items():
            if key.strip().lower() == tf_key:
                return int(value)
    return None


def window_sequence(
    seq: str,
    length: int,
    *,
    center: str,
    summit_index: Optional[int] = None,
) -> str:
    if length <= 0:
        raise ValueError("window length must be > 0")
    if len(seq) < length:
        raise ValueError(f"sequence length {len(seq)} shorter than requested window {length}")
    if center == "midpoint":
        start = (len(seq) - length) // 2
    elif center == "summit":
        if summit_index is None:
            raise ValueError("summit-centered windows require a summit index")
        start = summit_index - (length // 2)
    else:
        raise ValueError("window center must be 'midpoint' or 'summit'")
    end = start + length
    if start < 0 or end > len(seq):
        raise ValueError("window exceeds sequence bounds")
    return seq[start:end]


def window_interval(
    interval: GenomicInterval,
    length: int,
    *,
    center: str,
    summit_offset: Optional[int] = None,
) -> GenomicInterval:
    if length <= 0:
        raise ValueError("window length must be > 0")
    span = interval.end - interval.start
    if span <= 0:
        raise ValueError("invalid genomic interval")
    if center == "midpoint":
        midpoint = interval.start + (span // 2)
    elif center == "summit":
        if summit_offset is None:
            raise ValueError("summit-centered windows require a summit offset")
        midpoint = interval.start + summit_offset
    else:
        raise ValueError("window center must be 'midpoint' or 'summit'")
    start = midpoint - (length // 2)
    end = start + length
    if start < 0:
        raise ValueError("window start is negative")
    return GenomicInterval(contig=interval.contig, start=start, end=end, assembly=interval.assembly)
