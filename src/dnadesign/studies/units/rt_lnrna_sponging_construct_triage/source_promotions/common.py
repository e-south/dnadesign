"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/source_promotions/common.py

Shared source-promotion primitives for the RT-lnRNA study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import SourcePromotionContractError

DNA_ALPHABET = frozenset("ACGT")
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


@dataclass(frozen=True, slots=True)
class ConstructWindowPolicy:
    context_id: str
    target_start_0: int
    target_length_nt: int
    template_length_nt: int
    lnrna_template_span_0: tuple[int, int]
    rt_cds_template_span_0: tuple[int, int]

    def __post_init__(self) -> None:
        if self.target_start_0 < 0:
            raise SourcePromotionContractError("Construct target_start_0 must be non-negative.")
        if self.target_length_nt <= 0:
            raise SourcePromotionContractError("Construct target_length_nt must be positive.")
        if self.template_length_nt <= 0:
            raise SourcePromotionContractError("Construct template_length_nt must be positive.")
        _require_span(self.lnrna_template_span_0, label="lnrna_template_span_0")
        _require_span(self.rt_cds_template_span_0, label="rt_cds_template_span_0")
        if self.lnrna_template_span_0[1] > self.rt_cds_template_span_0[0]:
            raise SourcePromotionContractError("Construct lnRNA template span must precede RT CDS template span.")


def read_tsv(path: Path) -> tuple[Mapping[str, str], ...]:
    if not path.exists():
        raise SourcePromotionContractError(f"Source promotion table is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle, delimiter="\t"))


def require_dna(value: str, *, label: str) -> str:
    sequence = str(value or "").strip().upper()
    if not sequence:
        raise SourcePromotionContractError(f"{label} must be non-empty.")
    bad = sorted({base for base in sequence if base not in DNA_ALPHABET})
    if bad:
        raise SourcePromotionContractError(f"{label} contains non-DNA4 bases: {bad}.")
    return sequence


def require_no_internal_stop_codons(sequence: str, *, label: str) -> None:
    stop_positions = [
        index // 3 + 1 for index in range(0, max(len(sequence) - 3, 0), 3) if sequence[index : index + 3] in STOP_CODONS
    ]
    if stop_positions:
        raise SourcePromotionContractError(
            f"{label} contains internal stop codon(s) at AA position(s): {stop_positions}."
        )


def construct_window_fit_issue(
    *,
    lnrna_sequence: str,
    rt_cds_sequence: str,
    window_policy: ConstructWindowPolicy,
) -> str:
    issue = _construct_window_fit_issue(
        lnrna_length=len(lnrna_sequence),
        rt_cds_length=len(rt_cds_sequence),
        window_policy=window_policy,
    )
    return issue


def _construct_window_fit_issue(
    *,
    lnrna_length: int,
    rt_cds_length: int,
    window_policy: ConstructWindowPolicy,
) -> str:
    lnrna_start, lnrna_end = window_policy.lnrna_template_span_0
    rt_start, rt_end = window_policy.rt_cds_template_span_0
    base_lnrna_length = lnrna_end - lnrna_start
    base_lnrna_center = lnrna_start + (base_lnrna_length // 2)
    realized_lnrna_start = lnrna_start
    realized_lnrna_end = realized_lnrna_start + lnrna_length
    realized_lnrna_center = realized_lnrna_start + (lnrna_length // 2)
    realized_rt_start = rt_start + (lnrna_length - base_lnrna_length)
    realized_rt_end = realized_rt_start + rt_cds_length
    window_start = window_policy.target_start_0 + (realized_lnrna_center - base_lnrna_center)
    window_end = window_start + window_policy.target_length_nt
    realized_template_length = (
        window_policy.template_length_nt + (lnrna_length - base_lnrna_length) + (rt_cds_length - (rt_end - rt_start))
    )
    if (
        window_start >= 0
        and window_end <= realized_template_length
        and realized_lnrna_start >= window_start
        and realized_rt_end <= window_end
    ):
        return ""
    return (
        f"source lnRNA plus RT CDS falls outside {window_policy.context_id}: "
        f"lnrna_length={lnrna_length}, rt_cds_length={rt_cds_length}, "
        f"window_start_0={window_start}, window_end_0={window_end}, "
        f"realized_lnrna_span_0={realized_lnrna_start}:{realized_lnrna_end}, "
        f"realized_rt_cds_span_0={realized_rt_start}:{realized_rt_end}, "
        f"realized_template_length={realized_template_length}."
    )


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


def row_id(row: Mapping[str, str]) -> str:
    return str(row.get("record_id") or row.get("observation_id") or row.get("source_record_id") or "").strip()


def format_ratio(value: float) -> str:
    return f"{value:.6f}"


def format_span(value: tuple[int, int] | None) -> str:
    if value is None:
        return ""
    return f"{value[0]}:{value[1]}"


def join_row_ids(rows: list[Mapping[str, str]]) -> str:
    return ";".join(row_id(row) for row in rows)


def join_values(rows: list[Mapping[str, str]], field: str) -> str:
    return ";".join(str(row.get(field) or "").strip() for row in rows if str(row.get(field) or "").strip())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def slug(value: str) -> str:
    out = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    text = "".join(out).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "source_record"


def duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return tuple(sorted(repeated))


def _require_span(span: tuple[int, int], *, label: str) -> None:
    start, end = span
    if start < 0 or end <= start:
        raise SourcePromotionContractError(f"{label} must be a non-empty zero-based half-open span.")
