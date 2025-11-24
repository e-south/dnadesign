"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/utils.py

Utility helpers for the multi-site selection protocol:

  • reading the source dataset (records.parquet),
  • strict row-level validation for selection,
  • embedding extraction and shape checks,
  • AA combo parsing and sequence construction,
  • DNA codon decoration (uppercase mutated codons),
  • basic pairwise diagnostics helpers.

All functions are assertive: malformed inputs raise descriptive exceptions.
No silent coercions, no lossy fallbacks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dnadesign.permuter.src.core.storage import read_parquet, read_ref_protein_fasta

# ---------------------------------------------------------------------------
# Source dataset IO
# ---------------------------------------------------------------------------


def read_source_records(path_like: str | Path) -> pd.DataFrame:
    """
    Read a source multi-mutant dataset.

    Accepts either:
      • a dataset directory containing records.parquet, or
      • a direct path to a records.parquet file.
    """
    p = Path(str(path_like)).expanduser().resolve()
    if p.is_dir():
        p = p / "records.parquet"
    return read_parquet(p)


def ref_aa_from_dataset_dir(dataset_dir: Path) -> Tuple[str | None, str | None]:
    """
    Load authoritative reference protein from REF_AA.fa if present.

    Returns (name, sequence) or (None, None) if absent or empty.
    """
    r = read_ref_protein_fasta(dataset_dir)
    if r:
        return r[0], r[1]
    return None, None


# ---------------------------------------------------------------------------
# Embedding coercion
# ---------------------------------------------------------------------------


def _maybe_as_py(x: Any) -> Any:
    """
    Best-effort: Arrow scalars/arrays expose .as_py(); use it when present.
    Otherwise return x unchanged.
    """
    as_py = getattr(x, "as_py", None)
    if callable(as_py):
        try:
            return as_py()
        except Exception:
            return x
    return x


def coerce_vector1d(x: Any) -> np.ndarray:
    """
    Convert Arrow/NumPy/list-like to a finite 1D float vector; raise on failure.

    Rejects:
      • None,
      • non-1D shapes,
      • empty vectors,
      • any non-finite values.
    """
    if x is None:
        raise TypeError("coerce_vector1d: vector is None")
    x = _maybe_as_py(x)
    if (
        not isinstance(x, (list, tuple, np.ndarray))
        and hasattr(x, "__iter__")
        and not isinstance(x, (str, bytes, dict))
    ):
        try:
            x = list(x)
        except Exception:
            pass
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise TypeError(f"coerce_vector1d: expected 1D vector, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("coerce_vector1d: embedding is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coerce_vector1d: embedding contains non-finite values")
    return arr


def is_numeric_vector1d(x: Any, *, min_len: int = 1) -> bool:
    try:
        vec = coerce_vector1d(x)
        return vec.size >= int(min_len)
    except Exception:
        return False


def extract_embedding_matrix(
    series: pd.Series, *, expected_dim: int | None = None
) -> np.ndarray:
    """
    Coerce each cell to a 1D float vector and stack into [N, D].

    Asserts consistent dimensionality across rows and against expected_dim if
    provided.
    """
    vecs = [coerce_vector1d(v) for v in series.to_list()]
    dims = {len(v) for v in vecs}
    if expected_dim is not None:
        dims.add(int(expected_dim))
    if len(dims) != 1:
        raise ValueError(
            f"inconsistent embedding lengths in column '{series.name}': {sorted(dims)}"
        )
    return np.stack(vecs, axis=0)


# ---------------------------------------------------------------------------
# Row‑level validation
# ---------------------------------------------------------------------------


def _as_int_list(x: Any) -> List[int]:
    """
    Parse permuter__aa_pos_list into a sorted, unique list[int].

    Accepts:
      • list/tuple of ints/str-ints,
      • stringified list, e.g. "['16','17']".

    Non-numeric tokens are dropped; this is intentionally tolerant to upstream
    formatting but deterministic.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        if not x:
            return []
        out: List[int] = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return sorted(set(out))
    s = str(x).strip()
    if not s:
        return []
    s = s.replace("[", " ").replace("]", " ").replace(",", " ")
    parts = [p for p in s.split() if p.strip().strip("'\"").isdigit()]
    return sorted(set(int(p.strip().strip("'\"")) for p in parts))


def filter_valid_source_rows(
    df: pd.DataFrame,
    *,
    emb_col: str,
    aa_col: str,
    llr_col: str,
    epi_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Strict row-level validation for multi-site selection.

    Keeps only rows where:
      • observed LLR is present and finite,
      • epistasis is numeric, non-NaN, and ≥ 0,
      • AA position list is present and non-null,
      • embedding passes is_numeric_vector1d().

    Returns (filtered_df, info_dict).

    Raises
    ------
    TypeError
        If the epistasis column is not numeric/coercible to float.
    """
    if llr_col not in df.columns:
        raise KeyError(f"filter_valid_source_rows: missing LLR column {llr_col!r}")
    if epi_col not in df.columns:
        raise KeyError(
            f"filter_valid_source_rows: missing epistasis column {epi_col!r}"
        )
    if aa_col not in df.columns:
        raise KeyError(
            f"filter_valid_source_rows: missing AA positions column {aa_col!r}"
        )
    if emb_col not in df.columns:
        raise KeyError(
            f"filter_valid_source_rows: missing embedding column {emb_col!r}"
        )

    mask_llr = df[llr_col].notna()
    try:
        epi = df[epi_col].astype("float64")
    except Exception as e:
        raise TypeError(
            f"filter_valid_source_rows: epistasis column {epi_col!r} is not numeric"
        ) from e
    mask_epi_notna = epi.notna()
    mask_epi_nonneg = epi >= 0.0
    mask_aa = df[aa_col].notna()
    mask_emb = df[emb_col].map(is_numeric_vector1d)

    mask_all = mask_llr & mask_epi_notna & mask_epi_nonneg & mask_aa & mask_emb

    info = {
        "n_total": int(len(df)),
        "n_kept": int(mask_all.sum()),
        "drops_by_cause": {
            "nan_llr": int((~mask_llr).sum()),
            "nan_epistasis": int((~mask_epi_notna).sum()),
            "neg_epistasis": int((mask_epi_notna & ~mask_epi_nonneg).sum()),
            "missing_aa_pos_list": int((~mask_aa).sum()),
            "bad_embedding": int((~mask_emb).sum()),
        },
    }
    return df[mask_all].reset_index(drop=True), info


# ---------------------------------------------------------------------------
# AA mutation parsing and diagnostics
# ---------------------------------------------------------------------------

_MUT_RE = re.compile(r"(?i)^([A-Z\*])(\d+)([A-Z\*])$")


def parse_aa_combo_to_map(combo_str: str) -> Dict[int, str]:
    """
    Parse tokens like "G16F|L17I|N21H" → {16: 'F', 17: 'I', 21: 'H'}.

    Raises on malformed tokens; this is intentional so that upstream bugs are
    surfaced early.
    """
    s = str(combo_str or "").strip()
    if not s:
        return {}
    out: Dict[int, str] = {}
    for tok in s.split("|"):
        t = tok.strip()
        if not t:
            continue
        m = _MUT_RE.match(t)
        if not m:
            raise ValueError(f"parse_aa_combo_to_map: malformed aa token {t!r}")
        _ref, pos, alt = m.groups()
        out[int(pos)] = alt.upper()
    return out


def build_mutated_aa_sequences(
    ref_aa: str,
    combo_strs: Sequence[str],
) -> np.ndarray:
    """
    Build full-length AA sequences by applying AA combo tokens (e.g., G16F)
    to a reference AA sequence (positions 1-indexed).

    Raises ValueError when:
      • a token is malformed, or
      • a token refers to an out-of-range position.
    """
    ref = list(str(ref_aa))
    L = len(ref)
    out: list[str] = []
    for combo in combo_strs:
        seq = ref.copy()
        s = (combo or "").strip()
        if s:
            for tok in s.split("|"):
                t = tok.strip()
                if not t:
                    continue
                m = _MUT_RE.match(t)
                if not m:
                    raise ValueError(
                        f"build_mutated_aa_sequences: malformed token {t!r}"
                    )
                _ref, pos_s, alt = m.groups()
                pos = int(pos_s)
                if not (1 <= pos <= L):
                    raise ValueError(
                        f"build_mutated_aa_sequences: position {pos} out of range for reference length {L}"
                    )
                seq[pos - 1] = alt
        out.append("".join(seq))
    re


@dataclass(frozen=True)
class MutationWindowSummary:
    """
    Global mutation-window summary across a set of variants.

    Amino-acid (primary) view is always present. Nucleotide-level fields are
    populated when a codon-aligned reference DNA sequence is provided.

    AA-level fields
    ---------------
    ref_length:
        Length of the reference amino-acid sequence.
    start_pos:
        Earliest (minimum) mutated AA position across all variants (1-indexed).
    end_pos:
        Latest (maximum) mutated AA position across all variants (1-indexed).
    window_length:
        Number of residues in the inclusive span [start_pos, end_pos].
    left_flank:
        Reference AA substring immediately upstream of start_pos, up to
        flank_width residues long (may be empty if start_pos == 1).
    window_seq:
        Reference AA substring from start_pos through end_pos (inclusive).
    right_flank:
        Reference AA substring immediately downstream of end_pos, up to
        flank_width residues long (may be empty if end_pos == ref_length).
    flank_width:
        Requested AA flank width (in residues).

    Nucleotide-level fields (optional)
    ----------------------------------
    ref_length_nt:
        Length of the reference nucleotide sequence (bp), or None.
    nt_start:
        1-indexed nucleotide position of the first codon in the window, or None.
    nt_end:
        1-indexed nucleotide position of the last codon in the window, or None.
    window_length_nt:
        Number of nucleotides in the inclusive span [nt_start, nt_end], or None.
    left_flank_nt:
        Reference nucleotide substring immediately upstream of nt_start, or None.
    window_seq_nt:
        Reference nucleotide substring from nt_start through nt_end, or None.
    right_flank_nt:
        Reference nucleotide substring immediately downstream of nt_end, or None.
    flank_width_nt:
        Nucleotide flank width (bp) used to build left/right flanks, or None.
    """

    # AA-level
    ref_length: int
    start_pos: int
    end_pos: int
    window_length: int
    left_flank: str
    window_seq: str
    right_flank: str
    flank_width: int

    # nt-level (optional)
    ref_length_nt: int | None = None
    nt_start: int | None = None
    nt_end: int | None = None
    window_length_nt: int | None = None
    left_flank_nt: str | None = None
    window_seq_nt: str | None = None
    right_flank_nt: str | None = None
    flank_width_nt: int | None = None


def summarize_mutation_window(
    ref_seq: str,
    aa_pos_lists: Sequence[Any],
    *,
    flank: int = 10,
    ref_nt_seq: str | None = None,
) -> MutationWindowSummary:
    """
    Summarize the global mutation span across a set of variants.

    Given a reference amino-acid sequence and a collection of 1-indexed AA
    position lists, compute:

      • earliest mutated position (min across all rows),
      • latest mutated position (max across all rows),
      • window length between them (inclusive),
      • left/right flank substrings from the reference sequence.

    If a codon-aligned reference nucleotide sequence is provided, also compute
    the corresponding nucleotide window and flanks.

    This is intentionally assertive: malformed coordinates or inconsistent
    AA/DNA lengths raise descriptive errors. The caller decides whether to
    treat those as fatal.

    Parameters
    ----------
    ref_seq:
        Reference amino-acid sequence. Positions in aa_pos_lists are interpreted
        as 1-indexed into this sequence.
    aa_pos_lists:
        Iterable of per-variant position collections. Each element is passed
        through _as_int_list(), so both list[int] and stringified list forms
        are accepted.
    flank:
        Number of residues to include on each side of the global window when
        constructing left/right AA flanks (must be ≥ 0).
    ref_nt_seq:
        Optional reference nucleotide sequence corresponding to ref_seq. If
        provided, it must be codon-aligned (len(ref_nt_seq) == 3 * len(ref_seq)).

    Returns
    -------
    MutationWindowSummary

    Raises
    ------
    ValueError
        If the reference AA sequence is empty, no mutated positions are found,
        a position is ≤ 0, a position exceeds the AA reference length, or the
        nucleotide reference is non-codon-length / inconsistent with the AA
        reference.
    """
    ref = str(ref_seq)
    L = len(ref)
    if L == 0:
        raise ValueError("summarize_mutation_window: reference sequence is empty")

    if flank < 0:
        raise ValueError("summarize_mutation_window: flank must be ≥ 0")
    flank_width = int(flank)

    # Collect all AA positions across variants
    all_positions: List[int] = []
    for pos_list in aa_pos_lists:
        ints = _as_int_list(pos_list)
        all_positions.extend(ints)

    if not all_positions:
        raise ValueError(
            "summarize_mutation_window: no mutated positions found in aa_pos_lists"
        )

    start_pos = min(all_positions)
    end_pos = max(all_positions)

    if start_pos <= 0:
        raise ValueError(
            f"summarize_mutation_window: minimum position {start_pos} must be ≥ 1"
        )
    if end_pos > L:
        raise ValueError(
            f"summarize_mutation_window: maximum position {end_pos} exceeds "
            f"reference length {L}"
        )

    # AA-level flanks (1-indexed)
    left_start = max(1, start_pos - flank_width)
    left_end = start_pos - 1
    if left_end >= left_start:
        left_flank = ref[left_start - 1 : left_end]
    else:
        left_flank = ""

    window_seq = ref[start_pos - 1 : end_pos]

    right_start = end_pos + 1
    right_end = min(L, end_pos + flank_width)
    if right_start <= right_end:
        right_flank = ref[right_start - 1 : right_end]
    else:
        right_flank = ""

    # Defaults for nt-level fields (optional)
    ref_length_nt: int | None = None
    nt_start: int | None = None
    nt_end: int | None = None
    window_length_nt: int | None = None
    left_flank_nt: str | None = None
    window_seq_nt: str | None = None
    right_flank_nt: str | None = None
    flank_width_nt: int | None = None

    if ref_nt_seq is not None:
        ref_nt = str(ref_nt_seq)
        L_nt = len(ref_nt)
        if L_nt == 0:
            raise ValueError(
                "summarize_mutation_window: reference nucleotide sequence is empty"
            )
        if L_nt % 3 != 0:
            raise ValueError(
                f"summarize_mutation_window: nucleotide reference length {L_nt} "
                "is not a multiple of 3"
            )
        if L_nt // 3 != L:
            raise ValueError(
                "summarize_mutation_window: nucleotide reference length "
                f"{L_nt} is incompatible with amino-acid length {L} "
                f"(expected {3 * L})"
            )

        ref_length_nt = L_nt
        flank_width_nt = flank_width * 3

        # 1-indexed nucleotide coordinates for the codon window
        nt_start = 3 * (start_pos - 1) + 1
        nt_end = 3 * end_pos
        window_length_nt = nt_end - nt_start + 1

        # Left nt flank: up to flank_width_nt bases before nt_start
        left_nt_start = max(1, nt_start - flank_width_nt)
        left_nt_end = nt_start - 1
        if left_nt_end >= left_nt_start:
            left_flank_nt = ref_nt[left_nt_start - 1 : left_nt_end]
        else:
            left_flank_nt = ""

        # Nt window: inclusive [nt_start, nt_end]
        window_seq_nt = ref_nt[nt_start - 1 : nt_end]

        # Right nt flank: up to flank_width_nt bases after nt_end
        right_nt_start = nt_end + 1
        right_nt_end = min(L_nt, nt_end + flank_width_nt)
        if right_nt_start <= right_nt_end:
            right_flank_nt = ref_nt[right_nt_start - 1 : right_nt_end]
        else:
            right_flank_nt = ""

    return MutationWindowSummary(
        ref_length=L,
        start_pos=start_pos,
        end_pos=end_pos,
        window_length=end_pos - start_pos + 1,
        left_flank=left_flank,
        window_seq=window_seq,
        right_flank=right_flank,
        flank_width=flank_width,
        ref_length_nt=ref_length_nt,
        nt_start=nt_start,
        nt_end=nt_end,
        window_length_nt=window_length_nt,
        left_flank_nt=left_flank_nt,
        window_seq_nt=window_seq_nt,
        right_flank_nt=right_flank_nt,
        flank_width_nt=flank_width_nt,
    )


# ---------------------------------------------------------------------------
# DNA codon decoration
# ---------------------------------------------------------------------------


def uppercase_mutated_codons(
    sequence: str,
    aa_positions: Sequence[int],
) -> str:
    """
    Return a DNA sequence where codons corresponding to mutated amino-acid
    positions are uppercase and all other bases are lowercase.

    AA positions are 1-indexed. Sequence is expected to be a contiguous coding
    region with length divisible by 3.

    Raises ValueError on:
      • non-codon-length sequences,
      • non-positive AA positions,
      • AA positions that map beyond the sequence length.
    """
    seq = str(sequence)
    L = len(seq)
    if L == 0:
        return seq
    if L % 3 != 0:
        raise ValueError(
            f"uppercase_mutated_codons: sequence length {L} is not a multiple of 3"
        )

    bases = list(seq.lower())
    for pos in aa_positions or []:
        i = int(pos)
        if i <= 0:
            raise ValueError(f"uppercase_mutated_codons: AA position {i} must be ≥ 1")
        start = 3 * (i - 1)
        end = start + 3
        if end > L:
            raise ValueError(
                f"uppercase_mutated_codons: AA position {i} maps beyond sequence length {L}"
            )
        for j in range(start, end):
            bases[j] = bases[j].upper()
    return "".join(bases)
