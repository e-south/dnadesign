"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dnadesign.permuter.src.core.storage import read_parquet, read_ref_protein_fasta


def _as_int_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        if not x:
            return []
        # values may be str numbers
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return sorted(list(set(out)))
    # string (e.g., "['16','17']") → very defensive parse
    s = str(x).strip()
    if not s:
        return []
    s = s.replace("[", " ").replace("]", " ").replace(",", " ")
    parts = [p for p in s.split() if p.strip().strip("'\"").isdigit()]
    return sorted(list(set(int(p.strip().strip("'\"")) for p in parts)))


def read_source_records(path_like: str | Path) -> pd.DataFrame:
    p = Path(str(path_like)).expanduser().resolve()
    if p.is_dir():
        p = p / "records.parquet"
    return read_parquet(p)


def ref_aa_from_dataset_dir(dataset_dir: Path) -> Tuple[str | None, str | None]:
    r = read_ref_protein_fasta(dataset_dir)
    if r:
        return r[0], r[1]
    return None, None


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
    - Rejects empty, non-finite, and non-1D inputs.
    """
    if x is None:
        raise TypeError("vector is None")
    x = _maybe_as_py(x)
    # If generic iterable (not str/bytes/dict), list(...) to materialize
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
        raise TypeError("embedding is not 1D")
    if arr.size == 0:
        raise ValueError("embedding is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding contains non-finite values")
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
    Coerce each cell to a 1D float vector and stack into a 2D matrix.
    Assert consistent dimensionality across all rows (and against expected_dim if provided).
    """
    vecs = [coerce_vector1d(v) for v in series.to_list()]
    dims = {len(v) for v in vecs}
    if expected_dim is not None:
        dims.add(int(expected_dim))
    if len(dims) != 1:
        raise ValueError(
            f"inconsistent embedding lengths in '{series.name}': {sorted(dims)}"
        )
    return np.stack(vecs, axis=0)


def filter_valid_source_rows(
    df: pd.DataFrame, *, emb_col: str, aa_col: str, llr_col: str, exp_col: str
) -> tuple[pd.DataFrame, dict]:
    mask_llr = df[llr_col].notna()
    mask_exp = df[exp_col].notna()
    mask_aa = df[aa_col].notna()
    mask_emb = df[emb_col].map(is_numeric_vector1d)
    mask_all = mask_llr & mask_exp & mask_aa & mask_emb
    info = {
        "n_total": int(len(df)),
        "n_kept": int(mask_all.sum()),
        "drops_by_cause": {
            "nan_llr": int((~mask_llr).sum()),
            "nan_expected": int((~mask_exp).sum()),
            "missing_aa_pos_list": int((~mask_aa).sum()),
            "bad_embedding": int((~mask_emb).sum()),
        },
    }
    return df[mask_all].reset_index(drop=True), info


_MUT_RE = re.compile(r"(?i)^([A-Z\*])(\d+)([A-Z\*])$")


def parse_aa_combo_to_map(combo_str: str) -> Dict[int, str]:
    """
    Parse tokens like 'G16F|L17I|N21H' → {16:'F', 17:'I', 21:'H'}.
    Assertive: raises on malformed tokens.
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
            raise ValueError(f"malformed aa token: {t!r}")
        _ref, pos, alt = m.groups()
        out[int(pos)] = alt.upper()
    return out


def pairwise_hamming_from_mutmaps(maps: List[Dict[int, str]]) -> np.ndarray:
    """
    Hamming distance on variant amino-acid sequences, restricted to mutated positions:
      - pos mutated in both, same alt → 0
      - pos mutated in both, different alt → 1
      - pos mutated in exactly one → 1
    """
    n = len(maps)
    out: List[int] = []
    for i in range(n):
        mi = maps[i]
        for j in range(i + 1, n):
            mj = maps[j]
            d = 0
            for p in mi.keys() | mj.keys():
                ai = mi.get(p)
                aj = mj.get(p)
                if ai != aj:
                    d += 1
            out.append(d)
    return np.asarray(out, dtype=int)
