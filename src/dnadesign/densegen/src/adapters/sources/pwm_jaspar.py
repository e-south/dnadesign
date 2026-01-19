"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_jaspar.py

PWM input source (JASPAR PFM format) with explicit sampling policies.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_jaspar(path: Path) -> List[PWMMotif]:
    motifs: List[PWMMotif] = []
    current_id: Optional[str] = None
    rows: dict[str, List[float]] = {}

    def _finalize():
        nonlocal current_id, rows
        if current_id is None:
            return
        for base in ("A", "C", "G", "T"):
            if base not in rows:
                raise ValueError(f"JASPAR motif {current_id} missing row for {base}")
        lengths = {len(rows[b]) for b in ("A", "C", "G", "T")}
        if len(lengths) != 1:
            raise ValueError(f"JASPAR motif {current_id} has inconsistent row lengths: {sorted(lengths)}")
        width = lengths.pop()
        if width <= 0:
            raise ValueError(f"JASPAR motif {current_id} has zero width.")
        matrix = []
        for i in range(width):
            a, c, g, t = rows["A"][i], rows["C"][i], rows["G"][i], rows["T"][i]
            total = a + c + g + t
            if total <= 0:
                raise ValueError(f"JASPAR motif {current_id} has zero-sum column at {i}.")
            matrix.append({"A": a / total, "C": c / total, "G": g / total, "T": t / total})
        motifs.append(PWMMotif(motif_id=current_id, matrix=matrix, background=normalize_background(None)))
        current_id = None
        rows = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _finalize()
            parts = line[1:].strip().split()
            if not parts:
                raise ValueError("JASPAR motif header missing ID.")
            current_id = parts[0].strip()
            rows = {}
            continue
        if current_id is None:
            continue
        base = line[0].upper()
        if base not in {"A", "C", "G", "T"}:
            raise ValueError(f"Unexpected row label '{base}' in JASPAR motif {current_id}")
        nums = _NUM_RE.findall(line[1:])
        if not nums:
            raise ValueError(f"JASPAR row for {current_id} ({base}) is empty.")
        rows[base] = [float(x) for x in nums]

    _finalize()
    if not motifs:
        raise ValueError("No motifs found in JASPAR file.")
    return motifs


@dataclass
class PWMJasparDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_ids: Optional[List[str]]
    sampling: dict

    def load_data(self, *, rng=None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        jaspar_path = resolve_path(self.cfg_path, self.path)
        if not (jaspar_path.exists() and jaspar_path.is_file()):
            raise FileNotFoundError(f"PWM JASPAR file not found. Looked here:\n  - {jaspar_path}")

        motifs = _parse_jaspar(jaspar_path)
        if self.motif_ids:
            keep = set(self.motif_ids)
            motifs = [m for m in motifs if m.motif_id in keep]
            if not motifs:
                raise ValueError(f"No motifs matched motif_ids in {jaspar_path}")

        sampling = dict(self.sampling or {})
        strategy = str(sampling.get("strategy", "stochastic"))
        n_sites = int(sampling.get("n_sites"))
        oversample_factor = int(sampling.get("oversample_factor", 10))
        max_candidates = sampling.get("max_candidates")
        max_seconds = sampling.get("max_seconds")
        threshold = sampling.get("score_threshold")
        percentile = sampling.get("score_percentile")
        length_policy = str(sampling.get("length_policy", "exact"))
        length_range = sampling.get("length_range")
        trim_window_length = sampling.get("trim_window_length")
        trim_window_strategy = sampling.get("trim_window_strategy", "max_info")

        entries = []
        all_rows = []
        for motif in motifs:
            selected = sample_pwm_sites(
                rng,
                motif,
                strategy=strategy,
                n_sites=n_sites,
                oversample_factor=oversample_factor,
                max_candidates=max_candidates,
                max_seconds=max_seconds,
                score_threshold=threshold,
                score_percentile=percentile,
                length_policy=length_policy,
                length_range=length_range,
                trim_window_length=trim_window_length,
                trim_window_strategy=str(trim_window_strategy),
            )
            for seq in selected:
                entries.append((motif.motif_id, seq, str(jaspar_path)))
                all_rows.append({"tf": motif.motif_id, "tfbs": seq, "source": str(jaspar_path)})

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
