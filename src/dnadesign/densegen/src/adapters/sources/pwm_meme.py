"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_meme.py

PWM input source (MEME format) with explicit sampling policies.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites


def _parse_background(line: str) -> dict[str, float]:
    parts = line.strip().split()
    if len(parts) % 2 != 0:
        raise ValueError("Invalid background frequency line (expected pairs).")
    bg: dict[str, float] = {}
    for i in range(0, len(parts), 2):
        base = parts[i].upper()
        try:
            val = float(parts[i + 1])
        except ValueError as e:
            raise ValueError(f"Invalid background frequency: {parts[i + 1]}") from e
        bg[base] = val
    for base in ("A", "C", "G", "T"):
        if base not in bg:
            raise ValueError("Background frequencies must include A/C/G/T.")
    return bg


def _parse_meme(path: Path) -> List[PWMMotif]:
    text = path.read_text().splitlines()
    alphabet_ok = False
    background: Optional[dict[str, float]] = None
    motifs: List[PWMMotif] = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("ALPHABET="):
            alphabet = line.split("=", 1)[1].strip().upper()
            if alphabet != "ACGT":
                raise ValueError(f"MEME alphabet must be ACGT, got {alphabet!r}")
            alphabet_ok = True
            i += 1
            continue
        if line.startswith("Background letter frequencies"):
            if i + 1 >= len(text):
                raise ValueError("MEME background frequencies are missing values.")
            background = _parse_background(text[i + 1])
            i += 2
            continue
        if line.startswith("MOTIF"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("MOTIF line missing motif id.")
            motif_id = parts[1].strip()
            i += 1
            # find letter-probability matrix header
            while i < len(text) and "letter-probability matrix" not in text[i]:
                i += 1
            if i >= len(text):
                raise ValueError(f"Missing letter-probability matrix for motif {motif_id}")
            header = text[i]
            # parse width
            tokens = header.replace("=", " ").split()
            if "w" not in tokens:
                raise ValueError(f"PWM header missing width for motif {motif_id}")
            w_idx = tokens.index("w") + 1
            try:
                width = int(float(tokens[w_idx]))
            except Exception as e:
                raise ValueError(f"Invalid PWM width for motif {motif_id}") from e
            i += 1
            rows: List[dict[str, float]] = []
            for _ in range(width):
                if i >= len(text):
                    raise ValueError(f"PWM rows truncated for motif {motif_id}")
                row = text[i].strip()
                i += 1
                if not row:
                    raise ValueError(f"Empty PWM row for motif {motif_id}")
                parts = row.split()
                if len(parts) < 4:
                    raise ValueError(f"PWM row must have 4 columns for motif {motif_id}")
                try:
                    probs = [float(p) for p in parts[:4]]
                except ValueError as e:
                    raise ValueError(f"Invalid PWM probabilities for motif {motif_id}") from e
                row_map = {"A": probs[0], "C": probs[1], "G": probs[2], "T": probs[3]}
                s = sum(row_map.values())
                if s <= 0:
                    raise ValueError(f"PWM row sums to 0 for motif {motif_id}")
                row_map = {k: v / s for k, v in row_map.items()}
                rows.append(row_map)
            motifs.append(PWMMotif(motif_id=motif_id, matrix=rows, background=normalize_background(background)))
            continue
        i += 1

    if not alphabet_ok:
        raise ValueError("MEME file missing ALPHABET=ACGT header.")
    if not motifs:
        raise ValueError("No motifs found in MEME file.")
    return motifs


@dataclass
class PWMMemeDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_ids: Optional[List[str]]
    sampling: dict

    def load_data(self, *, rng=None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        meme_path = resolve_path(self.cfg_path, self.path)
        if not (meme_path.exists() and meme_path.is_file()):
            raise FileNotFoundError(f"PWM MEME file not found. Looked here:\n  - {meme_path}")

        motifs = _parse_meme(meme_path)
        if self.motif_ids:
            keep = set(self.motif_ids)
            motifs = [m for m in motifs if m.motif_id in keep]
            if not motifs:
                raise ValueError(f"No motifs matched motif_ids in {meme_path}")

        sampling = dict(self.sampling or {})
        strategy = str(sampling.get("strategy", "stochastic"))
        n_sites = int(sampling.get("n_sites"))
        oversample_factor = int(sampling.get("oversample_factor", 10))
        threshold = sampling.get("score_threshold")
        percentile = sampling.get("score_percentile")

        entries = []
        all_rows = []
        for motif in motifs:
            selected = sample_pwm_sites(
                rng,
                motif,
                strategy=strategy,
                n_sites=n_sites,
                oversample_factor=oversample_factor,
                score_threshold=threshold,
                score_percentile=percentile,
            )

            for seq in selected:
                entries.append((motif.motif_id, seq, str(meme_path)))
                all_rows.append({"tf": motif.motif_id, "tfbs": seq})

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
