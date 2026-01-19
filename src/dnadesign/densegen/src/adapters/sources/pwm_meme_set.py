"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_meme_set.py

PWM input source for multiple MEME files merged into a single TF pool.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dnadesign.cruncher.io.parsers.meme import MemeMotif, parse_meme_file

from .base import BaseDataSource, resolve_path
from .pwm_meme import _background_from_meta, _motif_to_pwm
from .pwm_sampling import sample_pwm_sites


def _filter_motifs(motifs: List[MemeMotif], keep: set[str]) -> list[MemeMotif]:
    if not keep:
        return motifs
    filtered: list[MemeMotif] = []
    for motif in motifs:
        cand = {motif.motif_id, motif.motif_name, motif.motif_label}
        cand = {str(x).strip().lower() for x in cand if x}
        if cand & keep:
            filtered.append(motif)
    return filtered


@dataclass
class PWMMemeSetDataSource(BaseDataSource):
    paths: List[str]
    cfg_path: Path
    motif_ids: Optional[List[str]]
    sampling: dict

    def load_data(self, *, rng=None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        resolved = [resolve_path(self.cfg_path, path) for path in self.paths]
        for path in resolved:
            if not (path.exists() and path.is_file()):
                raise FileNotFoundError(f"PWM MEME file not found. Looked here:\n  - {path}")

        keep = None
        if self.motif_ids:
            keep = {m.strip().lower() for m in self.motif_ids if m}

        motifs_payload: list[tuple[MemeMotif, dict[str, float], Path]] = []
        available_ids: set[str] = set()
        for path in resolved:
            result = parse_meme_file(path)
            background = _background_from_meta(result.meta)
            motifs = result.motifs
            available_ids.update({m.motif_id for m in motifs if m.motif_id})
            motifs = _filter_motifs(motifs, keep or set())
            for motif in motifs:
                motifs_payload.append((motif, background, path))

        if keep and not motifs_payload:
            available = ", ".join(sorted(available_ids)) if available_ids else "-"
            raise ValueError(f"No motifs matched motif_ids across MEME inputs. Available: {available}")

        motif_ids = [m.motif_id for m, _, _ in motifs_payload if m.motif_id]
        if len(set(motif_ids)) != len(motif_ids):
            raise ValueError("Duplicate motif_id values found across pwm_meme_set inputs.")

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
        for motif, background, path in motifs_payload:
            pwm = _motif_to_pwm(motif, background)
            selected = sample_pwm_sites(
                rng,
                pwm,
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
                entries.append((pwm.motif_id, seq, str(path)))
                all_rows.append({"tf": pwm.motif_id, "tfbs": seq, "source": str(path)})

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
