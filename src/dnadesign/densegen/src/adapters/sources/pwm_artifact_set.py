"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_artifact_set.py

PWM input source for a set of per-motif JSON artifacts.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import BaseDataSource, resolve_path
from .pwm_artifact import load_artifact
from .pwm_sampling import sample_pwm_sites


@dataclass
class PWMArtifactSetDataSource(BaseDataSource):
    paths: List[str]
    cfg_path: Path
    sampling: dict
    overrides_by_motif_id: dict[str, dict] | None = None

    def load_data(self, *, rng=None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")

        resolved = [resolve_path(self.cfg_path, path) for path in self.paths]
        for path in resolved:
            if not (path.exists() and path.is_file()):
                raise FileNotFoundError(f"PWM artifact not found. Looked here:\n  - {path}")

        motifs = [load_artifact(path) for path in resolved]
        motif_ids = [m.motif_id for m in motifs]
        if len(set(motif_ids)) != len(motif_ids):
            raise ValueError("Duplicate motif_id values found across pwm_artifact_set paths.")

        sampling = dict(self.sampling or {})
        # Alignment (5): per-motif sampling overrides for artifact sets.
        overrides = dict(self.overrides_by_motif_id or {})
        if overrides:
            unknown = [key for key in overrides if key not in motif_ids]
            if unknown:
                preview = ", ".join(unknown[:10])
                raise ValueError(f"pwm_artifact_set.overrides_by_motif_id contains unknown motif_id: {preview}")

        entries = []
        all_rows = []
        for motif, path in zip(motifs, resolved):
            sampling_cfg = sampling
            override = overrides.get(motif.motif_id)
            if override:
                sampling_cfg = {**sampling, **dict(override)}
            strategy = str(sampling_cfg.get("strategy", "stochastic"))
            n_sites = int(sampling_cfg.get("n_sites"))
            oversample_factor = int(sampling_cfg.get("oversample_factor", 10))
            max_candidates = sampling_cfg.get("max_candidates")
            max_seconds = sampling_cfg.get("max_seconds")
            threshold = sampling_cfg.get("score_threshold")
            percentile = sampling_cfg.get("score_percentile")
            length_policy = str(sampling_cfg.get("length_policy", "exact"))
            length_range = sampling_cfg.get("length_range")
            trim_window_length = sampling_cfg.get("trim_window_length")
            trim_window_strategy = sampling_cfg.get("trim_window_strategy", "max_info")
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
                entries.append((motif.motif_id, seq, str(path)))
                all_rows.append({"tf": motif.motif_id, "tfbs": seq, "source": str(path)})

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
