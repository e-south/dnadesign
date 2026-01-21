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

from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from .base import BaseDataSource, resolve_path
from .pwm_artifact import load_artifact
from .pwm_sampling import sample_pwm_sites


@dataclass
class PWMArtifactSetDataSource(BaseDataSource):
    paths: List[str]
    cfg_path: Path
    sampling: dict
    overrides_by_motif_id: dict[str, dict] | None = None
    input_name: str = ""

    def load_data(self, *, rng=None, outputs_root: Path | None = None):
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
            motif_hash = hash_pwm_motif(
                motif_label=motif.motif_id,
                matrix=motif.matrix,
                background=motif.background,
                source_kind="pwm_artifact_set",
            )
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
            scoring_backend = str(sampling_cfg.get("scoring_backend", "densegen")).lower()
            pvalue_threshold = sampling_cfg.get("pvalue_threshold")
            pvalue_bins = sampling_cfg.get("pvalue_bins")
            mining = sampling_cfg.get("mining")
            bgfile = sampling_cfg.get("bgfile")
            selection_policy = str(sampling_cfg.get("selection_policy", "random_uniform"))
            keep_all_candidates_debug = bool(sampling_cfg.get("keep_all_candidates_debug", False))
            include_matched_sequence = bool(sampling_cfg.get("include_matched_sequence", False))
            bgfile_path: Path | None = None
            if bgfile is not None:
                bgfile_path = resolve_path(self.cfg_path, str(bgfile))
                if not (bgfile_path.exists() and bgfile_path.is_file()):
                    raise FileNotFoundError(f"PWM sampling bgfile not found. Looked here:\n  - {bgfile_path}")
            debug_output_dir: Path | None = None
            if keep_all_candidates_debug and outputs_root is not None:
                debug_output_dir = Path(outputs_root) / "candidates" / self.input_name
            return_meta = scoring_backend == "fimo"
            result = sample_pwm_sites(
                rng,
                motif,
                input_name=self.input_name,
                strategy=strategy,
                n_sites=n_sites,
                oversample_factor=oversample_factor,
                max_candidates=max_candidates,
                max_seconds=max_seconds,
                score_threshold=threshold,
                score_percentile=percentile,
                scoring_backend=scoring_backend,
                pvalue_threshold=pvalue_threshold,
                pvalue_bins=pvalue_bins,
                mining=mining,
                bgfile=bgfile_path,
                selection_policy=selection_policy,
                keep_all_candidates_debug=keep_all_candidates_debug,
                include_matched_sequence=include_matched_sequence,
                debug_output_dir=debug_output_dir,
                debug_label=f"{Path(path).stem}__{motif.motif_id}",
                length_policy=length_policy,
                length_range=length_range,
                trim_window_length=trim_window_length,
                trim_window_strategy=str(trim_window_strategy),
                return_metadata=return_meta,
            )
            if return_meta:
                selected, meta_by_seq = result  # type: ignore[misc]
            else:
                selected = result  # type: ignore[assignment]
                meta_by_seq = {}

            for seq in selected:
                entries.append((motif.motif_id, seq, str(path)))
                meta = meta_by_seq.get(seq, {}) if meta_by_seq else {}
                start = meta.get("fimo_start")
                stop = meta.get("fimo_stop")
                strand = meta.get("fimo_strand")
                tfbs_id = hash_tfbs_id(
                    motif_id=motif_hash,
                    sequence=seq,
                    scoring_backend=scoring_backend,
                    matched_start=int(start) if start is not None else None,
                    matched_stop=int(stop) if stop is not None else None,
                    matched_strand=str(strand) if strand is not None else None,
                )
                row = {
                    "tf": motif.motif_id,
                    "tfbs": seq,
                    "source": str(path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta:
                    row.update(meta)
                all_rows.append(row)

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
