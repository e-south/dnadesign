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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ...config import PWMSamplingConfig
from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_artifact import load_artifact
from .pwm_sampling import sample_pwm_sites, sampling_kwargs_from_config, validate_mmr_core_length
from .stage_a.stage_a_progress import StageAProgressManager


@dataclass
class PWMArtifactSetDataSource(BaseDataSource):
    paths: List[str]
    cfg_path: Path
    sampling: PWMSamplingConfig
    overrides_by_motif_id: dict[str, PWMSamplingConfig] | None = None
    input_name: str = ""

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")

        resolved = [resolve_path(self.cfg_path, path) for path in self.paths]
        for path in resolved:
            if not (path.exists() and path.is_file()):
                raise FileNotFoundError(f"PWM artifact not found. Looked here:\n  - {path}")

        motifs = [load_artifact(path) for path in resolved]
        motif_ids = [m.motif_id for m in motifs]
        if len(set(motif_ids)) != len(motif_ids):
            raise ValueError("Duplicate motif_id values found across pwm_artifact_set paths.")

        sampling = self.sampling
        # Alignment (5): per-motif sampling overrides for artifact sets.
        overrides = dict(self.overrides_by_motif_id or {})
        if overrides:
            unknown = [key for key in overrides if key not in motif_ids]
            if unknown:
                preview = ", ".join(unknown[:10])
                raise ValueError(f"pwm_artifact_set.overrides_by_motif_id contains unknown motif_id: {preview}")

        sampling_kwargs_by_motif: dict[str, dict] = {}
        for motif in motifs:
            sampling_cfg = overrides.get(motif.motif_id, sampling)
            sampling_kwargs = sampling_kwargs_from_config(sampling_cfg)
            selection_cfg = sampling_kwargs.get("selection")
            selection_policy = str(getattr(selection_cfg, "policy", None) or "top_score")
            validate_mmr_core_length(
                motif_id=motif.motif_id,
                motif_width=len(motif.matrix),
                selection_policy=selection_policy,
                length_policy=str(sampling_kwargs.get("length_policy") or "exact"),
                length_range=sampling_kwargs.get("length_range"),
                trim_window_length=sampling_kwargs.get("trim_window_length"),
            )
            sampling_kwargs_by_motif[motif.motif_id] = sampling_kwargs

        progress_manager = StageAProgressManager(stream=sys.stdout)
        entries = []
        all_rows = []
        summaries = []
        for motif, path in zip(motifs, resolved):
            motif_hash = hash_pwm_motif(
                motif_label=motif.motif_id,
                matrix=motif.matrix,
                background=motif.background,
                source_kind="pwm_artifact_set",
            )
            sampling_kwargs = sampling_kwargs_by_motif[motif.motif_id]
            bgfile = sampling_kwargs.get("bgfile")
            keep_all_candidates_debug = bool(sampling_kwargs.get("keep_all_candidates_debug", False))
            bgfile_path: Path | None = None
            if bgfile is not None:
                bgfile_path = resolve_path(self.cfg_path, str(bgfile))
                if not (bgfile_path.exists() and bgfile_path.is_file()):
                    raise FileNotFoundError(f"Stage-A PWM sampling bgfile not found. Looked here:\n  - {bgfile_path}")
            debug_output_dir: Path | None = None
            if keep_all_candidates_debug:
                if outputs_root is None:
                    raise ValueError("keep_all_candidates_debug requires outputs_root to be set.")
                if run_id is None:
                    raise ValueError("keep_all_candidates_debug requires run_id to be set.")
                debug_output_dir = candidates_root(Path(outputs_root), run_id) / self.input_name
            return_meta = True
            result = sample_pwm_sites(
                rng,
                motif,
                input_name=self.input_name,
                motif_hash=motif_hash,
                run_id=run_id,
                mining=sampling_kwargs["mining"],
                bgfile=bgfile_path,
                keep_all_candidates_debug=keep_all_candidates_debug,
                include_matched_sequence=sampling_kwargs["include_matched_sequence"],
                uniqueness_key=sampling_kwargs["uniqueness_key"],
                selection=sampling_kwargs["selection"],
                debug_output_dir=debug_output_dir,
                debug_label=f"{Path(path).stem}__{motif.motif_id}",
                length_policy=sampling_kwargs["length_policy"],
                length_range=sampling_kwargs["length_range"],
                trim_window_length=sampling_kwargs["trim_window_length"],
                trim_window_strategy=str(sampling_kwargs["trim_window_strategy"]),
                progress_manager=progress_manager,
                return_metadata=return_meta,
                return_summary=True,
                strategy=str(sampling_kwargs["strategy"]),
                n_sites=int(sampling_kwargs["n_sites"]),
            )
            if return_meta:
                selected, meta_by_seq, summary = result  # type: ignore[misc]
            else:
                selected, summary = result  # type: ignore[assignment]
                meta_by_seq = {}
            if summary is not None:
                summaries.append(summary)

            for seq in selected:
                entries.append((motif.motif_id, seq, str(path)))
                meta = meta_by_seq[seq] if return_meta else None
                start = meta.fimo_start if meta is not None else None
                stop = meta.fimo_stop if meta is not None else None
                strand = meta.fimo_strand if meta is not None else None
                tfbs_id = hash_tfbs_id(
                    motif_id=motif_hash,
                    sequence=seq,
                    scoring_backend="fimo",
                    matched_start=int(start) if start is not None else None,
                    matched_stop=int(stop) if stop is not None else None,
                    matched_strand=str(strand) if strand is not None else None,
                )
                row = {
                    "tf": motif.motif_id,
                    "tfbs": seq,
                    "regulator_id": motif.motif_id,
                    "tfbs_sequence": seq,
                    "source": str(path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta is not None:
                    row.update(meta.to_dict())
                all_rows.append(row)

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df, summaries
