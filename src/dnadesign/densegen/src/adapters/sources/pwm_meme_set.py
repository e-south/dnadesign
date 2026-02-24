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

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dnadesign.cruncher.meme import MemeMotif, parse_meme_file

from ...config import PWMSamplingConfig
from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from ...core.stage_a.stage_a_progress import StageAProgressManager
from .base import BaseDataSource, resolve_path
from .pwm_meme import _background_from_meta, _motif_to_pwm
from .pwm_sampling import (
    enforce_cross_regulator_core_collisions,
    sample_pwm_sites,
    sampling_kwargs_from_config,
    validate_mmr_core_length,
)


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
    sampling: PWMSamplingConfig
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
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

        sampling_kwargs = sampling_kwargs_from_config(self.sampling)
        collision_mode_raw = sampling_kwargs.get("cross_regulator_core_collisions")
        if not isinstance(collision_mode_raw, str) or not collision_mode_raw.strip():
            raise ValueError("pwm.sampling.uniqueness.cross_regulator_core_collisions is required.")
        collision_mode = collision_mode_raw.strip()
        selection_cfg = sampling_kwargs.get("selection")
        if selection_cfg is None or not hasattr(selection_cfg, "policy"):
            raise ValueError("pwm.sampling.selection is required.")
        selection_policy = str(selection_cfg.policy).strip()
        if not selection_policy:
            raise ValueError("pwm.sampling.selection.policy must be a non-empty string.")
        length_policy_raw = sampling_kwargs.get("length_policy")
        if not isinstance(length_policy_raw, str) or not length_policy_raw.strip():
            raise ValueError("pwm.sampling.length.policy is required.")
        length_policy = length_policy_raw.strip()
        for motif, _, _ in motifs_payload:
            validate_mmr_core_length(
                motif_id=str(motif.motif_id),
                motif_width=len(motif.prob_matrix),
                selection_policy=selection_policy,
                length_policy=length_policy,
                length_range=sampling_kwargs["length_range"],
                trim_window_length=sampling_kwargs["trim_window_length"],
            )
        bgfile = sampling_kwargs["bgfile"]
        keep_all_candidates_debug = bool(sampling_kwargs["keep_all_candidates_debug"])
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

        progress_manager = StageAProgressManager(stream=sys.stdout)
        entries = []
        all_rows = []
        summaries = []
        for motif, background, path in motifs_payload:
            pwm = _motif_to_pwm(motif, background)
            motif_hash = hash_pwm_motif(
                motif_label=pwm.motif_id,
                matrix=pwm.matrix,
                background=pwm.background,
                source_kind="pwm_meme_set",
            )
            return_meta = True
            result = sample_pwm_sites(
                rng,
                pwm,
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
                debug_label=f"{Path(path).stem}__{pwm.motif_id}",
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
                entries.append((pwm.motif_id, seq, str(path)))
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
                    "tf": pwm.motif_id,
                    "tfbs": seq,
                    "regulator_id": pwm.motif_id,
                    "tfbs_sequence": seq,
                    "source": str(path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta is not None:
                    row.update(meta.to_dict())
                all_rows.append(row)

        import pandas as pd

        enforce_cross_regulator_core_collisions(
            all_rows,
            mode=collision_mode,
            input_name=self.input_name,
            source_kind="pwm_meme_set",
        )
        df = pd.DataFrame(all_rows)
        return entries, df, summaries
