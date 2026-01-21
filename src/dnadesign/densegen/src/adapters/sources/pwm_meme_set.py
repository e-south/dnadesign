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

from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
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
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
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
        scoring_backend = str(sampling.get("scoring_backend", "densegen")).lower()
        pvalue_threshold = sampling.get("pvalue_threshold")
        pvalue_bins = sampling.get("pvalue_bins")
        mining = sampling.get("mining")
        bgfile = sampling.get("bgfile")
        selection_policy = str(sampling.get("selection_policy", "random_uniform"))
        keep_all_candidates_debug = bool(sampling.get("keep_all_candidates_debug", False))
        include_matched_sequence = bool(sampling.get("include_matched_sequence", False))
        bgfile_path: Path | None = None
        if bgfile is not None:
            bgfile_path = resolve_path(self.cfg_path, str(bgfile))
            if not (bgfile_path.exists() and bgfile_path.is_file()):
                raise FileNotFoundError(f"PWM sampling bgfile not found. Looked here:\n  - {bgfile_path}")
        debug_output_dir: Path | None = None
        if keep_all_candidates_debug:
            if outputs_root is None:
                raise ValueError("keep_all_candidates_debug requires outputs_root to be set.")
            if run_id is None:
                raise ValueError("keep_all_candidates_debug requires run_id to be set.")
            debug_output_dir = candidates_root(Path(outputs_root), run_id) / self.input_name

        entries = []
        all_rows = []
        for motif, background, path in motifs_payload:
            pwm = _motif_to_pwm(motif, background)
            motif_hash = hash_pwm_motif(
                motif_label=pwm.motif_id,
                matrix=pwm.matrix,
                background=pwm.background,
                source_kind="pwm_meme_set",
            )
            return_meta = scoring_backend == "fimo"
            result = sample_pwm_sites(
                rng,
                pwm,
                input_name=self.input_name,
                motif_hash=motif_hash,
                run_id=run_id,
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
                debug_label=f"{Path(path).stem}__{pwm.motif_id}",
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
                entries.append((pwm.motif_id, seq, str(path)))
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
                    "tf": pwm.motif_id,
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
