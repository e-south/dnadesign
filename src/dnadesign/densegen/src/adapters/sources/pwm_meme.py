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

from dnadesign.cruncher.io.parsers.meme import MemeMotif, parse_meme_file

from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites


def _background_from_meta(meta) -> dict[str, float]:
    freqs = getattr(meta, "background_freqs", None)
    if freqs is None:
        return normalize_background(None)
    try:
        a, c, g, t = freqs
    except Exception as exc:
        raise ValueError("MEME background frequencies must include A/C/G/T.") from exc
    return normalize_background({"A": float(a), "C": float(c), "G": float(g), "T": float(t)})


def _motif_to_pwm(motif: MemeMotif, background: dict[str, float]) -> PWMMotif:
    rows: List[dict[str, float]] = []
    for row in motif.prob_matrix:
        if len(row) != 4:
            raise ValueError(f"MEME motif {motif.motif_id} row must have 4 probabilities.")
        rows.append({"A": float(row[0]), "C": float(row[1]), "G": float(row[2]), "T": float(row[3])})
    log_odds = None
    if motif.log_odds_matrix is not None:
        log_odds = []
        for row in motif.log_odds_matrix:
            if len(row) != 4:
                raise ValueError(f"MEME motif {motif.motif_id} log-odds row must have 4 values.")
            log_odds.append({"A": float(row[0]), "C": float(row[1]), "G": float(row[2]), "T": float(row[3])})
    return PWMMotif(motif_id=str(motif.motif_id), matrix=rows, background=background, log_odds=log_odds)


@dataclass
class PWMMemeDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_ids: Optional[List[str]]
    sampling: dict
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        meme_path = resolve_path(self.cfg_path, self.path)
        if not (meme_path.exists() and meme_path.is_file()):
            raise FileNotFoundError(f"PWM MEME file not found. Looked here:\n  - {meme_path}")

        result = parse_meme_file(meme_path)
        background = _background_from_meta(result.meta)
        motifs = result.motifs
        if self.motif_ids:
            keep = {m.strip().lower() for m in self.motif_ids if m}
            filtered: list[MemeMotif] = []
            for motif in motifs:
                cand = {motif.motif_id, motif.motif_name, motif.motif_label}
                cand = {str(x).strip().lower() for x in cand if x}
                if cand & keep:
                    filtered.append(motif)
            motifs = filtered
            if not motifs:
                available = ", ".join(sorted({m.motif_id for m in result.motifs if m.motif_id}))
                raise ValueError(f"No motifs matched motif_ids in {meme_path}. Available: {available}")

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
        for motif in motifs:
            pwm = _motif_to_pwm(motif, background)
            motif_hash = hash_pwm_motif(
                motif_label=pwm.motif_id,
                matrix=pwm.matrix,
                background=pwm.background,
                source_kind="pwm_meme",
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
                debug_label=f"{meme_path.stem}__{pwm.motif_id}",
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
                entries.append((pwm.motif_id, seq, str(meme_path)))
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
                    "source": str(meme_path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta:
                    row.update(meta)
                all_rows.append(row)

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df
