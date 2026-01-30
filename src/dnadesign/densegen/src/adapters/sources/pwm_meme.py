"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_meme.py

PWM input source (MEME format) with explicit Stage-A sampling policies.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dnadesign.cruncher.io.parsers.meme import MemeMotif, parse_meme_file

from ...config import PWMSamplingConfig
from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_sampling import sample_pwm_sites, sampling_kwargs_from_config
from .stage_a_sampling_utils import normalize_background
from .stage_a_types import PWMMotif


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
    sampling: PWMSamplingConfig
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
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

        sampling_kwargs = sampling_kwargs_from_config(self.sampling)
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

        entries = []
        all_rows = []
        summaries = []
        for motif in motifs:
            pwm = _motif_to_pwm(motif, background)
            motif_hash = hash_pwm_motif(
                motif_label=pwm.motif_id,
                matrix=pwm.matrix,
                background=pwm.background,
                source_kind="pwm_meme",
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
                debug_label=f"{meme_path.stem}__{pwm.motif_id}",
                length_policy=sampling_kwargs["length_policy"],
                length_range=sampling_kwargs["length_range"],
                trim_window_length=sampling_kwargs["trim_window_length"],
                trim_window_strategy=str(sampling_kwargs["trim_window_strategy"]),
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
                entries.append((pwm.motif_id, seq, str(meme_path)))
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
                    "source": str(meme_path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta is not None:
                    row.update(meta.to_dict())
                all_rows.append(row)

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df, summaries
