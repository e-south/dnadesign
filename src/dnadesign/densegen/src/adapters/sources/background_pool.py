"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/background_pool.py

Stage-A background pool generator with optional FIMO-based negative selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dnadesign.cruncher.io.parsers.meme import parse_meme_file

from ...config import BackgroundPoolSamplingConfig, resolve_relative_path
from ...core.artifacts.ids import hash_label_motif, hash_tfbs_id
from ...core.sequence_constraints.kmers import reverse_complement
from ...core.sequence_constraints.sampler import ConstrainedSequenceError, generate_constrained_sequence
from ...utils.sequence_utils import gc_fraction
from .base import BaseDataSource
from .pwm_artifact import load_artifact as load_pwm_artifact
from .pwm_fimo import (
    aggregate_best_hits,
    build_candidate_records,
    run_fimo,
    write_candidates_fasta,
    write_minimal_meme_motif,
)
from .pwm_jaspar import _parse_jaspar
from .pwm_meme import _background_from_meta, _motif_to_pwm
from .stage_a.stage_a_progress import _BackgroundSamplingProgress
from .stage_a.stage_a_sampling_utils import (
    _background_cdf,
    _pwm_theoretical_max_score,
    _sample_background_batch,
    build_log_odds,
)
from .stage_a.stage_a_types import PWMMotif

log = logging.getLogger(__name__)

BACKGROUND_FIMO_PVALUE_THRESH = 1e-4


def _parse_pwm_matrix_csv(path: Path, *, motif_id: str, columns: dict[str, str]) -> PWMMotif:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {k.upper(): v for k, v in columns.items()}
    required = {b: cols.get(b, b) for b in ("A", "C", "G", "T")}
    missing = [name for name in required.values() if name not in df.columns]
    if missing:
        raise ValueError(f"PWM matrix CSV missing columns: {missing}")

    matrix = []
    for i, row in df.iterrows():
        vals = {}
        for base, col in required.items():
            val = row[col]
            try:
                num = float(val)
            except Exception as exc:
                raise ValueError(f"PWM matrix CSV row {i} has non-numeric {base} value: {val}") from exc
            if num < 0:
                raise ValueError(f"PWM matrix CSV row {i} has negative {base} value: {num}")
            vals[base] = num
        total = sum(vals.values())
        if total <= 0:
            raise ValueError(f"PWM matrix CSV row {i} sums to 0.")
        matrix.append({b: v / total for b, v in vals.items()})
    return PWMMotif(
        motif_id=str(motif_id).strip(),
        matrix=matrix,
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )


def _load_pwms_from_input(cfg, cfg_path: Path) -> list[PWMMotif]:
    input_type = str(getattr(cfg, "type", ""))
    if input_type == "pwm_artifact":
        path = resolve_relative_path(cfg_path, getattr(cfg, "path"))
        return [load_pwm_artifact(path)]
    if input_type == "pwm_artifact_set":
        motifs: list[PWMMotif] = []
        for raw in getattr(cfg, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            motifs.append(load_pwm_artifact(path))
        return motifs
    if input_type == "pwm_meme":
        path = resolve_relative_path(cfg_path, getattr(cfg, "path"))
        result = parse_meme_file(path)
        background = _background_from_meta(result.meta)
        motifs = result.motifs
        motif_ids = getattr(cfg, "motif_ids", None)
        if motif_ids:
            keep = {m.strip().lower() for m in motif_ids if m}
            filtered = []
            for motif in motifs:
                cand = {motif.motif_id, motif.motif_name, motif.motif_label}
                cand = {str(x).strip().lower() for x in cand if x}
                if cand & keep:
                    filtered.append(motif)
            motifs = filtered
        return [_motif_to_pwm(motif, background) for motif in motifs]
    if input_type == "pwm_meme_set":
        motifs: list[PWMMotif] = []
        motif_ids = getattr(cfg, "motif_ids", None)
        for raw in getattr(cfg, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            result = parse_meme_file(path)
            background = _background_from_meta(result.meta)
            parsed = result.motifs
            if motif_ids:
                keep = {m.strip().lower() for m in motif_ids if m}
                filtered = []
                for motif in parsed:
                    cand = {motif.motif_id, motif.motif_name, motif.motif_label}
                    cand = {str(x).strip().lower() for x in cand if x}
                    if cand & keep:
                        filtered.append(motif)
                parsed = filtered
            motifs.extend(_motif_to_pwm(motif, background) for motif in parsed)
        return motifs
    if input_type == "pwm_jaspar":
        path = resolve_relative_path(cfg_path, getattr(cfg, "path"))
        motifs = _parse_jaspar(path)
        motif_ids = getattr(cfg, "motif_ids", None)
        if motif_ids:
            keep = {m.strip().lower() for m in motif_ids if m}
            motifs = [m for m in motifs if str(m.motif_id).strip().lower() in keep]
        return motifs
    if input_type == "pwm_matrix_csv":
        path = resolve_relative_path(cfg_path, getattr(cfg, "path"))
        motif_id = getattr(cfg, "motif_id", None)
        if not motif_id:
            raise ValueError("pwm_matrix_csv.motif_id must be set for background_pool FIMO exclusion.")
        columns = getattr(cfg, "columns", None)
        columns = columns.model_dump() if hasattr(columns, "model_dump") else dict(columns or {})
        return [_parse_pwm_matrix_csv(path, motif_id=str(motif_id), columns=columns)]
    raise ValueError(f"Unsupported PWM input type for background_pool: {input_type}")


def _filter_forbidden_kmers(sequences: Iterable[str], kmers: list[str]) -> list[str]:
    if not kmers:
        return list(sequences)
    filtered: list[str] = []
    for seq in sequences:
        if any(kmer in seq for kmer in kmers):
            continue
        filtered.append(seq)
    return filtered


def _expand_forbidden_kmers(
    kmers: list[str],
    *,
    include_reverse_complements: bool,
    strands: str,
) -> list[str]:
    if not kmers:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for motif in kmers:
        seq = str(motif).strip().upper()
        if not seq:
            continue
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
        if include_reverse_complements or str(strands).strip().lower() == "both":
            rc = reverse_complement(seq)
            if rc not in seen:
                seen.add(rc)
                out.append(rc)
    return out


def _filter_gc(sequences: Iterable[str], *, gc_min: float | None, gc_max: float | None) -> list[str]:
    if gc_min is None and gc_max is None:
        return list(sequences)
    filtered: list[str] = []
    for seq in sequences:
        gc = gc_fraction(seq)
        if gc_min is not None and gc < gc_min:
            continue
        if gc_max is not None and gc > gc_max:
            continue
        filtered.append(seq)
    return filtered


def _resolve_lengths(rng: np.random.Generator, *, count: int, length_cfg) -> list[int]:
    policy = getattr(length_cfg, "policy", "range")
    length_range = getattr(length_cfg, "range", None)
    exact = getattr(length_cfg, "exact", None)
    if policy == "exact":
        if exact is None:
            raise ValueError("background_pool.sampling.length.exact is required when policy=exact")
        return [int(exact)] * int(count)
    if length_range is None:
        raise ValueError("background_pool.sampling.length.range is required when policy=range")
    lo, hi = int(length_range[0]), int(length_range[1])
    if lo == hi:
        return [lo] * int(count)
    return rng.integers(low=lo, high=hi + 1, size=int(count)).tolist()


def _run_fimo_exclusion(
    *,
    motifs: list[PWMMotif],
    sequences: list[str],
    allow_zero_hit_only: bool,
    max_score_norm: float | None,
) -> tuple[list[str], dict[str, float | None]]:
    if not sequences:
        return [], {}
    records = build_candidate_records("bg", sequences)
    record_index = {rec_id: idx for idx, (rec_id, _seq) in enumerate(records)}
    max_scores: list[float | None] = [None] * len(sequences)
    max_norms: list[float | None] = [None] * len(sequences)

    with tempfile.TemporaryDirectory(prefix="densegen-bg-") as tmpdir:
        tmp_path = Path(tmpdir)
        fasta_path = tmp_path / "candidates.fasta"
        write_candidates_fasta(records, fasta_path)

        motif_meta: list[tuple[PWMMotif, Path, float | None]] = []
        for idx, motif in enumerate(motifs):
            meme_path = tmp_path / f"motif_{idx}.meme"
            write_minimal_meme_motif(motif, meme_path)
            log_odds = motif.log_odds or build_log_odds(motif.matrix, motif.background)
            max_score = _pwm_theoretical_max_score(log_odds)
            if max_score <= 0 or not np.isfinite(max_score):
                raise ValueError(f"PWM motif '{motif.motif_id}' has invalid theoretical max score: {max_score}")
            motif_meta.append((motif, meme_path, float(max_score)))

        for motif, meme_path, max_score in motif_meta:
            rows, _ = run_fimo(
                meme_motif_path=meme_path,
                fasta_path=fasta_path,
                thresh=BACKGROUND_FIMO_PVALUE_THRESH,
                norc=False,
                include_matched_sequence=False,
                return_tsv=False,
            )
            best_hits = aggregate_best_hits(rows, allowed_strands={"+", "-"})
            for rec_id, hit in best_hits.items():
                idx = record_index.get(rec_id)
                if idx is None:
                    continue
                score = float(hit.score)
                prev = max_scores[idx]
                if prev is None or score > prev:
                    max_scores[idx] = score
                if max_score_norm is not None:
                    norm = score / max_score if max_score else float("-inf")
                    prev_norm = max_norms[idx]
                    if prev_norm is None or norm > prev_norm:
                        max_norms[idx] = norm

    accepted: list[str] = []
    score_map: dict[str, float | None] = {}
    for seq, score, norm in zip(sequences, max_scores, max_norms):
        if allow_zero_hit_only:
            if score is not None:
                continue
        else:
            if norm is not None and max_score_norm is not None and norm > max_score_norm:
                continue
        accepted.append(seq)
        score_map[seq] = score
    return accepted, score_map


@dataclass
class BackgroundPoolDataSource(BaseDataSource):
    cfg_path: Path
    sampling: BackgroundPoolSamplingConfig
    input_name: str
    pwm_inputs: list[object]

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("background_pool sampling requires an RNG; pass the pipeline RNG explicitly.")
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        target = int(self.sampling.n_sites)
        mining = self.sampling.mining
        budget = mining.budget
        max_candidates = int(budget.candidates)
        batch_size = int(mining.batch_size)
        length_cfg = self.sampling.length
        filters = self.sampling.filters
        gc_cfg = self.sampling.gc or None
        gc_min = float(gc_cfg.min) if gc_cfg and gc_cfg.min is not None else None
        gc_max = float(gc_cfg.max) if gc_cfg and gc_cfg.max is not None else None
        forbid_kmers = list(filters.forbid_kmers or [])
        forbid_kmers_expanded = _expand_forbidden_kmers(
            forbid_kmers,
            include_reverse_complements=bool(getattr(filters, "include_reverse_complements", False)),
            strands=str(getattr(filters, "strands", "forward")),
        )

        fimo_cfg = filters.fimo_exclude
        allow_zero_hit_only = False
        max_score_norm = None
        motifs: list[PWMMotif] = []
        if fimo_cfg is not None:
            allow_zero_hit_only = bool(fimo_cfg.allow_zero_hit_only)
            max_score_norm = fimo_cfg.max_score_norm
            if not self.pwm_inputs:
                raise ValueError(
                    "background_pool.sampling.filters.fimo_exclude.pwms_input requires at least one PWM input."
                )
            for pwm_input in self.pwm_inputs:
                motifs.extend(_load_pwms_from_input(pwm_input, self.cfg_path))
            if not motifs:
                raise ValueError("background_pool.fimo_exclude did not resolve any PWM motifs.")

        background_cdf = _background_cdf({"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
        accepted: list[str] = []
        seen: set[str] = set()
        best_scores: dict[str, float | None] = {}
        generated = 0
        batch_index = 0
        reject_fimo = 0
        reject_kmer = 0
        reject_gc = 0
        reject_dup = 0
        batch_total = int(np.ceil(max_candidates / batch_size)) if batch_size > 0 else None
        progress = _BackgroundSamplingProgress(
            input_name=self.input_name,
            target=max_candidates,
            accepted_target=target,
            stream=sys.stdout,
        )
        if fimo_cfg is not None:
            progress.set_phase("fimo")
        progress.update(
            generated=0,
            accepted=0,
            batch_index=0,
            batch_total=batch_total,
            reject_fimo=0,
            reject_kmer=0,
            reject_gc=0,
            reject_dup=0,
            force=True,
        )

        try:
            while len(accepted) < target and generated < max_candidates:
                batch_index += 1
                remaining = max_candidates - generated
                batch = min(batch_size, remaining)
                lengths = _resolve_lengths(rng, count=batch, length_cfg=length_cfg)
                if forbid_kmers_expanded:
                    sequences: list[str] = []
                    for length in lengths:
                        try:
                            seq = generate_constrained_sequence(
                                length=int(length),
                                gc_min=float(gc_min) if gc_min is not None else 0.0,
                                gc_max=float(gc_max) if gc_max is not None else 1.0,
                                forbid_kmers=forbid_kmers_expanded,
                                left_context="",
                                right_context="",
                                rng=rng,
                            )
                        except ConstrainedSequenceError as exc:
                            raise ValueError(
                                "background_pool constrained sampling is infeasible for "
                                f"length={int(length)}, gc_min={gc_min}, gc_max={gc_max}, "
                                f"forbidden_kmers={len(forbid_kmers_expanded)}."
                            ) from exc
                        sequences.append(seq)
                    generated += batch
                    if len(_filter_forbidden_kmers(sequences, forbid_kmers_expanded)) != len(sequences):
                        raise RuntimeError("background_pool constrained sampling produced forbidden kmer sequences.")
                    if len(_filter_gc(sequences, gc_min=gc_min, gc_max=gc_max)) != len(sequences):
                        raise RuntimeError("background_pool constrained sampling produced out-of-range GC sequences.")
                else:
                    length_groups: dict[int, list[int]] = {}
                    for idx, length in enumerate(lengths):
                        length_groups.setdefault(int(length), []).append(idx)
                    sequences = [""] * batch
                    for length, indices in length_groups.items():
                        seqs = _sample_background_batch(rng, background_cdf, count=len(indices), length=length)
                        for idx, seq in zip(indices, seqs):
                            sequences[idx] = seq
                    generated += batch
                    before = len(sequences)
                    sequences = _filter_gc(sequences, gc_min=gc_min, gc_max=gc_max)
                    reject_gc += before - len(sequences)

                unique_sequences: list[str] = []
                seen_batch: set[str] = set()
                for seq in sequences:
                    if not seq:
                        continue
                    if seq in seen or seq in seen_batch:
                        reject_dup += 1
                        continue
                    seen_batch.add(seq)
                    unique_sequences.append(seq)
                sequences = unique_sequences

                if fimo_cfg is not None and sequences:
                    before = len(sequences)
                    sequences, scores = _run_fimo_exclusion(
                        motifs=motifs,
                        sequences=sequences,
                        allow_zero_hit_only=allow_zero_hit_only,
                        max_score_norm=max_score_norm,
                    )
                    reject_fimo += before - len(sequences)
                    best_scores.update(scores)

                for seq in sequences:
                    if seq in seen:
                        continue
                    seen.add(seq)
                    accepted.append(seq)
                    if len(accepted) >= target:
                        break

                progress.update(
                    generated=generated,
                    accepted=len(accepted),
                    batch_index=batch_index,
                    batch_total=batch_total,
                    reject_fimo=reject_fimo,
                    reject_kmer=reject_kmer,
                    reject_gc=reject_gc,
                    reject_dup=reject_dup,
                )
        finally:
            progress.finish()

        if len(accepted) < target:
            raise ValueError(
                "background_pool sampling shortfall: "
                f"needed {target}, retained {len(accepted)} after {generated} candidates."
            )

        motif_id = hash_label_motif(label=self.input_name, source_kind="background_pool")
        rows: list[dict] = []
        for seq in accepted:
            row = {
                "tf": self.input_name,
                "tfbs": seq,
                "tfbs_core": seq,
                "source": "background_pool",
                "motif_id": motif_id,
                "tfbs_id": hash_tfbs_id(motif_id=motif_id, sequence=seq, scoring_backend="background_pool"),
            }
            if seq in best_scores:
                row["best_hit_score"] = best_scores.get(seq)
            rows.append(row)

        df = pd.DataFrame(rows)
        entries = [(self.input_name, seq, "background_pool") for seq in accepted]
        return entries, df, None
