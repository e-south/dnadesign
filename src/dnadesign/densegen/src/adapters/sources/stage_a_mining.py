"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_mining.py

Stage-A PWM mining (candidate generation + FIMO scoring).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from ...core.artifacts.ids import hash_candidate_id
from ...core.stage_a_constants import FIMO_REPORT_THRESH
from .pwm_fimo import (
    FimoHit,
    aggregate_best_hits,
    build_candidate_records,
    run_fimo,
    write_candidates_fasta,
    write_minimal_meme_motif,
)
from .stage_a_paths import safe_label
from .stage_a_progress import _format_stage_a_milestone
from .stage_a_sampling_utils import (
    _matrix_cdf,
    _sample_background_batch,
    _sample_from_background_cdf,
    _sample_pwm_batch,
    build_log_odds,
    select_pwm_window_by_length,
)
from .stage_a_types import FimoCandidate, PWMMotif

log = logging.getLogger(__name__)


class _ProgressLike(Protocol):
    target: int

    def update(
        self,
        *,
        generated: int,
        accepted: Optional[int],
        batch_index: Optional[int] = None,
        batch_total: Optional[int] = None,
        force: bool = False,
    ) -> None: ...

    def set_phase(self, phase: str) -> None: ...


@dataclass(frozen=True)
class StageAMiningResult:
    candidates_by_seq: dict[str, FimoCandidate]
    candidates_with_hit: int
    eligible_raw: int
    generated_total: int
    requested_final: int
    batches: int
    generated_by_batch: list[int]
    unique_by_batch: list[int]
    lengths_all: list[int]
    cap_applied: bool
    time_limited: bool
    mining_time_limited: bool
    intended_core_by_seq: dict[str, tuple[int, int]]
    core_offset_by_seq: dict[str, int]
    candidate_records: list[dict] | None
    debug_dir: Path | None
    debug_tsv_lines: list[str] | None
    debug_tsv_path: Path | None


def mine_pwm_candidates(
    *,
    rng: np.random.Generator,
    motif: PWMMotif,
    matrix: list[dict[str, float]],
    background_cdf: np.ndarray,
    matrix_cdf: np.ndarray,
    width: int,
    strategy: str,
    length_policy: str,
    length_range: Sequence[int] | None,
    mining_batch_size: int,
    mining_log_every: int,
    budget_mode: str,
    budget_growth_factor: float,
    budget_max_candidates: int | None,
    budget_min_candidates: int | None,
    budget_max_seconds: float | None,
    budget_target_tier_fraction: float | None,
    n_candidates: int,
    requested: int,
    n_sites: int,
    bgfile: Path | str | None,
    keep_all_candidates_debug: bool,
    include_matched_sequence: bool,
    debug_output_dir: Path | None,
    debug_label: str | None,
    motif_hash: str | None,
    input_name: str,
    run_id: str,
    scoring_backend: str,
    uniqueness_key: str,
    progress: _ProgressLike | None,
    provided_sequences: Optional[list[str]] = None,
    intended_core_by_seq: Optional[dict[str, tuple[int, int]]] = None,
    core_offset_by_seq: Optional[dict[str, int]] = None,
) -> StageAMiningResult:
    log.info(
        "FIMO mining config for %s: mode=%s target=%d batch=%d max_seconds=%s max_candidates=%s thresh=%s",
        motif.motif_id,
        budget_mode,
        n_candidates,
        mining_batch_size,
        str(budget_max_seconds) if budget_max_seconds is not None else "-",
        str(budget_max_candidates) if budget_max_candidates is not None else "-",
        FIMO_REPORT_THRESH,
        extra={"suppress_stdout": True},
    )
    debug_path: Optional[Path] = None
    debug_dir = debug_output_dir
    if keep_all_candidates_debug:
        if debug_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="densegen-fimo-")
            debug_dir = Path(tmp_dir)
            log.warning(
                "Stage-A PWM sampling keep_all_candidates_debug enabled without outputs_root; "
                "writing FIMO debug TSVs to %s",
                debug_dir,
            )
        debug_dir.mkdir(parents=True, exist_ok=True)
        label = safe_label(debug_label or motif.motif_id)
        debug_path = debug_dir / f"{label}__fimo.tsv"

    def _merge_tsv(existing: list[str], text: str) -> None:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return
        if not existing:
            existing.extend(lines)
            return
        header_skipped = False
        for ln in lines:
            if ln.lstrip().startswith("#"):
                continue
            if not header_skipped:
                header_skipped = True
                continue
            existing.append(ln)

    def _resolve_length() -> int:
        if length_policy == "exact":
            return int(width)
        if length_range is None or len(length_range) != 2:
            raise ValueError("pwm.sampling.length.range must be provided when length.policy=range")
        lo, hi = int(length_range[0]), int(length_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length.range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length.range must be min <= max")
        return int(rng.integers(lo, hi + 1))

    def _embed_with_background(seq: str, target_len: int) -> tuple[str, int]:
        if target_len == len(seq):
            return seq, 0
        extra = target_len - len(seq)
        left_len = int(rng.integers(0, extra + 1))
        right_len = extra - left_len
        left = _sample_from_background_cdf(rng, background_cdf, left_len)
        right = _sample_from_background_cdf(rng, background_cdf, right_len)
        return f"{left}{seq}{right}", int(left_len)

    def _generate_batch(
        count: int,
        intended_core: dict[str, tuple[int, int]],
        offsets: dict[str, int],
    ) -> tuple[list[str], list[int], bool]:
        batch_start = time.monotonic()
        sequences: list[str] = []
        lengths: list[int] = []
        time_limited = False
        target_lengths = []
        for _ in range(count):
            if budget_max_seconds is not None and target_lengths:
                if (time.monotonic() - batch_start) >= float(budget_max_seconds):
                    time_limited = True
                    break
            target_len = _resolve_length()
            target_lengths.append(int(target_len))
        if not target_lengths:
            return sequences, lengths, time_limited
        counts_by_length: dict[int, int] = {}
        for target_len in target_lengths:
            counts_by_length[int(target_len)] = counts_by_length.get(int(target_len), 0) + 1
        for target_len, group_count in counts_by_length.items():
            core_len = min(int(width), int(target_len))
            if strategy == "background":
                cores = _sample_background_batch(rng, background_cdf, count=int(group_count), length=int(core_len))
            else:
                _, matrix_cdf_for_len, _ = _resolve_window(int(target_len))
                cores = _sample_pwm_batch(rng, matrix_cdf_for_len, count=int(group_count))
            for core in cores:
                full_seq, left_len = _embed_with_background(core, int(target_len))
                intended_start = int(left_len) + 1
                intended_stop = int(left_len) + int(core_len)
                intended_core[full_seq] = (intended_start, intended_stop)
                offsets[full_seq] = int(left_len)
                sequences.append(full_seq)
                lengths.append(int(target_len))
        return sequences, lengths, time_limited

    candidates_by_seq: dict[str, FimoCandidate] = {}
    candidates_with_hit = 0
    eligible_raw = 0
    lengths_all: list[int] = []
    generated_total = 0
    time_limited = False
    mining_time_limited = False
    cap_applied = False
    batches = 0
    unique_by_batch: list[int] = []
    generated_by_batch: list[int] = []
    tsv_lines: list[str] = []
    requested_final = int(requested)
    candidate_records: list[dict] | None = [] if keep_all_candidates_debug else None
    intended_core_by_seq = dict(intended_core_by_seq or {})
    core_offset_by_seq = dict(core_offset_by_seq or {})

    def _record_candidate(
        *,
        seq: str,
        hit,
        accepted: bool,
        reject_reason: str | None,
    ) -> None:
        if candidate_records is None:
            return
        resolved_motif_id = motif_hash or motif.motif_id
        candidate_id = hash_candidate_id(
            input_name=input_name,
            motif_id=resolved_motif_id,
            sequence=seq,
            scoring_backend=scoring_backend,
        )
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "input_name": input_name,
                "motif_id": resolved_motif_id,
                "motif_label": motif.motif_id,
                "scoring_backend": scoring_backend,
                "sequence": seq,
                "best_hit_score": None if hit is None else hit.score,
                "start": None if hit is None else hit.start,
                "stop": None if hit is None else hit.stop,
                "strand": None if hit is None else hit.strand,
                "matched_sequence": None if hit is None else hit.matched_sequence,
                "accepted": bool(accepted),
                "selected": False,
                "reject_reason": reject_reason,
            }
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meme_path = tmp_path / "motif.meme"
        motif_for_fimo = PWMMotif(motif_id=motif.motif_id, matrix=matrix, background=motif.background)
        write_minimal_meme_motif(motif_for_fimo, meme_path)
        fimo_bgfile = bgfile if bgfile is not None else "motif-file"
        log_odds_full = build_log_odds(matrix, motif.background, smoothing_alpha=0.0)
        window_cache: dict[int, tuple[list[dict[str, float]], np.ndarray, Path]] = {}

        def _resolve_window(target_len: int) -> tuple[list[dict[str, float]], np.ndarray, Path]:
            if target_len >= int(width):
                return matrix, matrix_cdf, meme_path
            cached = window_cache.get(int(target_len))
            if cached is not None:
                return cached
            window = select_pwm_window_by_length(
                matrix=matrix,
                log_odds=log_odds_full,
                length=int(target_len),
            )
            trimmed_matrix = window.matrix
            trimmed_cdf = _matrix_cdf(trimmed_matrix)
            trimmed_motif = PWMMotif(
                motif_id=motif.motif_id,
                matrix=trimmed_matrix,
                background=motif.background,
            )
            trimmed_path = tmp_path / f"motif_{int(target_len)}.meme"
            write_minimal_meme_motif(trimmed_motif, trimmed_path)
            window_cache[int(target_len)] = (trimmed_matrix, trimmed_cdf, trimmed_path)
            return window_cache[int(target_len)]

        if progress is not None:
            progress.set_phase("fimo")
        if provided_sequences is not None:
            lengths_all = [len(seq) for seq in provided_sequences]
            fasta_path = tmp_path / "candidates.fasta"
            records = build_candidate_records(motif.motif_id, provided_sequences, start_index=0)
            write_candidates_fasta(records, fasta_path)
            fimo_label = f"batch=1/1 candidates={len(records)}"
            fimo_start = time.monotonic()
            log.info(
                _format_stage_a_milestone(
                    motif_id=motif.motif_id,
                    phase="fimo batch start",
                    detail=fimo_label,
                )
            )
            rows, raw_tsv = run_fimo(
                meme_motif_path=meme_path,
                fasta_path=fasta_path,
                bgfile=fimo_bgfile,
                thresh=FIMO_REPORT_THRESH,
                norc=True,
                include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                return_tsv=debug_path is not None,
            )
            log.info(
                _format_stage_a_milestone(
                    motif_id=motif.motif_id,
                    phase="fimo batch end",
                    detail=f"{fimo_label} hits={len(rows)}",
                    elapsed=time.monotonic() - fimo_start,
                )
            )
            if debug_path is not None and raw_tsv is not None:
                _merge_tsv(tsv_lines, raw_tsv)
            best_hits = aggregate_best_hits(rows)
            for rec_id, seq in records:
                hit = best_hits.get(rec_id)
                if hit is None:
                    _record_candidate(
                        seq=seq,
                        hit=None,
                        accepted=False,
                        reject_reason="no_hit",
                    )
                    continue
                candidates_with_hit += 1
                if hit.score <= 0:
                    _record_candidate(
                        seq=seq,
                        hit=hit,
                        accepted=False,
                        reject_reason="score_non_positive",
                    )
                    continue
                eligible_raw += 1
                _record_candidate(
                    seq=seq,
                    hit=hit,
                    accepted=True,
                    reject_reason=None,
                )
                prev = candidates_by_seq.get(seq)
                if (
                    prev is None
                    or hit.score > prev.score
                    or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                ):
                    candidates_by_seq[seq] = FimoCandidate(
                        seq=seq,
                        score=hit.score,
                        start=hit.start,
                        stop=hit.stop,
                        strand=hit.strand,
                        matched_sequence=hit.matched_sequence,
                    )
            generated_total = len(provided_sequences)
            batches = 1
            unique_by_batch.append(len(candidates_by_seq))
            generated_by_batch.append(generated_total)
            if progress is not None:
                progress.update(generated=generated_total, accepted=len(candidates_by_seq), force=True)
            requested_final = int(requested)
        else:
            if n_candidates <= 0:
                return StageAMiningResult(
                    candidates_by_seq={},
                    candidates_with_hit=0,
                    eligible_raw=0,
                    generated_total=0,
                    requested_final=requested_final,
                    batches=0,
                    generated_by_batch=[],
                    unique_by_batch=[],
                    lengths_all=[],
                    cap_applied=False,
                    time_limited=False,
                    mining_time_limited=False,
                    intended_core_by_seq={},
                    core_offset_by_seq={},
                    candidate_records=candidate_records,
                    debug_dir=debug_dir,
                    debug_tsv_lines=tsv_lines if debug_path is not None else None,
                    debug_tsv_path=debug_path,
                )
            if requested_final > 0 and n_candidates == requested_final:
                lengths_all = [int(width) for _ in range(int(n_candidates))]
            mining_start = time.monotonic()
            target_candidates = int(n_candidates)
            core_by_seq: dict[str, str] = {}
            core_counts: dict[str, int] = {}

            def _eligible_unique_count() -> int:
                if uniqueness_key == "core":
                    return int(len(core_counts))
                return int(len(candidates_by_seq))

            def _record_core(seq: str, cand: FimoCandidate) -> None:
                from .stage_a_selection import _core_sequence

                core = _core_sequence(cand)
                prev_core = core_by_seq.get(seq)
                if prev_core is not None:
                    core_counts[prev_core] = core_counts.get(prev_core, 0) - 1
                    if core_counts.get(prev_core) == 0:
                        core_counts.pop(prev_core, None)
                core_by_seq[seq] = core
                core_counts[core] = core_counts.get(core, 0) + 1

            while True:
                if budget_max_seconds is not None and (time.monotonic() - mining_start) >= float(budget_max_seconds):
                    mining_time_limited = True
                    break
                if generated_total >= target_candidates:
                    if budget_mode == "fixed_candidates":
                        break
                    if budget_mode == "tier_target":
                        if budget_min_candidates is None or generated_total >= int(budget_min_candidates):
                            if budget_target_tier_fraction is not None:
                                required_unique = int(np.ceil(float(n_sites) / float(budget_target_tier_fraction)))
                                if _eligible_unique_count() >= required_unique:
                                    break
                    if budget_max_candidates is not None and generated_total >= int(budget_max_candidates):
                        cap_applied = True
                        break
                    next_target = int(np.ceil(target_candidates * float(budget_growth_factor)))
                    if budget_max_candidates is not None:
                        next_target = min(next_target, int(budget_max_candidates))
                    if next_target <= target_candidates:
                        cap_applied = True
                        break
                    target_candidates = next_target
                    continue
                remaining = int(target_candidates) - generated_total
                if remaining <= 0:
                    continue
                batch_target = min(int(mining_batch_size), remaining)
                sequences, lengths, batch_limited = _generate_batch(
                    batch_target, intended_core_by_seq, core_offset_by_seq
                )
                if batch_limited:
                    time_limited = True
                if not sequences:
                    break
                lengths_all.extend(lengths)
                records = build_candidate_records(motif.motif_id, sequences, start_index=generated_total)
                records_by_len: dict[int, list[tuple[str, str]]] = {}
                for (rec_id, seq), target_len in zip(records, lengths):
                    records_by_len.setdefault(int(target_len), []).append((rec_id, seq))
                best_hits: dict[str, FimoHit] = {}
                for target_len, group_records in records_by_len.items():
                    fasta_path = tmp_path / f"candidates_{int(target_len)}.fasta"
                    write_candidates_fasta(group_records, fasta_path)
                    _, _, motif_path = _resolve_window(int(target_len))
                    fimo_label = f"batch={batches + 1}/- len={int(target_len)} candidates={len(group_records)}"
                    fimo_start = time.monotonic()
                    log.info(
                        _format_stage_a_milestone(
                            motif_id=motif.motif_id,
                            phase="fimo batch start",
                            detail=fimo_label,
                        )
                    )
                    rows, raw_tsv = run_fimo(
                        meme_motif_path=motif_path,
                        fasta_path=fasta_path,
                        bgfile=fimo_bgfile,
                        thresh=FIMO_REPORT_THRESH,
                        norc=True,
                        include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                        return_tsv=debug_path is not None,
                    )
                    log.info(
                        _format_stage_a_milestone(
                            motif_id=motif.motif_id,
                            phase="fimo batch end",
                            detail=f"{fimo_label} hits={len(rows)}",
                            elapsed=time.monotonic() - fimo_start,
                        )
                    )
                    if debug_path is not None and raw_tsv is not None:
                        _merge_tsv(tsv_lines, raw_tsv)
                    group_best = aggregate_best_hits(rows)
                    best_hits.update(group_best)
                for rec_id, seq in records:
                    hit = best_hits.get(rec_id)
                    if hit is None:
                        _record_candidate(seq=seq, hit=None, accepted=False, reject_reason="no_hit")
                        continue
                    candidates_with_hit += 1
                    if hit.score <= 0:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=False,
                            reject_reason="score_non_positive",
                        )
                        continue
                    eligible_raw += 1
                    _record_candidate(seq=seq, hit=hit, accepted=True, reject_reason=None)
                    prev = candidates_by_seq.get(seq)
                    if (
                        prev is None
                        or hit.score > prev.score
                        or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                    ):
                        cand = FimoCandidate(
                            seq=seq,
                            score=hit.score,
                            start=hit.start,
                            stop=hit.stop,
                            strand=hit.strand,
                            matched_sequence=hit.matched_sequence,
                        )
                        candidates_by_seq[seq] = cand
                        if uniqueness_key == "core":
                            _record_core(seq, cand)
                generated_total += len(sequences)
                batches += 1
                unique_by_batch.append(_eligible_unique_count())
                generated_by_batch.append(generated_total)
                if progress is not None:
                    progress.update(
                        generated=generated_total,
                        accepted=_eligible_unique_count(),
                        batch_index=batches,
                        batch_total=None,
                    )
                if mining_log_every > 0 and batches % mining_log_every == 0:
                    log.info(
                        "FIMO mining %s batch %d/%s: generated=%d/%d eligible_unique=%d",
                        motif.motif_id,
                        batches,
                        "-",
                        generated_total,
                        target_candidates,
                        _eligible_unique_count(),
                    )
            requested_final = int(target_candidates)

    if progress is not None:
        progress.set_phase("postprocess")
    return StageAMiningResult(
        candidates_by_seq=candidates_by_seq,
        candidates_with_hit=candidates_with_hit,
        eligible_raw=eligible_raw,
        generated_total=generated_total,
        requested_final=requested_final,
        batches=batches,
        generated_by_batch=generated_by_batch,
        unique_by_batch=unique_by_batch,
        lengths_all=lengths_all,
        cap_applied=cap_applied,
        time_limited=time_limited,
        mining_time_limited=mining_time_limited,
        intended_core_by_seq=intended_core_by_seq,
        core_offset_by_seq=core_offset_by_seq,
        candidate_records=candidate_records,
        debug_dir=debug_dir,
        debug_tsv_lines=tsv_lines if debug_path is not None else None,
        debug_tsv_path=debug_path,
    )
