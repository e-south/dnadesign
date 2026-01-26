"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/pwm_sampling.py

Shared Stage-A PWM sampling utilities.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ...core.artifacts.ids import hash_candidate_id
from ...utils import logging_utils

SMOOTHING_ALPHA = 1e-6
SCORE_HIST_BINS = 60
log = logging.getLogger(__name__)
_SAFE_LABEL_RE = None


def _format_rate(rate: float) -> str:
    if rate >= 1000.0:
        return f"{rate / 1000.0:.1f}k/s"
    return f"{rate:.1f}/s"


def _format_pwm_progress_line(
    *,
    motif_id: str,
    backend: str,
    generated: int,
    target: int,
    accepted: Optional[int],
    accepted_target: Optional[int],
    batch_index: Optional[int],
    batch_total: Optional[int],
    elapsed: float,
) -> str:
    safe_target = max(1, int(target))
    gen_pct = min(100, int(100 * generated / safe_target))
    parts = [f"PWM {motif_id}", backend, f"gen {gen_pct}% ({generated}/{safe_target})"]
    if batch_index is not None:
        total_label = "-" if batch_total is None else str(int(batch_total))
        parts.append(f"batch {int(batch_index)}/{total_label}")
    elapsed_label = f"{max(0.0, float(elapsed)):.1f}s"
    rate = generated / elapsed if elapsed > 0 else 0.0
    parts.append(elapsed_label)
    parts.append(_format_rate(rate))
    return " | ".join(parts)


@dataclass
class _PwmSamplingProgress:
    motif_id: str
    backend: str
    target: int
    accepted_target: Optional[int]
    stream: TextIO
    min_interval: float = 0.2

    def __post_init__(self) -> None:
        self._enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._start = time.monotonic()
        self._last_update = self._start
        self._last_len = 0
        self._last_state: tuple[int, Optional[int], Optional[int]] | None = None
        self._shown = False
        if self._enabled:
            logging_utils.set_progress_active(True)

    def update(
        self,
        *,
        generated: int,
        accepted: Optional[int],
        batch_index: Optional[int] = None,
        batch_total: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_update) < float(self.min_interval):
            return
        state = (int(generated), batch_index, batch_total)
        if self._shown and state == self._last_state and logging_utils.is_progress_line_visible():
            self._last_update = now
            return
        line = _format_pwm_progress_line(
            motif_id=self.motif_id,
            backend=self.backend,
            generated=int(generated),
            target=int(self.target),
            accepted=accepted,
            accepted_target=self.accepted_target,
            batch_index=batch_index,
            batch_total=batch_total,
            elapsed=now - self._start,
        )
        padded = line.ljust(self._last_len)
        self._last_len = max(self._last_len, len(line))
        self.stream.write(f"\r{padded}")
        self.stream.flush()
        self._last_update = now
        self._last_state = state
        self._shown = True
        logging_utils.mark_progress_line_visible()

    def finish(self) -> None:
        if self._enabled:
            logging_utils.set_progress_active(False)
        if not self._shown:
            return
        self.stream.write("\n")
        self.stream.flush()


def _safe_label(text: str) -> str:
    global _SAFE_LABEL_RE
    if _SAFE_LABEL_RE is None:
        import re

        _SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
    cleaned = _SAFE_LABEL_RE.sub("_", str(text).strip())
    return cleaned or "motif"


def _mining_attr(mining, name: str, default=None):
    if mining is None:
        return default
    if hasattr(mining, name):
        return getattr(mining, name)
    if isinstance(mining, dict):
        return mining.get(name, default)
    return default


@dataclass(frozen=True)
class FimoCandidate:
    seq: str
    score: float
    start: int
    stop: int
    strand: str
    matched_sequence: Optional[str] = None


def _write_candidate_records(
    records: list[dict],
    *,
    debug_output_dir: Path,
    debug_label: str,
    motif_id: str,
    motif_hash: str | None = None,
) -> Path:
    suffix = ""
    if motif_hash:
        suffix = f"__{_safe_label(motif_hash[:10])}"
    safe_label = f"{_safe_label(debug_label or motif_id)}{suffix}"
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    path = debug_output_dir / f"candidates__{safe_label}.parquet"
    df = pd.DataFrame(records)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if "candidate_id" not in existing.columns or "candidate_id" not in df.columns:
                raise ValueError(
                    f"Candidate append requires candidate_id in {path}. "
                    "Clear outputs/pools/candidates or use --fresh to reset."
                )
            if set(existing.columns) != set(df.columns):
                raise ValueError(
                    f"Candidate schema mismatch for {path}. Clear outputs/pools/candidates or use --fresh to reset."
                )
            df = df[existing.columns]
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["candidate_id"], keep="last")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Failed to append candidate records to {path}") from exc
    df.to_parquet(path, index=False)
    return path


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]  # per-position A/C/G/T probabilities
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None


@dataclass(frozen=True)
class PWMSamplingSummary:
    input_name: Optional[str]
    regulator: str
    backend: str
    generated: int
    target: int
    target_sites: Optional[int]
    eligible: int
    retained: int
    retained_len_min: Optional[int]
    retained_len_median: Optional[float]
    retained_len_mean: Optional[float]
    retained_len_max: Optional[int]
    retained_score_min: Optional[float]
    retained_score_median: Optional[float]
    retained_score_mean: Optional[float]
    retained_score_max: Optional[float]
    eligible_tier_counts: Optional[List[int]]
    retained_tier_counts: Optional[List[int]]
    tier0_score: Optional[float]
    tier1_score: Optional[float]
    eligible_score_hist_edges: Optional[List[float]] = None
    eligible_score_hist_counts: Optional[List[int]] = None


def normalize_background(bg: Optional[dict[str, float]]) -> dict[str, float]:
    if not bg:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    total = sum(bg.values())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    return {k: float(v) / total for k, v in bg.items()}


def build_log_odds(
    matrix: List[dict[str, float]],
    background: dict[str, float],
    *,
    smoothing_alpha: float = SMOOTHING_ALPHA,
) -> List[dict[str, float]]:
    # Alignment (3): smooth PWM probabilities to avoid -inf log-odds on zeros.
    bg = normalize_background(background)
    log_odds: List[dict[str, float]] = []
    for row in matrix:
        lod_row: dict[str, float] = {}
        for base in ("A", "C", "G", "T"):
            p = float(row.get(base, 0.0))
            b = float(bg.get(base, 0.0))
            if smoothing_alpha > 0.0:
                p = (1.0 - smoothing_alpha) * p + smoothing_alpha * b
            if p <= 0.0 or b <= 0.0:
                lod_row[base] = float("-inf")
            else:
                lod_row[base] = float(np.log(p / b))
        log_odds.append(lod_row)
    return log_odds


def score_sequence(
    seq: str,
    matrix: List[dict[str, float]],
    *,
    log_odds: Optional[List[dict[str, float]]] = None,
    background: Optional[dict[str, float]] = None,
) -> float:
    score = 0.0
    if log_odds is None:
        if background is None:
            background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        log_odds = build_log_odds(matrix, background)
    for base, probs in zip(seq, log_odds):
        lod = probs.get(base, float("-inf"))
        if lod == float("-inf"):
            return float("-inf")
        score += float(lod)
    return score


def _summarize_lengths(lengths: Sequence[int]) -> tuple[Optional[int], Optional[float], Optional[float], Optional[int]]:
    if not lengths:
        return None, None, None, None
    arr = np.asarray(lengths, dtype=float)
    return int(arr.min()), float(np.median(arr)), float(arr.mean()), int(arr.max())


def _summarize_scores(
    scores: Sequence[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not scores:
        return None, None, None, None
    arr = np.asarray(scores, dtype=float)
    return float(arr.min()), float(np.median(arr)), float(arr.mean()), float(arr.max())


def _rank_scored_sequences(scored: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    best_by_seq: dict[str, float] = {}
    for seq, score in scored:
        seq = str(seq)
        val = float(score)
        prev = best_by_seq.get(seq)
        if prev is None or val > prev:
            best_by_seq[seq] = val
    return sorted(best_by_seq.items(), key=lambda item: (-item[1], item[0]))


def _score_tier_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    n0 = max(1, int(math.ceil(0.01 * total)))
    n1 = min(total - n0, int(math.ceil(0.09 * total)))
    n2 = total - n0 - n1
    return n0, n1, n2


def _assign_score_tiers(ranked: Sequence[tuple[str, float]]) -> list[int]:
    total = len(ranked)
    n0, n1, _n2 = _score_tier_counts(total)
    tiers: list[int] = []
    for idx in range(total):
        if idx < n0:
            tiers.append(0)
        elif idx < n0 + n1:
            tiers.append(1)
        else:
            tiers.append(2)
    return tiers


def _build_score_hist(
    scores: Sequence[float],
    *,
    bins: int = SCORE_HIST_BINS,
) -> tuple[list[float], list[int]]:
    vals = [float(v) for v in scores if v is not None]
    if not vals:
        return [], []
    lo = min(vals)
    hi = max(vals)
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    counts, _ = np.histogram(np.asarray(vals, dtype=float), bins=edges)
    return [float(v) for v in edges], [int(v) for v in counts]


def _build_summary(
    *,
    generated: int,
    target: int,
    target_sites: Optional[int],
    eligible: Sequence[str],
    retained: Sequence[str],
    retained_scores: Optional[Sequence[float]] = None,
    eligible_tier_counts: Optional[Sequence[int]] = None,
    retained_tier_counts: Optional[Sequence[int]] = None,
    tier0_score: Optional[float] = None,
    tier1_score: Optional[float] = None,
    eligible_score_hist_edges: Optional[Sequence[float]] = None,
    eligible_score_hist_counts: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
    regulator: Optional[str] = None,
    backend: Optional[str] = None,
) -> PWMSamplingSummary:
    lengths = [len(seq) for seq in retained]
    min_len, median_len, mean_len, max_len = _summarize_lengths(lengths)
    score_min, score_median, score_mean, score_max = _summarize_scores(retained_scores or [])
    return PWMSamplingSummary(
        input_name=input_name,
        regulator=str(regulator or ""),
        backend=str(backend or ""),
        generated=int(generated),
        target=int(target),
        target_sites=int(target_sites) if target_sites is not None else None,
        eligible=int(len(eligible)),
        retained=int(len(retained)),
        retained_len_min=min_len,
        retained_len_median=median_len,
        retained_len_mean=mean_len,
        retained_len_max=max_len,
        retained_score_min=score_min,
        retained_score_median=score_median,
        retained_score_mean=score_mean,
        retained_score_max=score_max,
        eligible_tier_counts=list(eligible_tier_counts) if eligible_tier_counts is not None else None,
        retained_tier_counts=list(retained_tier_counts) if retained_tier_counts is not None else None,
        tier0_score=float(tier0_score) if tier0_score is not None else None,
        tier1_score=float(tier1_score) if tier1_score is not None else None,
        eligible_score_hist_edges=list(eligible_score_hist_edges) if eligible_score_hist_edges is not None else None,
        eligible_score_hist_counts=list(eligible_score_hist_counts) if eligible_score_hist_counts is not None else None,
    )


def sample_sequence_from_background(rng: np.random.Generator, probs: dict[str, float], length: int) -> str:
    bases = ["A", "C", "G", "T"]
    weights = np.array([probs[b] for b in bases], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(bases), size=length, replace=True, p=weights)
    return "".join(bases[i] for i in idx)


def sample_sequence_from_pwm(rng: np.random.Generator, matrix: List[dict[str, float]]) -> str:
    seq = []
    for probs in matrix:
        bases = ["A", "C", "G", "T"]
        weights = np.array([probs[b] for b in bases], dtype=float)
        weights = weights / weights.sum()
        seq.append(bases[int(rng.choice(len(bases), p=weights))])
    return "".join(seq)


def _matrix_array(matrix: List[dict[str, float]]) -> np.ndarray:
    rows = []
    for row in matrix:
        rows.append(
            [
                float(row.get("A", 0.0)),
                float(row.get("C", 0.0)),
                float(row.get("G", 0.0)),
                float(row.get("T", 0.0)),
            ]
        )
    return np.asarray(rows, dtype=float)


def _information_bits(matrix: np.ndarray) -> float:
    p = matrix + 1e-9
    return float((2 + (p * np.log2(p)).sum(axis=1)).sum())


def _select_pwm_window(
    *,
    matrix: List[dict[str, float]],
    log_odds: List[dict[str, float]],
    length: int,
    strategy: str,
) -> tuple[List[dict[str, float]], List[dict[str, float]], int, float]:
    if length < 1:
        raise ValueError("pwm.sampling.trim_window_length must be >= 1")
    if length > len(matrix):
        raise ValueError(f"pwm.sampling.trim_window_length={length} exceeds motif width {len(matrix)}")
    if strategy != "max_info":
        raise ValueError("pwm.sampling.trim_window_strategy must be 'max_info'")
    if length == len(matrix):
        return matrix, log_odds, 0, 0.0
    arr = _matrix_array(matrix)
    best_start = 0
    best_score = float("-inf")
    last_start = len(matrix) - length
    for start in range(last_start + 1):
        window = arr[start : start + length]
        score = _information_bits(window)
        if score > best_score:
            best_score = score
            best_start = start
    return (
        matrix[best_start : best_start + length],
        log_odds[best_start : best_start + length],
        best_start,
        best_score,
    )


def sample_pwm_sites(
    rng: np.random.Generator,
    motif: PWMMotif,
    *,
    input_name: Optional[str] = None,
    motif_hash: str | None = None,
    run_id: str | None = None,
    strategy: str,
    n_sites: int,
    oversample_factor: int,
    scoring_backend: str = "fimo",
    mining: Optional[object] = None,
    bgfile: Optional[str | Path] = None,
    keep_all_candidates_debug: bool = False,
    include_matched_sequence: bool = False,
    debug_output_dir: Optional[Path] = None,
    debug_label: Optional[str] = None,
    length_policy: str = "exact",
    length_range: Optional[Sequence[int]] = None,
    trim_window_length: Optional[int] = None,
    trim_window_strategy: str = "max_info",
    return_metadata: bool = False,
    return_summary: bool = False,
) -> Union[
    List[str],
    Tuple[List[str], dict[str, dict]],
    Tuple[List[str], PWMSamplingSummary],
    Tuple[List[str], dict[str, dict], Optional[PWMSamplingSummary]],
]:
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be > 0")
    scoring_backend = str(scoring_backend or "fimo").lower()
    if scoring_backend != "fimo":
        raise ValueError(f"Stage-A PWM sampling requires scoring_backend='fimo', got '{scoring_backend}'.")
    if keep_all_candidates_debug and run_id is None:
        raise ValueError("Stage-A PWM sampling keep_all_candidates_debug requires run_id to be set.")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("Stage-A PWM sampling strategy 'consensus' requires n_sites=1")

    width = len(motif.matrix)
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")
    if length_policy not in {"exact", "range"}:
        raise ValueError(f"Unsupported pwm length_policy: {length_policy}")
    log_odds = motif.log_odds or build_log_odds(motif.matrix, motif.background)
    window_label = "full"
    if trim_window_length is not None:
        matrix, log_odds, window_start, window_score = _select_pwm_window(
            matrix=motif.matrix,
            log_odds=log_odds,
            length=int(trim_window_length),
            strategy=str(trim_window_strategy),
        )
        width = len(matrix)
        window_label = f"{width}@{window_start}"
        log.debug(
            "Stage-A PWM sampling trimmed motif %s to window length %d (start=%d, score=%.3f).",
            motif.motif_id,
            width,
            window_start,
            window_score,
        )
    else:
        matrix = motif.matrix

    score_label = "best_hit_score"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    def _cap_label(
        cap_applied: bool,
        time_limited: bool,
    ) -> str:
        cap_label = ""
        if time_limited:
            mining_max_seconds = _mining_attr(mining, "max_seconds")
            if mining_max_seconds is not None:
                cap_label = (
                    f"{cap_label}; max_seconds={mining_max_seconds}"
                    if cap_label
                    else (f" (max_seconds={mining_max_seconds})")
                )
        return cap_label

    def _context(length_obs: str, cap_applied: bool, requested: int, generated: int, time_limited: bool) -> dict:
        return {
            "motif_id": motif.motif_id,
            "width": width,
            "strategy": strategy,
            "length_label": length_label,
            "window_label": window_label,
            "length_observed": length_obs,
            "score_label": score_label,
            "n_sites": n_sites,
            "oversample_factor": oversample_factor,
            "requested_candidates": requested,
            "generated_candidates": generated,
            "cap_applied": cap_applied,
            "cap_label": _cap_label(cap_applied, time_limited),
            "time_limited": time_limited,
            "mining_batch_size": _mining_attr(mining, "batch_size"),
            "mining_max_seconds": _mining_attr(mining, "max_seconds"),
            "mining_log_every_batches": _mining_attr(mining, "log_every_batches"),
        }

    def _resolve_length() -> int:
        if length_policy == "exact":
            return width
        if length_range is None or len(length_range) != 2:
            raise ValueError("pwm.sampling.length_range must be provided when length_policy=range")
        lo, hi = int(length_range[0]), int(length_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length_range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length_range must be min <= max")
        if lo < width:
            raise ValueError(f"pwm.sampling.length_range min must be >= motif width ({width}), got {lo}")
        return int(rng.integers(lo, hi + 1))

    def _embed_with_background(seq: str, target_len: int) -> str:
        if target_len == len(seq):
            return seq
        extra = target_len - len(seq)
        left_len = int(rng.integers(0, extra + 1))
        right_len = extra - left_len
        left = sample_sequence_from_background(rng, motif.background, left_len)
        right = sample_sequence_from_background(rng, motif.background, right_len)
        return f"{left}{seq}{right}"

    progress: _PwmSamplingProgress | None = None

    def _score_with_fimo(
        *,
        n_candidates: int,
        cap_applied: bool,
        requested: int,
        sequences: Optional[List[str]] = None,
    ) -> tuple[List[str], dict[str, dict]]:
        import tempfile

        from .pwm_fimo import (
            aggregate_best_hits,
            build_candidate_records,
            run_fimo,
            write_candidates_fasta,
            write_minimal_meme_motif,
        )

        mining_batch_size = int(_mining_attr(mining, "batch_size", n_candidates))
        mining_max_seconds = _mining_attr(mining, "max_seconds")
        mining_log_every = int(_mining_attr(mining, "log_every_batches", 1))
        log.info(
            "FIMO mining config for %s: target=%d batch=%d max_seconds=%s thresh=1.0",
            motif.motif_id,
            n_candidates,
            mining_batch_size,
            str(mining_max_seconds) if mining_max_seconds is not None else "-",
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
            label = _safe_label(debug_label or motif.motif_id)
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

        def _generate_batch(count: int) -> tuple[list[str], list[int], bool]:
            batch_start = time.monotonic()
            sequences: list[str] = []
            lengths: list[int] = []
            time_limited = False
            for _ in range(count):
                if mining_max_seconds is not None and sequences:
                    if (time.monotonic() - batch_start) >= float(mining_max_seconds):
                        time_limited = True
                        break
                target_len = _resolve_length()
                lengths.append(int(target_len))
                if strategy == "background":
                    core = sample_sequence_from_background(rng, motif.background, width)
                else:
                    core = sample_sequence_from_pwm(rng, matrix)
                full_seq = _embed_with_background(core, target_len)
                sequences.append(full_seq)
            return sequences, lengths, time_limited

        candidates_by_seq: dict[str, FimoCandidate] = {}
        lengths_all: list[int] = []
        generated_total = 0
        time_limited = False
        mining_time_limited = False
        batches = 0
        tsv_lines: list[str] = []
        provided_sequences = sequences
        candidate_records: list[dict] | None = [] if keep_all_candidates_debug else None

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
            # FIMO background uses the MEME motif background unless bgfile is provided.
            write_minimal_meme_motif(motif_for_fimo, meme_path)
            if provided_sequences is not None:
                lengths_all = [len(seq) for seq in provided_sequences]
                fasta_path = tmp_path / "candidates.fasta"
                records = build_candidate_records(motif.motif_id, provided_sequences, start_index=0)
                write_candidates_fasta(records, fasta_path)
                rows, raw_tsv = run_fimo(
                    meme_motif_path=meme_path,
                    fasta_path=fasta_path,
                    bgfile=Path(bgfile) if bgfile is not None else None,
                    thresh=1.0,
                    norc=True,
                    include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                    return_tsv=debug_path is not None,
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
                    if hit.score <= 0:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=False,
                            reject_reason="score_non_positive",
                        )
                        continue
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
                if progress is not None:
                    progress.update(generated=generated_total, accepted=len(candidates_by_seq), force=True)
            else:
                mining_start = time.monotonic()
                while generated_total < n_candidates:
                    if mining_max_seconds is not None and (time.monotonic() - mining_start) >= float(
                        mining_max_seconds
                    ):
                        mining_time_limited = True
                        break
                    remaining = int(n_candidates) - generated_total
                    if remaining <= 0:
                        break
                    batch_target = min(int(mining_batch_size), remaining)
                    sequences, lengths, batch_limited = _generate_batch(batch_target)
                    if batch_limited:
                        time_limited = True
                    if not sequences:
                        break
                    lengths_all.extend(lengths)
                    fasta_path = tmp_path / "candidates.fasta"
                    records = build_candidate_records(motif.motif_id, sequences, start_index=generated_total)
                    write_candidates_fasta(records, fasta_path)
                    rows, raw_tsv = run_fimo(
                        meme_motif_path=meme_path,
                        fasta_path=fasta_path,
                        bgfile=Path(bgfile) if bgfile is not None else None,
                        thresh=1.0,
                        norc=True,
                        include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                        return_tsv=debug_path is not None,
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
                        if hit.score <= 0:
                            _record_candidate(
                                seq=seq,
                                hit=hit,
                                accepted=False,
                                reject_reason="score_non_positive",
                            )
                            continue
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
                    generated_total += len(sequences)
                    batches += 1
                    if progress is not None:
                        progress.update(
                            generated=generated_total,
                            accepted=len(candidates_by_seq),
                            batch_index=batches,
                            batch_total=None,
                        )
                    if mining_log_every > 0 and batches % mining_log_every == 0:
                        log.info(
                            "FIMO mining %s batch %d/%s: generated=%d/%d eligible=%d",
                            motif.motif_id,
                            batches,
                            "-",
                            generated_total,
                            n_candidates,
                            len(candidates_by_seq),
                        )

        if debug_path is not None and tsv_lines:
            debug_path.write_text("\n".join(tsv_lines) + "\n")
            log.info("FIMO debug TSV written: %s", debug_path)

        eligible_hits = len(candidates_by_seq)
        length_obs = "-"
        if lengths_all:
            length_obs = (
                f"{min(lengths_all)}..{max(lengths_all)}"
                if min(lengths_all) != max(lengths_all)
                else str(lengths_all[0])
            )

        context = _context(length_obs, cap_applied, requested, generated_total, time_limited)
        context["mining_batch_size"] = mining_batch_size
        context["mining_max_seconds"] = mining_max_seconds
        context["mining_time_limited"] = mining_time_limited
        ranked = sorted(candidates_by_seq.values(), key=lambda cand: (-cand.score, cand.seq))
        tiers = _assign_score_tiers([(cand.seq, cand.score) for cand in ranked])
        rank_by_seq = {cand.seq: idx for idx, cand in enumerate(ranked)}
        tier_by_seq = {cand.seq: tiers[idx] for idx, cand in enumerate(ranked)}
        eligible_tier_counts = [0, 0, 0]
        for tier in tiers:
            eligible_tier_counts[tier] += 1
        picked = ranked[: int(n_sites)]
        retained_tier_counts = [0, 0, 0]
        for cand in picked:
            retained_tier_counts[tier_by_seq[cand.seq]] += 1
        if len(ranked) < n_sites:
            msg_lines = [
                (
                    "Stage-A PWM sampling shortfall for motif "
                    f"'{context.get('motif_id')}' "
                    f"(width={context.get('width')}, strategy={context.get('strategy')}, "
                    f"length={context.get('length_label')}, window={context.get('window_label')}, "
                    f"score={context.get('score_label')})."
                ),
                (
                    f"Requested n_sites={context.get('n_sites')} oversample_factor={context.get('oversample_factor')} "
                    f"-> candidates requested={context.get('requested_candidates')} "
                    f"generated={context.get('generated_candidates')}"
                    f"{context.get('cap_label')}."
                ),
                (f"Eligible unique sequences={len(ranked)} (need {n_sites})."),
            ]
            if context.get("length_observed"):
                msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
            suggestions = [
                "reduce n_sites",
                "increase oversample_factor",
            ]
            if context.get("mining_max_seconds") is not None and context.get("mining_time_limited"):
                suggestions.append("increase mining.max_seconds")
            if context.get("width") is not None and int(context.get("width")) <= 6:
                suggestions.append("try length_policy=range with a longer length_range")
            msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
            log.warning(" ".join(msg_lines))
        n0, n1, _n2 = _score_tier_counts(len(ranked))
        tier0_score = ranked[n0 - 1].score if n0 > 0 else None
        tier1_score = ranked[n0 + n1 - 1].score if n1 > 0 else None
        eligible_scores = [cand.score for cand in ranked]
        hist_edges, hist_counts = _build_score_hist(eligible_scores)
        if progress is not None:
            progress.update(
                generated=generated_total,
                accepted=len(candidates_by_seq),
                batch_index=batches if batches > 0 else None,
                batch_total=None,
                force=True,
            )
            progress.finish()
        log.info(
            "FIMO yield for motif %s: eligible=%d retained=%d",
            motif.motif_id,
            eligible_hits,
            len(picked),
            extra={"suppress_stdout": True},
        )
        meta_by_seq: dict[str, dict] = {}
        for cand in picked:
            meta = {
                "best_hit_score": cand.score,
                "rank_within_regulator": rank_by_seq[cand.seq],
                "tier": tier_by_seq[cand.seq],
                "fimo_start": cand.start,
                "fimo_stop": cand.stop,
                "fimo_strand": cand.strand,
            }
            if cand.matched_sequence:
                meta["fimo_matched_sequence"] = cand.matched_sequence
            meta_by_seq[cand.seq] = meta
        if candidate_records is not None and debug_dir is not None:
            selected_set = {c.seq for c in picked}
            for row in candidate_records:
                if row.get("sequence") in selected_set:
                    row["selected"] = True
                    row["reject_reason"] = None
                elif row.get("accepted"):
                    row["reject_reason"] = "not_selected"
            try:
                path = _write_candidate_records(
                    candidate_records,
                    debug_output_dir=debug_dir,
                    debug_label=debug_label or motif.motif_id,
                    motif_id=motif.motif_id,
                    motif_hash=motif_hash,
                )
                log.info("FIMO candidate records written: %s", path)
            except Exception:
                log.warning("Failed to write FIMO candidate records.", exc_info=True)
        summary = None
        if return_summary:
            summary = _build_summary(
                generated=generated_total,
                target=requested,
                target_sites=n_sites,
                eligible=list(candidates_by_seq.keys()),
                retained=[c.seq for c in picked],
                retained_scores=[cand.score for cand in picked],
                eligible_tier_counts=eligible_tier_counts,
                retained_tier_counts=retained_tier_counts,
                tier0_score=tier0_score,
                tier1_score=tier1_score,
                eligible_score_hist_edges=hist_edges,
                eligible_score_hist_counts=hist_counts,
                input_name=input_name,
                regulator=motif.motif_id,
                backend=scoring_backend,
            )
        return [c.seq for c in picked], meta_by_seq, summary

    if strategy == "consensus":
        progress = _PwmSamplingProgress(
            motif_id=motif.motif_id,
            backend=scoring_backend,
            target=1,
            accepted_target=n_sites,
            stream=sys.stdout,
        )
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)
        target_len = _resolve_length()
        full_seq = _embed_with_background(seq, target_len)
        selected, meta, summary = _score_with_fimo(
            n_candidates=1,
            cap_applied=False,
            requested=1,
            sequences=[full_seq],
        )
        if return_metadata and return_summary:
            return selected, meta, summary
        if return_metadata:
            return selected, meta
        if return_summary:
            return selected, summary
        return selected

    requested_candidates = max(1, n_sites * oversample_factor)
    n_candidates = max(1, int(requested_candidates))
    cap_applied = False
    progress = _PwmSamplingProgress(
        motif_id=motif.motif_id,
        backend=scoring_backend,
        target=requested_candidates,
        accepted_target=n_sites,
        stream=sys.stdout,
    )
    selected, meta, summary = _score_with_fimo(
        cap_applied=cap_applied,
        requested=requested_candidates,
        n_candidates=n_candidates,
    )
    if return_metadata and return_summary:
        return selected, meta, summary
    if return_metadata:
        return selected, meta
    if return_summary:
        return selected, summary
    return selected
