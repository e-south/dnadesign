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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ...core.artifacts.ids import hash_candidate_id
from ...core.pvalue_bins import resolve_pvalue_strata
from ...utils import logging_utils

SMOOTHING_ALPHA = 1e-6
PVAL_HIST_BINS = 60
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
    pvalue: float
    score: float
    bin_id: int
    bin_low: float
    bin_high: float
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
    strata_bins: Optional[str]
    pvalue_strata: Optional[List[float]]
    retain_depth: Optional[int]
    retain_bins: Optional[List[int]]
    eligible_bin_counts: Optional[List[int]]
    retained_bin_counts: Optional[List[int]]
    eligible_pvalue_hist_edges: Optional[List[float]] = None
    eligible_pvalue_hist_counts: Optional[List[int]] = None


@dataclass(frozen=True)
class SelectionOutcome:
    selected: List[str]
    retained: List[str]


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


def _build_pvalue_hist(
    pvalues: Sequence[float],
    *,
    floor: float,
    min_edge: Optional[float] = None,
    bins: int = PVAL_HIST_BINS,
) -> tuple[list[float], list[int]]:
    vals = [float(v) for v in pvalues if v is not None and float(v) > 0.0]
    if not vals:
        return [], []
    lo = min(vals)
    if min_edge is not None:
        lo = min(lo, float(min_edge))
    hi = max(vals)
    hi = max(hi, float(floor))
    hi = min(1.0, hi)
    if hi <= lo:
        hi = min(1.0, lo * 10.0) if lo > 0 else min(1.0, float(floor))
    if hi <= lo:
        hi = lo * 1.000001
    edges = np.logspace(np.log10(lo), np.log10(hi), num=int(bins) + 1)
    counts, _ = np.histogram(vals, bins=edges)
    return [float(v) for v in edges], [int(v) for v in counts]


def _build_summary(
    *,
    generated: int,
    target: int,
    target_sites: Optional[int],
    eligible: Sequence[str],
    retained: Sequence[str],
    strata_bins: Optional[str],
    pvalue_strata: Optional[Sequence[float]] = None,
    retain_depth: Optional[int] = None,
    retain_bins: Optional[Sequence[int]] = None,
    eligible_bin_counts: Optional[Sequence[int]] = None,
    retained_bin_counts: Optional[Sequence[int]] = None,
    eligible_pvalue_hist_edges: Optional[Sequence[float]] = None,
    eligible_pvalue_hist_counts: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
    regulator: Optional[str] = None,
    backend: Optional[str] = None,
) -> PWMSamplingSummary:
    lengths = [len(seq) for seq in retained]
    min_len, median_len, mean_len, max_len = _summarize_lengths(lengths)
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
        strata_bins=strata_bins,
        pvalue_strata=list(pvalue_strata) if pvalue_strata is not None else None,
        retain_depth=int(retain_depth) if retain_depth is not None else None,
        retain_bins=list(retain_bins) if retain_bins is not None else None,
        eligible_bin_counts=list(eligible_bin_counts) if eligible_bin_counts is not None else None,
        retained_bin_counts=list(retained_bin_counts) if retained_bin_counts is not None else None,
        eligible_pvalue_hist_edges=list(eligible_pvalue_hist_edges) if eligible_pvalue_hist_edges is not None else None,
        eligible_pvalue_hist_counts=list(eligible_pvalue_hist_counts)
        if eligible_pvalue_hist_counts is not None
        else None,
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


def select_by_score(
    candidates: List[Tuple[str, float]],
    *,
    n_sites: int,
    threshold: Optional[float],
    percentile: Optional[float],
    keep_low: bool,
    context: Optional[dict] = None,
) -> SelectionOutcome:
    scores = np.array([s for _, s in candidates], dtype=float)
    if threshold is None:
        cutoff = np.percentile(scores, float(percentile))  # type: ignore[arg-type]
    else:
        cutoff = float(threshold)
    if context is not None:
        if threshold is None and percentile is not None:
            context = dict(context)
            context["score_label"] = f"percentile={percentile} (cutoff={cutoff:.4g})"
        elif threshold is not None:
            context = dict(context)
            context["score_label"] = f"threshold={threshold}"
    if keep_low:
        picked = [(seq, score) for seq, score in candidates if score <= cutoff]
        ordered = sorted(picked, key=lambda item: item[1])
    else:
        picked = [(seq, score) for seq, score in candidates if score >= cutoff]
        ordered = sorted(picked, key=lambda item: item[1], reverse=True)
    seen: set[str] = set()
    unique: list[str] = []
    for seq, _score in ordered:
        if seq in seen:
            continue
        seen.add(seq)
        unique.append(seq)
    if len(unique) < n_sites:
        unique_total = len({seq for seq, _ in candidates})
        if context is None:
            log.warning(
                "Stage-A PWM sampling shortfall: retained=%d need=%d (candidates=%d).",
                len(unique),
                n_sites,
                unique_total,
            )
            return SelectionOutcome(selected=unique, retained=unique)
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
            (f"Unique candidates before filtering={unique_total}, after filtering={len(unique)} (need {n_sites})."),
        ]
        if context.get("length_observed"):
            msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
        suggestions = [
            "reduce n_sites",
            "lower score_threshold",
            "increase oversample_factor",
        ]
        if context.get("cap_applied"):
            if context.get("mining_max_candidates") is not None:
                suggestions.append("increase mining.max_candidates (cap was hit)")
            else:
                suggestions.append("increase max_candidates (cap was hit)")
        if context.get("time_limited"):
            suggestions.append("increase max_seconds (time limit was hit)")
        if context.get("width") is not None and int(context.get("width")) <= 6:
            suggestions.append("try length_policy=range with a longer length_range")
        msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
        log.warning(" ".join(msg_lines))
        return SelectionOutcome(selected=unique, retained=unique)
    return SelectionOutcome(selected=unique[:n_sites], retained=unique)


def _resolve_pvalue_strata(pvalue_strata: Sequence[float] | None) -> list[float]:
    edges = resolve_pvalue_strata(pvalue_strata)
    if not edges:
        raise ValueError("pvalue_strata must contain at least one edge.")
    cleaned: list[float] = []
    prev = 0.0
    for edge in edges:
        edge_val = float(edge)
        if not (0.0 < edge_val <= 1.0):
            raise ValueError("pvalue_strata values must be in (0, 1].")
        if edge_val <= prev:
            raise ValueError("pvalue_strata must be strictly increasing.")
        cleaned.append(edge_val)
        prev = edge_val
    return cleaned


def _assign_pvalue_bin(pvalue: float, edges: Sequence[float]) -> tuple[int, float, float]:
    low = 0.0
    for idx, edge in enumerate(edges):
        if pvalue <= edge:
            return idx, low, float(edge)
        low = float(edge)
    if not edges:
        return 0, 0.0, 1.0
    if len(edges) == 1:
        return 0, 0.0, float(edges[0])
    return len(edges) - 1, float(edges[-2]), float(edges[-1])


def _format_pvalue_bins(
    edges: Sequence[float],
    counts: Sequence[int],
    *,
    only_bins: Optional[Sequence[int]] = None,
) -> str:
    if not edges or not counts:
        return "-"
    only_set = {int(idx) for idx in only_bins} if only_bins is not None else None
    labels: list[str] = []
    low = 0.0
    for idx, (edge, count) in enumerate(zip(edges, counts)):
        if only_set is not None and idx not in only_set:
            low = float(edge)
            continue
        labels.append(f"({low:.0e},{float(edge):.0e}]:{int(count)}")
        low = float(edge)
    return " ".join(labels) if labels else "-"


def _format_pvalue_bin_pairs(
    edges: Sequence[float],
    eligible_counts: Sequence[int],
    retained_counts: Sequence[int],
) -> str:
    if not edges or not eligible_counts or not retained_counts:
        return "-"
    labels: list[str] = []
    for idx, (eligible, retained) in enumerate(zip(eligible_counts, retained_counts)):
        labels.append(f"b{idx} {int(eligible)}/{int(retained)}")
    return " | ".join(labels) if labels else "-"


def _select_fimo_candidates(
    candidates: List[FimoCandidate],
    *,
    n_sites: int,
    keep_weak: bool,
    context: dict,
) -> List[FimoCandidate]:
    unique: list[FimoCandidate] = []
    seen: set[str] = set()
    for cand in candidates:
        if cand.seq in seen:
            continue
        seen.add(cand.seq)
        unique.append(cand)
    if len(unique) < n_sites:
        msg_lines = [
            (
                "Stage-A PWM sampling shortfall for motif "
                f"'{context.get('motif_id')}' "
                f"(width={context.get('width')}, strategy={context.get('strategy')}, "
                f"length={context.get('length_label')}, window={context.get('window_label')}, "
                f"backend=fimo, stratified, pvalue={context.get('pvalue_label')})."
            ),
            (
                f"Requested n_sites={context.get('n_sites')} oversample_factor={context.get('oversample_factor')} "
                f"-> candidates requested={context.get('requested_candidates')} "
                f"generated={context.get('generated_candidates')}"
                f"{context.get('cap_label')}."
            ),
            (f"Unique candidates after filtering={len(unique)} (need {n_sites})."),
        ]
        if context.get("length_observed"):
            msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
        if context.get("pvalue_strata_label") is not None:
            msg_lines.append(f"P-value strata={context.get('pvalue_strata_label')}.")
        if context.get("retain_depth") is not None:
            msg_lines.append(f"Retain depth={context.get('retain_depth')}.")
        suggestions = [
            "reduce n_sites",
            "relax pvalue_strata floor (e.g., 1e-4 → 1e-3)",
            "increase oversample_factor",
        ]
        if context.get("retain_depth") is not None:
            suggestions.append("increase retain_depth")
        if context.get("cap_applied"):
            suggestions.append("increase max_candidates (cap was hit)")
        if context.get("time_limited"):
            suggestions.append("increase max_seconds (time limit was hit)")
        if context.get("mining_max_candidates") is not None and context.get("mining_candidates_limited"):
            suggestions.append("increase mining.max_candidates")
        if context.get("mining_max_batches") is not None and context.get("mining_batches_limited"):
            suggestions.append("increase mining.max_batches")
        if context.get("mining_max_seconds") is not None and context.get("mining_time_limited"):
            suggestions.append("increase mining.max_seconds")
        if context.get("width") is not None and int(context.get("width")) <= 6:
            suggestions.append("try length_policy=range with a longer length_range")
        msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
        log.warning(" ".join(msg_lines))
        return unique
    if keep_weak:
        ordered = sorted(unique, key=lambda c: (-c.pvalue, c.score))
    else:
        ordered = sorted(unique, key=lambda c: (c.pvalue, -c.score))
    return ordered[:n_sites]


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
    max_candidates: Optional[int] = None,
    max_seconds: Optional[float] = None,
    score_threshold: Optional[float],
    score_percentile: Optional[float],
    scoring_backend: str = "densegen",
    pvalue_strata: Optional[Sequence[float]] = None,
    retain_depth: Optional[int] = None,
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
    if max_seconds is not None and float(max_seconds) <= 0:
        raise ValueError("max_seconds must be > 0 when set")
    scoring_backend = str(scoring_backend or "densegen").lower()
    if scoring_backend not in {"densegen", "fimo"}:
        raise ValueError(f"Unsupported Stage-A PWM sampling scoring_backend: {scoring_backend}")
    if scoring_backend == "densegen":
        if score_threshold is None and score_percentile is None:
            raise ValueError(
                "Stage-A PWM sampling requires score_threshold or score_percentile when scoring_backend='densegen'"
            )
        if score_threshold is not None and score_percentile is not None:
            raise ValueError(
                "Stage-A PWM sampling requires exactly one of score_threshold or score_percentile when "
                "scoring_backend='densegen'"
            )
        if pvalue_strata is not None:
            raise ValueError("pvalue_strata is only valid when scoring_backend='fimo'")
        if retain_depth is not None:
            raise ValueError("retain_depth is only valid when scoring_backend='fimo'")
        if mining is not None:
            raise ValueError("mining is only valid when scoring_backend='fimo'")
        if include_matched_sequence:
            raise ValueError("include_matched_sequence is only valid when scoring_backend='fimo'")
    else:
        if pvalue_strata is None:
            raise ValueError("Stage-A PWM sampling requires pvalue_strata when scoring_backend='fimo'")
        if retain_depth is None:
            raise ValueError("Stage-A PWM sampling requires retain_depth when scoring_backend='fimo'")
        if max_candidates is not None or max_seconds is not None:
            raise ValueError(
                "max_candidates/max_seconds are only supported for densegen scoring; "
                "use mining.max_candidates or mining.max_seconds for fimo."
            )
        if score_threshold is not None or score_percentile is not None:
            log.warning(
                "Stage-A PWM sampling scoring_backend=fimo ignores score_threshold/score_percentile for motif %s.",
                motif.motif_id,
            )
    if keep_all_candidates_debug and run_id is None:
        raise ValueError("Stage-A PWM sampling keep_all_candidates_debug requires run_id to be set.")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("Stage-A PWM sampling strategy 'consensus' requires n_sites=1")

    keep_low = strategy == "background"
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

    score_label = f"threshold={score_threshold}" if score_threshold is not None else f"percentile={score_percentile}"
    pvalue_label = None
    if scoring_backend == "fimo" and pvalue_strata is not None:
        floor = float(_resolve_pvalue_strata(pvalue_strata)[-1])
        comparator = ">=" if keep_low else "<="
        pvalue_label = f"{comparator}{floor:g}"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    def _cap_label(
        cap_applied: bool,
        time_limited: bool,
        *,
        mining_max_candidates: Optional[int] = None,
    ) -> str:
        cap_label = ""
        if cap_applied:
            if mining_max_candidates is not None:
                cap_label = f" (capped by mining.max_candidates={mining_max_candidates})"
            elif max_candidates is not None:
                cap_label = f" (capped by max_candidates={max_candidates})"
        if time_limited and max_seconds is not None:
            cap_label = f"{cap_label}; max_seconds={max_seconds}" if cap_label else f" (max_seconds={max_seconds})"
        return cap_label

    def _context(length_obs: str, cap_applied: bool, requested: int, generated: int, time_limited: bool) -> dict:
        mining_cfg = mining
        mining_max_candidates = _mining_attr(mining_cfg, "max_candidates")
        return {
            "motif_id": motif.motif_id,
            "width": width,
            "strategy": strategy,
            "length_label": length_label,
            "window_label": window_label,
            "length_observed": length_obs,
            "score_label": score_label,
            "pvalue_label": pvalue_label,
            "n_sites": n_sites,
            "oversample_factor": oversample_factor,
            "requested_candidates": requested,
            "generated_candidates": generated,
            "cap_applied": cap_applied,
            "cap_label": _cap_label(cap_applied, time_limited, mining_max_candidates=mining_max_candidates),
            "time_limited": time_limited,
            "mining_batch_size": _mining_attr(mining_cfg, "batch_size"),
            "mining_max_batches": _mining_attr(mining_cfg, "max_batches"),
            "mining_max_seconds": _mining_attr(mining_cfg, "max_seconds"),
            "mining_log_every_batches": _mining_attr(mining_cfg, "log_every_batches"),
            "mining_max_candidates": mining_max_candidates,
        }

    def _select(
        candidates: List[Tuple[str, float]],
        *,
        length_obs: str,
        cap_applied: bool,
        requested: int,
        generated: int,
        time_limited: bool,
    ):
        return select_by_score(
            candidates,
            n_sites=n_sites,
            threshold=score_threshold,
            percentile=score_percentile,
            keep_low=keep_low,
            context=_context(length_obs, cap_applied, requested, generated, time_limited),
        )

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

        if pvalue_strata is None:
            raise ValueError("pvalue_strata required for fimo backend")
        if retain_depth is None:
            raise ValueError("retain_depth required for fimo backend")
        resolved_bins = _resolve_pvalue_strata(pvalue_strata)
        floor = float(resolved_bins[-1])
        depth = int(retain_depth)
        if depth <= 0:
            raise ValueError("retain_depth must be >= 1")
        if depth > len(resolved_bins):
            raise ValueError("retain_depth cannot exceed the number of pvalue_strata bins")
        keep_weak = keep_low
        if keep_weak:
            retain_bins = list(range(len(resolved_bins) - depth, len(resolved_bins)))
        else:
            retain_bins = list(range(depth))
        retained_set = {int(idx) for idx in retain_bins}
        mining_batch_size = int(_mining_attr(mining, "batch_size", n_candidates))
        mining_max_batches = _mining_attr(mining, "max_batches")
        mining_max_candidates = _mining_attr(mining, "max_candidates")
        mining_max_seconds = _mining_attr(mining, "max_seconds")
        mining_log_every = int(_mining_attr(mining, "log_every_batches", 1))
        log.info(
            "FIMO mining config for %s: target=%d batch=%d "
            "max_batches=%s max_candidates=%s max_seconds=%s floor=%s retain_depth=%d",
            motif.motif_id,
            n_candidates,
            mining_batch_size,
            str(mining_max_batches) if mining_max_batches is not None else "-",
            str(mining_max_candidates) if mining_max_candidates is not None else "-",
            str(mining_max_seconds) if mining_max_seconds is not None else "-",
            f"{floor:g}",
            depth,
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
                if max_seconds is not None and sequences:
                    if (time.monotonic() - batch_start) >= float(max_seconds):
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

        eligible_bin_counts = [0 for _ in resolved_bins]
        candidates: List[FimoCandidate] = []
        seen: set[str] = set()
        lengths_all: list[int] = []
        generated_total = 0
        time_limited = False
        mining_time_limited = False
        mining_batches_limited = False
        mining_candidates_limited = False
        batches = 0
        tsv_lines: list[str] = []
        provided_sequences = sequences
        candidate_records: list[dict] | None = [] if keep_all_candidates_debug else None

        def _record_candidate(
            *,
            seq: str,
            hit,
            bin_id: int | None,
            bin_low: float | None,
            bin_high: float | None,
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
                    "pvalue": None if hit is None else hit.pvalue,
                    "score": None if hit is None else hit.score,
                    "bin_id": bin_id,
                    "bin_low": bin_low,
                    "bin_high": bin_high,
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
            if provided_sequences is not None:
                lengths_all = [len(seq) for seq in provided_sequences]
                fasta_path = tmp_path / "candidates.fasta"
                records = build_candidate_records(motif.motif_id, provided_sequences, start_index=0)
                write_candidates_fasta(records, fasta_path)
                thresh = 1.0 if keep_all_candidates_debug or keep_weak else float(floor)
                rows, raw_tsv = run_fimo(
                    meme_motif_path=meme_path,
                    fasta_path=fasta_path,
                    bgfile=Path(bgfile) if bgfile is not None else None,
                    thresh=thresh,
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
                            bin_id=None,
                            bin_low=None,
                            bin_high=None,
                            accepted=False,
                            reject_reason="no_hit",
                        )
                        continue
                    bin_id, bin_low, bin_high = _assign_pvalue_bin(hit.pvalue, resolved_bins)
                    if keep_weak:
                        eligible = hit.pvalue >= float(floor)
                    else:
                        eligible = hit.pvalue <= float(floor)
                    if not eligible:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            bin_id=bin_id,
                            bin_low=bin_low,
                            bin_high=bin_high,
                            accepted=False,
                            reject_reason="pvalue_floor",
                        )
                        continue
                    if seq in seen:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            bin_id=bin_id,
                            bin_low=bin_low,
                            bin_high=bin_high,
                            accepted=False,
                            reject_reason="duplicate",
                        )
                        continue
                    seen.add(seq)
                    eligible_bin_counts[bin_id] += 1
                    candidates.append(
                        FimoCandidate(
                            seq=seq,
                            pvalue=hit.pvalue,
                            score=hit.score,
                            bin_id=bin_id,
                            bin_low=bin_low,
                            bin_high=bin_high,
                            start=hit.start,
                            stop=hit.stop,
                            strand=hit.strand,
                            matched_sequence=hit.matched_sequence,
                        )
                    )
                    _record_candidate(
                        seq=seq,
                        hit=hit,
                        bin_id=bin_id,
                        bin_low=bin_low,
                        bin_high=bin_high,
                        accepted=True,
                        reject_reason=None,
                    )
                generated_total = len(provided_sequences)
                batches = 1
                if progress is not None:
                    progress.update(generated=generated_total, accepted=len(candidates), force=True)
            else:
                mining_start = time.monotonic()
                while generated_total < n_candidates:
                    if mining_max_batches is not None and batches >= int(mining_max_batches):
                        mining_batches_limited = True
                        break
                    if mining_max_candidates is not None and generated_total >= int(mining_max_candidates):
                        mining_candidates_limited = True
                        break
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
                    thresh = 1.0 if keep_all_candidates_debug or keep_weak else float(floor)
                    rows, raw_tsv = run_fimo(
                        meme_motif_path=meme_path,
                        fasta_path=fasta_path,
                        bgfile=Path(bgfile) if bgfile is not None else None,
                        thresh=thresh,
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
                                bin_id=None,
                                bin_low=None,
                                bin_high=None,
                                accepted=False,
                                reject_reason="no_hit",
                            )
                            continue
                        bin_id, bin_low, bin_high = _assign_pvalue_bin(hit.pvalue, resolved_bins)
                        if keep_weak:
                            eligible = hit.pvalue >= float(floor)
                        else:
                            eligible = hit.pvalue <= float(floor)
                        if not eligible:
                            _record_candidate(
                                seq=seq,
                                hit=hit,
                                bin_id=bin_id,
                                bin_low=bin_low,
                                bin_high=bin_high,
                                accepted=False,
                                reject_reason="pvalue_floor",
                            )
                            continue
                        if seq in seen:
                            _record_candidate(
                                seq=seq,
                                hit=hit,
                                bin_id=bin_id,
                                bin_low=bin_low,
                                bin_high=bin_high,
                                accepted=False,
                                reject_reason="duplicate",
                            )
                            continue
                        seen.add(seq)
                        eligible_bin_counts[bin_id] += 1
                        candidates.append(
                            FimoCandidate(
                                seq=seq,
                                pvalue=hit.pvalue,
                                score=hit.score,
                                bin_id=bin_id,
                                bin_low=bin_low,
                                bin_high=bin_high,
                                start=hit.start,
                                stop=hit.stop,
                                strand=hit.strand,
                                matched_sequence=hit.matched_sequence,
                            )
                        )
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            bin_id=bin_id,
                            bin_low=bin_low,
                            bin_high=bin_high,
                            accepted=True,
                            reject_reason=None,
                        )
                    generated_total += len(sequences)
                    batches += 1
                    if progress is not None:
                        progress.update(
                            generated=generated_total,
                            accepted=len(candidates),
                            batch_index=batches,
                            batch_total=mining_max_batches,
                        )
                    if mining_log_every > 0 and batches % mining_log_every == 0:
                        bins_label = _format_pvalue_bins(resolved_bins, eligible_bin_counts)
                        log.info(
                            "FIMO mining %s batch %d/%s: generated=%d/%d eligible=%d strata=%s",
                            motif.motif_id,
                            batches,
                            str(mining_max_batches) if mining_max_batches is not None else "-",
                            generated_total,
                            n_candidates,
                            len(candidates),
                            bins_label,
                        )

        if debug_path is not None and tsv_lines:
            debug_path.write_text("\n".join(tsv_lines) + "\n")
            log.info("FIMO debug TSV written: %s", debug_path)

        eligible_hits = sum(eligible_bin_counts)
        length_obs = "-"
        if lengths_all:
            length_obs = (
                f"{min(lengths_all)}..{max(lengths_all)}"
                if min(lengths_all) != max(lengths_all)
                else str(lengths_all[0])
            )

        context = _context(length_obs, cap_applied, requested, generated_total, time_limited)
        context["pvalue_strata_label"] = ",".join(f"{edge:g}" for edge in resolved_bins)
        context["retain_depth"] = depth
        context["mining_batch_size"] = mining_batch_size
        context["mining_max_batches"] = mining_max_batches
        context["mining_max_candidates"] = mining_max_candidates
        context["mining_max_seconds"] = mining_max_seconds
        context["mining_time_limited"] = mining_time_limited
        context["mining_batches_limited"] = mining_batches_limited
        context["mining_candidates_limited"] = mining_candidates_limited
        retained_candidates = [cand for cand in candidates if int(cand.bin_id) in retained_set]
        picked = _select_fimo_candidates(
            retained_candidates,
            n_sites=n_sites,
            keep_weak=keep_weak,
            context=context,
        )
        retained_bin_counts = [0 for _ in resolved_bins]
        for cand in picked:
            idx = max(0, min(int(cand.bin_id), len(resolved_bins) - 1))
            retained_bin_counts[idx] += 1
        strata_label = _format_pvalue_bin_pairs(resolved_bins, eligible_bin_counts, retained_bin_counts)
        if progress is not None:
            progress.update(
                generated=generated_total,
                accepted=len(candidates),
                batch_index=batches if batches > 0 else None,
                batch_total=mining_max_batches,
                force=True,
            )
            progress.finish()
        log.info(
            "FIMO yield for motif %s: eligible=%d retained=%d strata=%s",
            motif.motif_id,
            eligible_hits,
            len(picked),
            strata_label,
            extra={"suppress_stdout": True},
        )
        meta_by_seq: dict[str, dict] = {}
        for cand in picked:
            meta = {
                "fimo_score": cand.score,
                "fimo_pvalue": cand.pvalue,
                "fimo_bin_id": cand.bin_id,
                "fimo_bin_low": cand.bin_low,
                "fimo_bin_high": cand.bin_high,
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
                    bin_id = row.get("bin_id")
                    if bin_id is not None and int(bin_id) not in retained_set:
                        row["reject_reason"] = "outside_retained_bins"
                    else:
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
            retain_bin_ids = list(retain_bins)
            eligible_pvalues = [cand.pvalue for cand in candidates if cand.pvalue is not None]
            hist_edges, hist_counts = _build_pvalue_hist(
                eligible_pvalues,
                floor=floor,
                min_edge=resolved_bins[0] if resolved_bins else None,
            )
            summary = _build_summary(
                generated=generated_total,
                target=requested,
                target_sites=n_sites,
                eligible=[c.seq for c in candidates],
                retained=[c.seq for c in picked],
                strata_bins=strata_label,
                pvalue_strata=resolved_bins,
                retain_depth=depth,
                retain_bins=retain_bin_ids,
                eligible_bin_counts=eligible_bin_counts,
                retained_bin_counts=retained_bin_counts,
                eligible_pvalue_hist_edges=hist_edges,
                eligible_pvalue_hist_counts=hist_counts,
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
            accepted_target=n_sites if scoring_backend == "fimo" else None,
            stream=sys.stdout,
        )
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)
        target_len = _resolve_length()
        full_seq = _embed_with_background(seq, target_len)
        if scoring_backend == "densegen":
            score = score_sequence(seq, matrix, log_odds=log_odds, background=motif.background)
            selection = _select(
                [(full_seq, score)],
                length_obs=str(target_len),
                cap_applied=False,
                requested=1,
                generated=1,
                time_limited=False,
            )
            progress.update(generated=1, accepted=None, force=True)
            progress.finish()
            if return_metadata and return_summary:
                summary = _build_summary(
                    generated=1,
                    target=1,
                    target_sites=n_sites,
                    eligible=selection.retained,
                    retained=selection.selected,
                    strata_bins=None,
                    pvalue_strata=None,
                    retain_depth=None,
                    retain_bins=None,
                    eligible_bin_counts=None,
                    retained_bin_counts=None,
                    input_name=input_name,
                    regulator=motif.motif_id,
                    backend=scoring_backend,
                )
                return selection.selected, {}, summary
            if return_metadata:
                return selection.selected, {}
            if return_summary:
                summary = _build_summary(
                    generated=1,
                    target=1,
                    target_sites=n_sites,
                    eligible=selection.retained,
                    retained=selection.selected,
                    strata_bins=None,
                    pvalue_strata=None,
                    retain_depth=None,
                    retain_bins=None,
                    eligible_bin_counts=None,
                    retained_bin_counts=None,
                    input_name=input_name,
                    regulator=motif.motif_id,
                    backend=scoring_backend,
                )
                return selection.selected, summary
            return selection.selected
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
    n_candidates = requested_candidates
    cap_applied = False
    mining_max_candidates = _mining_attr(mining, "max_candidates")
    if scoring_backend == "densegen":
        if max_candidates is not None:
            cap_val = int(max_candidates)
            if cap_val <= 0:
                raise ValueError("max_candidates must be > 0 when set")
            if requested_candidates > cap_val:
                n_candidates = cap_val
                cap_applied = True
                log.warning(
                    "Stage-A PWM sampling capped candidate generation for motif %s: requested=%d max_candidates=%d",
                    motif.motif_id,
                    requested_candidates,
                    cap_val,
                )
        else:
            if mining_max_candidates is not None:
                mining_cap = int(mining_max_candidates)
                if mining_cap < n_sites:
                    log.warning(
                        "Stage-A PWM sampling mining.max_candidates=%d is below n_sites=%d for motif %s; "
                        "shortfall possible.",
                        mining_cap,
                        n_sites,
                        motif.motif_id,
                    )
                if mining_cap != requested_candidates:
                    cap_applied = mining_cap < requested_candidates
                    n_candidates = mining_cap
                    log.info(
                        "PWM mining candidate target for motif %s: requested=%d mining.max_candidates=%d",
                        motif.motif_id,
                        requested_candidates,
                        mining_cap,
                    )
    n_candidates = max(1, n_candidates)
    progress = _PwmSamplingProgress(
        motif_id=motif.motif_id,
        backend=scoring_backend,
        target=requested_candidates,
        accepted_target=n_sites if scoring_backend == "fimo" else None,
        stream=sys.stdout,
    )
    if scoring_backend == "densegen":
        candidates: List[Tuple[str, str]] = []
        lengths: List[int] = []
        start = time.monotonic()
        time_limited = False
        for _ in range(n_candidates):
            if max_seconds is not None and candidates:
                if (time.monotonic() - start) >= float(max_seconds):
                    time_limited = True
                    break
            target_len = _resolve_length()
            lengths.append(int(target_len))
            if strategy == "background":
                core = sample_sequence_from_background(rng, motif.background, width)
            else:
                core = sample_sequence_from_pwm(rng, matrix)
            full_seq = _embed_with_background(core, target_len)
            candidates.append((full_seq, core))
            progress.update(generated=len(candidates), accepted=None)
        progress.update(generated=len(candidates), accepted=None, force=True)
        progress.finish()
        if time_limited:
            log.warning(
                "Stage-A PWM sampling hit max_seconds for motif %s: generated=%d requested=%d",
                motif.motif_id,
                len(candidates),
                requested_candidates,
            )
        length_obs = "-"
        if lengths:
            length_obs = f"{min(lengths)}..{max(lengths)}" if min(lengths) != max(lengths) else str(lengths[0])
        scored = [
            (full_seq, score_sequence(core, matrix, log_odds=log_odds, background=motif.background))
            for full_seq, core in candidates
        ]
        selection = _select(
            scored,
            length_obs=length_obs,
            cap_applied=cap_applied,
            requested=requested_candidates,
            generated=len(candidates),
            time_limited=time_limited,
        )
        summary = None
        if return_summary:
            summary = _build_summary(
                generated=len(candidates),
                target=requested_candidates,
                target_sites=n_sites,
                eligible=selection.retained,
                retained=selection.selected,
                strata_bins=None,
                pvalue_strata=None,
                retain_depth=None,
                retain_bins=None,
                eligible_bin_counts=None,
                retained_bin_counts=None,
                input_name=input_name,
                regulator=motif.motif_id,
                backend=scoring_backend,
            )
        if return_metadata and return_summary:
            return selection.selected, {}, summary
        if return_metadata:
            return selection.selected, {}
        if return_summary:
            return selection.selected, summary
        return selection.selected
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
