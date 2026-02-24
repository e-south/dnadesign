"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/scoring.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np

from dnadesign.cruncher.core.pvalue import logodds_to_p_lookup
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.sequence import revcomp_int
from dnadesign.cruncher.core.state import SequenceState

logger = logging.getLogger(__name__)
_PWM_STATS_CACHE_MAXSIZE = 512


def _best_hit_from_llrs(
    llrs_fwd: np.ndarray,
    llrs_rev: np.ndarray | None,
    *,
    seq_len: int,
    width: int,
    prefer_strand: str = "+",
    eps: float = 1.0e-12,
) -> Tuple[float, int, str, str]:
    if llrs_fwd.size == 0:
        return float("-inf"), 0, prefer_strand, "empty"

    fwd_max = float(np.max(llrs_fwd))
    fwd_offsets = np.flatnonzero(np.abs(llrs_fwd - fwd_max) <= eps)
    fwd_offset = int(np.min(fwd_offsets)) if fwd_offsets.size else int(np.argmax(llrs_fwd))
    fwd_start = int(fwd_offset)
    best_score = fwd_max
    best_offset = fwd_offset
    best_strand = "+"
    best_start = fwd_start

    if llrs_rev is None:
        return best_score, best_offset, best_strand, "max_forward"

    rev_max = float(np.max(llrs_rev))
    rev_offsets = np.flatnonzero(np.abs(llrs_rev - rev_max) <= eps)
    if rev_offsets.size:
        rev_starts = seq_len - width - rev_offsets
        best_rev_idx = int(np.argmin(rev_starts))
        rev_offset = int(rev_offsets[best_rev_idx])
        rev_start = int(rev_starts[best_rev_idx])
    else:
        rev_offset = int(np.argmax(llrs_rev))
        rev_start = int(seq_len - width - rev_offset)

    if rev_max > best_score + eps:
        return rev_max, rev_offset, "-", "max_reverse"
    if abs(rev_max - best_score) <= eps:
        if rev_start < best_start:
            return rev_max, rev_offset, "-", "tie_leftmost"
        if rev_start == best_start and prefer_strand == "-":
            return rev_max, rev_offset, "-", "tie_prefer_strand"
    if abs(rev_max - best_score) <= eps and rev_start == best_start and prefer_strand == "+":
        return best_score, best_offset, best_strand, "tie_prefer_strand"
    if abs(rev_max - best_score) <= eps and rev_start > best_start:
        return best_score, best_offset, best_strand, "tie_leftmost"
    return best_score, best_offset, best_strand, "max_forward"


@dataclass(slots=True)
class _PWMInfo:
    """
    Holds all precomputed data for one PWM:
      • lom               : log-odds matrix (w x 4)
      • null_scores, tail_p: DP table → P(X ≥ LLR) for each possible LLR
      • width             : motif length
      • consensus_llr     : sum of column-max LLRs (for normalization)
      • consensus_neglogp_by_len : cache of -log₁₀(p_seq) for consensus per seq_length
      • null_mean         : mean of the null distribution of single-window LLRs
      • null_std          : standard deviation of that null distribution
    """

    lom: np.ndarray
    null_scores: np.ndarray
    tail_p: np.ndarray
    width: int

    consensus_llr: float = 0.0
    consensus_neglogp_by_len: Dict[int, float] = field(default_factory=dict)

    null_mean: float = 0.0
    null_std: float = 1.0


def _bg_key(background: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(background, dtype=float)
    if arr.shape != (4,):
        raise ValueError("background must be a length-4 probability vector")
    return tuple(float(x) for x in arr)


@lru_cache(maxsize=_PWM_STATS_CACHE_MAXSIZE)
def _pwm_stats_cached(
    lom_bytes: bytes,
    shape: tuple[int, int],
    bg_tuple: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    lom = np.frombuffer(lom_bytes, dtype=np.float64).reshape(shape)
    null_scores, tail_p = logodds_to_p_lookup(lom, np.asarray(bg_tuple, dtype=float))
    consensus_llr = float(np.max(lom, axis=1).sum())

    pmf = np.empty_like(tail_p)
    if tail_p.size > 1:
        pmf[:-1] = tail_p[:-1] - tail_p[1:]
    pmf[-1] = tail_p[-1]
    pmf = np.clip(pmf, 0.0, 1.0)

    mean_null = float((null_scores * pmf).sum())
    var_null = float(((null_scores - mean_null) ** 2 * pmf).sum())
    std_null = float(np.sqrt(var_null)) if var_null > 0 else 1.0
    return lom, null_scores, tail_p, consensus_llr, mean_null, std_null


def pwm_stats_cache_info():  # pragma: no cover - thin wrapper
    return _pwm_stats_cached.cache_info()


def clear_pwm_stats_cache() -> None:  # pragma: no cover - thin wrapper
    _pwm_stats_cached.cache_clear()


class Scorer:
    """
    Multi‐PWM scorer with supported scales:
      • "llr"                 → raw max LLR per PWM
      • "z"                   → z-score of raw LLR vs the PWM-specific null distribution
      • "logp"                → -log10(p_seq) per PWM
      • "normalized-llr"      → (raw_llr − null_mean) / (consensus_llr − null_mean)
      • "consensus-neglop-sum"→ normalized (-log10(p_seq) / -log10(p_consensus)) per PWM

    Usage:
        scorer = Scorer(pwms, background=(0.25,0.25,0.25,0.25), bidirectional=True, scale="z")
        per_tf = scorer.compute_all_per_pwm(seq_array, seq_length)
    """

    SUPPORTED_SCALES = {"llr", "z", "logp", "normalized-llr", "consensus-neglop-sum"}

    def __init__(
        self,
        pwms: Dict[str, PWM],
        *,
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        bidirectional: bool = True,
        scale: str = "logp",
        pseudocounts: float = 0.0,
        log_odds_clip: float | None = None,
    ) -> None:
        self.scale = scale.lower()
        if self.scale not in self.SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale '{scale}'; choose from {self.SUPPORTED_SCALES}.")

        self.bg = np.asarray(background, dtype=float)
        if self.bg.shape != (4,) or not np.isclose(self.bg.sum(), 1.0):
            raise ValueError("background must be a length-4 probability vector summing to 1.0.")

        self.bidirectional = bool(bidirectional)
        self.pseudocounts = float(pseudocounts)
        self.log_odds_clip = log_odds_clip if log_odds_clip is None else float(log_odds_clip)

        logger.debug(
            "Building Scorer (scale=%r, bidirectional=%s)",
            self.scale,
            self.bidirectional,
        )

        # Build per-PWM info (lom, null lookup, and cached null moments).
        self._cache: Dict[str, _PWMInfo] = {}
        bg_tuple = _bg_key(self.bg)
        for name, pwm in pwms.items():
            logger.debug("  Precomputing PWM info for %s", name)
            lom = np.asarray(
                pwm.log_odds(
                    self.bg,
                    pseudocounts=self.pseudocounts,
                    log_odds_clip=self.log_odds_clip,
                ),
                dtype=np.float64,
            )
            lom_contig = np.ascontiguousarray(lom)
            cached_lom, null_scores, tail_p, consensus_llr, mean_null, std_null = _pwm_stats_cached(
                lom_contig.tobytes(),
                lom_contig.shape,
                bg_tuple,
            )
            info = _PWMInfo(
                lom=cached_lom,
                null_scores=null_scores,
                tail_p=tail_p,
                width=cached_lom.shape[0],
            )
            info.consensus_llr = consensus_llr
            info.null_mean = mean_null
            info.null_std = std_null
            self._cache[name] = info

        logger.debug("Finished building cache with %d PWMs", len(self._cache))

    @staticmethod
    def _build_one_pwm_info(
        pwm: PWM,
        background: np.ndarray,
        *,
        pseudocounts: float = 0.0,
        log_odds_clip: float | None = None,
    ) -> _PWMInfo:
        """
        Given a PWM, compute its log‐odds matrix and null distribution lookup table.
        """
        lom = pwm.log_odds(background, pseudocounts=pseudocounts, log_odds_clip=log_odds_clip)
        null_scores, tail_p = logodds_to_p_lookup(lom, background)
        return _PWMInfo(lom=lom, null_scores=null_scores, tail_p=tail_p, width=lom.shape[0])

    def _interp_tail_p(self, raw_llr: float, info: _PWMInfo) -> float:
        """
        Return P(X ≥ raw_llr) from the DP table.
        """
        idx = np.searchsorted(info.null_scores, raw_llr, side="right") - 1
        idx = np.clip(idx, 0, info.tail_p.size - 1)
        return float(info.tail_p[idx])

    @staticmethod
    def _p_seq_from_p_win(p_win: float, n_offsets: int, *, bidirectional: bool) -> float:
        n_offsets = max(1, int(n_offsets))
        n_tests = n_offsets * (2 if bidirectional else 1)
        log_term = np.log1p(-float(p_win))
        return float(-np.expm1(float(n_tests) * log_term))

    def _per_pwm_neglogp(self, raw_llr: float, info: _PWMInfo, seq_length: int) -> float:
        """
        Compute −log10(p_seq) for a single PWM:
          p_win = P(X ≥ raw_llr) on one window,
          n_win = max(1, seq_length − width + 1),
          p_seq = 1 − (1 − p_win)^n_tests,
          where n_tests = n_win (forward-only) or 2 * n_win (bidirectional).
          return −log10(p_seq).
        """
        w = info.width
        p_win = self._interp_tail_p(raw_llr, info)
        n_win = max(1, seq_length - w + 1)
        p_seq = self._p_seq_from_p_win(p_win, n_win, bidirectional=self.bidirectional)
        p_seq = max(p_seq, 1e-300)
        neglogp = -np.log10(p_seq)
        logger.debug(
            "    PWMCALC: raw_llr=%.3f, width=%d, n_win=%d, neglogp=%.3f",
            raw_llr,
            w,
            n_win,
            neglogp,
        )
        return neglogp

    def _best_llr_and_location(
        self,
        seq: np.ndarray,
        info: _PWMInfo,
        *,
        rev: np.ndarray | None = None,
    ) -> Tuple[float, int, str, str]:
        """
        Scan a numeric‐encoded sequence to find the best raw LLR (and its offset & strand).
        """
        L, w = seq.size, info.width
        if L < w:
            return float("-inf"), 0, "+", "sequence_too_short"

        def _scan(arr: np.ndarray) -> np.ndarray:
            windows = np.lib.stride_tricks.sliding_window_view(arr, w)
            # windows shape: (L - w + 1, w)
            return info.lom[np.arange(w)[:, None], windows.T].sum(axis=0)

        llrs_fwd = _scan(seq)
        llrs_rev = None
        if self.bidirectional:
            if rev is None:
                rev = (3 - seq)[::-1]
            llrs_rev = _scan(rev)

        best_llr, best_offset, best_strand, tiebreak = _best_hit_from_llrs(
            llrs_fwd,
            llrs_rev,
            seq_len=L,
            width=w,
            prefer_strand="+",
        )

        logger.debug(
            "    BEST_LLR: best_llr=%.3f at offset=%d, strand=%s",
            best_llr,
            best_offset,
            best_strand,
        )
        return best_llr, best_offset, best_strand, tiebreak

    def _info(self, tf: str) -> _PWMInfo:
        info = self._cache.get(tf)
        if info is None:
            available = ", ".join(self._cache.keys())
            raise ValueError(f"Unknown TF '{tf}'. Available: {available}")
        return info

    @property
    def tf_names(self) -> list[str]:
        return list(self._cache.keys())

    @property
    def pwm_count(self) -> int:
        return len(self._cache)

    def consensus_llr(self, tf: str) -> float:
        return float(self._info(tf).consensus_llr)

    def pwm_width(self, tf: str) -> int:
        return int(self._info(tf).width)

    def consensus_sequence(self, tf: str) -> str:
        info = self._info(tf)
        idx_vec = np.argmax(info.lom, axis=1)
        return "".join("ACGT"[i] for i in idx_vec)

    def best_llr(self, seq: np.ndarray, tf: str) -> Tuple[float, int, str]:
        info = self._info(tf)
        raw_llr, offset, strand, _ = self._best_llr_and_location(seq, info)
        return raw_llr, offset, strand

    def best_hit(self, seq: np.ndarray, tf: str) -> Dict[str, object]:
        info = self._info(tf)
        rev = (3 - seq)[::-1] if self.bidirectional else None
        raw_llr, offset, strand, tiebreak = self._best_llr_and_location(seq, info, rev=rev)
        return self._hit_from_scan(
            seq,
            info,
            tf=tf,
            raw_llr=raw_llr,
            offset=offset,
            strand=strand,
            tiebreak=tiebreak,
        )

    def _hit_from_scan(
        self,
        seq: np.ndarray,
        info: _PWMInfo,
        *,
        tf: str,
        raw_llr: float,
        offset: int,
        strand: str,
        tiebreak: str,
    ) -> Dict[str, object]:
        width = int(info.width)
        if strand == "-":
            start = int(seq.size - width - offset)
        else:
            start = int(offset)
        window = np.asarray(seq, dtype=np.int8)[start : start + width]
        if window.size != width:
            raise ValueError(f"Best-hit window out of bounds for TF '{tf}'.")
        if strand == "-":
            core = revcomp_int(window)
        else:
            core = window
        return {
            "best_score_raw": float(raw_llr),
            "offset": int(start),
            "best_start": int(start),
            "strand": str(strand),
            "width": int(width),
            "best_window_seq": SequenceState(window).to_string(),
            "best_core_seq": SequenceState(core).to_string(),
            "best_hit_tiebreak": str(tiebreak),
        }

    def compute_all_per_pwm_and_hits(
        self,
        seq: np.ndarray,
        seq_length: int,
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
        """
        Compute per-TF scaled values plus best-hit metadata in one pass.
        """
        per_tf: Dict[str, float] = {}
        hits: Dict[str, Dict[str, object]] = {}
        rev = (3 - seq)[::-1] if self.bidirectional else None
        for tf, info in self._cache.items():
            raw_llr, offset, strand, tiebreak = self._best_llr_and_location(seq, info, rev=rev)
            per_tf[tf] = self._scaled_value_from_raw_llr(tf, raw_llr, seq_length)
            hits[tf] = self._hit_from_scan(
                seq,
                info,
                tf=tf,
                raw_llr=raw_llr,
                offset=offset,
                strand=strand,
                tiebreak=tiebreak,
            )
        return per_tf, hits

    def normalized_llr_map(self, seq: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        rev = (3 - seq)[::-1] if self.bidirectional else None
        for tf, info in self._cache.items():
            raw_llr, *_ = self._best_llr_and_location(seq, info, rev=rev)
            num, denom = raw_llr - info.null_mean, info.consensus_llr - info.null_mean
            frac = 0.0 if denom <= 0 else max(0.0, num / denom)
            out[tf] = float(frac)
        return out

    def normalized_llr_components(self, seq: np.ndarray) -> list[float]:
        return list(self.normalized_llr_map(seq).values())

    def compute_all_per_pwm(self, seq: np.ndarray, seq_length: int) -> Dict[str, float]:
        """
        For each PWM (TF), compute exactly one “scaled” value, depending on self.scale:

          • "llr" : raw LLR
          • "z"   : z-score → (raw_llr − null_mean) / null_std
          • "logp": −log10(p_seq)
          • "normalized-llr": (raw_llr − null_mean) / (consensus_llr − null_mean)
          • "consensus-neglop-sum":
              (-log10(p_seq) / precomputed_neglogp(consensus_llr))

        For "consensus-neglop-sum", we cache consensus neglogp by sequence length
        because the Bonferroni correction depends on the number of windows.
        """
        out: Dict[str, float] = {}
        logger.debug("compute_all_per_pwm: seq_length=%d, scale=%s", seq_length, self.scale)
        rev = (3 - seq)[::-1] if self.bidirectional else None
        for tf, info in self._cache.items():
            raw_llr, offset, strand, _ = self._best_llr_and_location(seq, info, rev=rev)

            if self.scale == "llr":
                out[tf] = float(raw_llr)
                continue

            # If z-score requested:
            if self.scale == "z":
                # z = (raw_llr - mean_null) / std_null
                z_val = (raw_llr - info.null_mean) / info.null_std
                out[tf] = float(z_val)
                continue

            if self.scale == "normalized-llr":
                num, denom = raw_llr - info.null_mean, info.consensus_llr - info.null_mean
                frac = 0.0 if denom <= 0 else max(0.0, num / denom)
                out[tf] = float(frac)
                continue

            # For any neglogp‐based result (logp or consensus‐neglop-sum):
            neglogp_seq = self._per_pwm_neglogp(raw_llr, info, seq_length)
            if self.scale == "logp":
                out[tf] = float(neglogp_seq)
                continue

            # Now scale must be "consensus-neglop-sum"
            neglogp_cons = info.consensus_neglogp_by_len.get(seq_length)
            if neglogp_cons is None:
                cons_llr = info.consensus_llr
                neglogp_cons = self._per_pwm_neglogp(cons_llr, info, seq_length)
                info.consensus_neglogp_by_len[seq_length] = neglogp_cons
                logger.debug(
                    "    Set consensus_neglogp for %s (len=%d) = %.3f",
                    tf,
                    seq_length,
                    neglogp_cons,
                )

            if neglogp_cons > 0.0:
                normalized = float(neglogp_seq / neglogp_cons)
                out[tf] = normalized
            else:
                out[tf] = 0.0

        logger.debug("  Per-TF scaled map: %s", out)
        return out

    def _scaled_value_from_raw_llr(self, tf: str, raw_llr: float, seq_length: int) -> float:
        info = self._info(tf)
        if self.scale == "llr":
            return float(raw_llr)
        if self.scale == "z":
            return float((raw_llr - info.null_mean) / info.null_std)
        if self.scale == "normalized-llr":
            num, denom = raw_llr - info.null_mean, info.consensus_llr - info.null_mean
            frac = 0.0 if denom <= 0 else max(0.0, num / denom)
            return float(frac)
        neglogp_seq = self._per_pwm_neglogp(raw_llr, info, seq_length)
        if self.scale == "logp":
            return float(neglogp_seq)
        neglogp_cons = info.consensus_neglogp_by_len.get(seq_length)
        if neglogp_cons is None:
            cons_llr = info.consensus_llr
            neglogp_cons = self._per_pwm_neglogp(cons_llr, info, seq_length)
            info.consensus_neglogp_by_len[seq_length] = neglogp_cons
        if neglogp_cons > 0.0:
            return float(neglogp_seq / neglogp_cons)
        return 0.0

    def scaled_values_from_raw_llr(self, raw_llr_by_tf: Dict[str, float], seq_length: int) -> list[float]:
        """
        Compute scaled values from precomputed raw LLRs without allocating a per-TF dict.
        """
        values: list[float] = []
        for tf, raw_llr in raw_llr_by_tf.items():
            values.append(self._scaled_value_from_raw_llr(tf, raw_llr, seq_length))
        return values

    def scaled_from_raw_llr(self, raw_llr_by_tf: Dict[str, float], seq_length: int) -> Dict[str, float]:
        """
        Compute per-TF scaled values given precomputed raw LLRs.
        This mirrors compute_all_per_pwm without rescanning.
        """
        out: Dict[str, float] = {}
        for tf, raw_llr in raw_llr_by_tf.items():
            out[tf] = self._scaled_value_from_raw_llr(tf, raw_llr, seq_length)
        return out

    def make_local_cache(self, seq: np.ndarray) -> "LocalScanCache":
        return LocalScanCache(self, seq)


@dataclass(slots=True)
class _ScanCacheEntry:
    info: _PWMInfo
    fwd_scores: np.ndarray
    rev_scores: np.ndarray | None


class LocalScanCache:
    """
    Incremental cache for single-base flips. Uses per-PWM window scores to
    recompute best raw LLRs without full rescans.
    """

    SUPPORTED_SCALES = {"llr", "z", "logp", "normalized-llr", "consensus-neglop-sum"}

    def __init__(self, scorer: Scorer, seq: np.ndarray) -> None:
        self.scorer = scorer
        self.entries: Dict[str, _ScanCacheEntry] = {}
        self.seq = seq
        self.L = int(seq.size)
        self.bidirectional = bool(scorer.bidirectional)
        self.rebuild(seq)

    def rebuild(self, seq: np.ndarray) -> None:
        self.seq = seq
        self.L = int(seq.size)
        rev = (3 - seq)[::-1] if self.bidirectional else None
        self.entries = {}
        for tf, info in self.scorer._cache.items():
            fwd_scores = self._window_scores(seq, info.lom)
            rev_scores = self._window_scores(rev, info.lom) if self.bidirectional and rev is not None else None
            self.entries[tf] = _ScanCacheEntry(info=info, fwd_scores=fwd_scores, rev_scores=rev_scores)

    @staticmethod
    def _window_scores(seq: np.ndarray | None, lom: np.ndarray) -> np.ndarray:
        if seq is None:
            return np.array([], dtype=float)
        w = int(lom.shape[0])
        if seq.size < w:
            return np.array([], dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(seq, w)
        scores = lom[np.arange(w)[:, None], windows.T].sum(axis=0)
        return np.asarray(scores, dtype=float)

    @staticmethod
    def _max_outside(scores: np.ndarray, start: int, end: int) -> float:
        if scores.size == 0:
            return float("-inf")
        max_val = float("-inf")
        if start > 0:
            max_val = max(max_val, float(np.max(scores[:start])))
        if end + 1 < scores.size:
            max_val = max(max_val, float(np.max(scores[end + 1 :])))
        return max_val

    @staticmethod
    def _max_in_forward(
        *,
        entry: _ScanCacheEntry,
        pos: int,
        old_base_int: int,
        start: int,
        end: int,
    ) -> np.ndarray:
        max_in = np.full(4, float("-inf"), dtype=float)
        for offset in range(start, end + 1):
            j = pos - offset
            row = entry.info.lom[j]
            base_score = float(entry.fwd_scores[offset])
            old_val = float(row[old_base_int])
            for b in range(4):
                candidate = base_score + float(row[b] - old_val)
                if candidate > max_in[b]:
                    max_in[b] = candidate
        return max_in

    @staticmethod
    def _max_in_reverse(
        *,
        entry: _ScanCacheEntry,
        rev_pos: int,
        old_base_int: int,
        start: int,
        end: int,
    ) -> np.ndarray:
        if entry.rev_scores is None:
            return np.full(4, float("-inf"), dtype=float)
        max_in = np.full(4, float("-inf"), dtype=float)
        old_comp = 3 - old_base_int
        for offset in range(start, end + 1):
            j = rev_pos - offset
            row = entry.info.lom[j]
            base_score = float(entry.rev_scores[offset])
            old_val = float(row[old_comp])
            for b in range(4):
                new_comp = 3 - b
                candidate = base_score + float(row[new_comp] - old_val)
                if candidate > max_in[b]:
                    max_in[b] = candidate
        return max_in

    def candidate_raw_llr_maps(self, pos: int, old_base: int) -> list[Dict[str, float]]:
        """
        Return per-base raw LLR maps for a single position change.
        """
        L = self.L
        old_base_int = int(old_base)
        maps = [dict() for _ in range(4)]
        for tf, entry in self.entries.items():
            info = entry.info
            w = int(info.width)
            nwin = L - w + 1
            if nwin <= 0:
                for b in range(4):
                    maps[b][tf] = float("-inf")
                continue

            start = max(0, pos - w + 1)
            end = min(pos, nwin - 1)
            max_out_fwd = self._max_outside(entry.fwd_scores, start, end)
            max_in_fwd = self._max_in_forward(
                entry=entry,
                pos=pos,
                old_base_int=old_base_int,
                start=start,
                end=end,
            )
            best_fwd = np.maximum(max_out_fwd, max_in_fwd)

            if self.bidirectional and entry.rev_scores is not None:
                rev_pos = L - 1 - pos
                start_r = max(0, rev_pos - w + 1)
                end_r = min(rev_pos, nwin - 1)
                max_out_rev = self._max_outside(entry.rev_scores, start_r, end_r)
                max_in_rev = self._max_in_reverse(
                    entry=entry,
                    rev_pos=rev_pos,
                    old_base_int=old_base_int,
                    start=start_r,
                    end=end_r,
                )
                best_raw = np.maximum(best_fwd, np.maximum(max_out_rev, max_in_rev))
            else:
                best_raw = best_fwd

            for b in range(4):
                maps[b][tf] = float(best_raw[b])
        return maps

    def apply_base_change(self, pos: int, old_base: int, new_base: int) -> None:
        if int(old_base) == int(new_base):
            return
        L = self.L
        old_base_int = int(old_base)
        new_base_int = int(new_base)
        for entry in self.entries.values():
            info = entry.info
            w = int(info.width)
            nwin = L - w + 1
            if nwin <= 0:
                continue
            start = max(0, pos - w + 1)
            end = min(pos, nwin - 1)
            for o in range(start, end + 1):
                j = pos - o
                delta = info.lom[j, new_base_int] - info.lom[j, old_base_int]
                entry.fwd_scores[o] += delta
            if self.bidirectional and entry.rev_scores is not None:
                r = L - 1 - pos
                start_r = max(0, r - w + 1)
                end_r = min(r, nwin - 1)
                new_c = 3 - new_base_int
                old_c = 3 - old_base_int
                for o in range(start_r, end_r + 1):
                    j = r - o
                    delta = info.lom[j, new_c] - info.lom[j, old_c]
                    entry.rev_scores[o] += delta
