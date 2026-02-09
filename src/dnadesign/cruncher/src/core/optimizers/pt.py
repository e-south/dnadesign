"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/pt.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from dnadesign.cruncher.core.optimizers.base import Optimizer
from dnadesign.cruncher.core.optimizers.cooling import make_beta_ladder, make_beta_scheduler
from dnadesign.cruncher.core.optimizers.helpers import _replace_block, slide_window, swap_block
from dnadesign.cruncher.core.optimizers.policies import (
    MOVE_KINDS,
    AdaptiveMoveController,
    AdaptiveProposalController,
    AdaptiveSwapPairController,
    MoveSchedule,
    TargetingPolicy,
    move_probs_array,
    targeted_start,
)
from dnadesign.cruncher.core.optimizers.progress import ProgressAdapter, passthrough_progress
from dnadesign.cruncher.core.optimizers.telemetry import NullTelemetry, OptimizerTelemetry
from dnadesign.cruncher.core.scoring import LocalScanCache
from dnadesign.cruncher.core.sequence import canon_int, hamming_distance, revcomp_int
from dnadesign.cruncher.core.state import SequenceState, make_seed

logger = logging.getLogger(__name__)


class PTGibbsOptimizer(Optimizer):
    """Parallel-Tempered Gibbs/Metropolis sampler."""

    # Construction
    def __init__(
        self,
        evaluator: Any,  # SequenceEvaluator
        cfg: Dict[str, Any],
        rng: np.random.Generator,
        *,
        pwms: Dict[str, Any],
        init_cfg: Any | None,
        telemetry: OptimizerTelemetry | None = None,
        progress: ProgressAdapter | None = None,
    ) -> None:
        super().__init__(evaluator, cfg, rng)

        # Core dimensions
        self.draws: int = int(cfg["draws"])
        self.tune: int = int(cfg["tune"])
        self.chains: int = int(cfg["chains"])
        self.min_dist: int = int(cfg["min_dist"])
        self.top_k: int = int(cfg["top_k"])
        self.sequence_length: int = int(cfg["sequence_length"])
        self.swap_stride: int = int(cfg.get("swap_stride", 1))
        self.bidirectional: bool = bool(cfg.get("bidirectional", False))
        self.dsdna_canonicalize: bool = bool(cfg.get("dsdna_canonicalize", False))
        self.score_scale: str = str(cfg.get("score_scale") or "")
        self.record_tune: bool = bool(cfg.get("record_tune", False))
        self.progress_bar: bool = bool(cfg.get("progress_bar", True))
        self.progress_every: int = int(cfg.get("progress_every", 0))
        self.build_trace: bool = bool(cfg.get("build_trace", True))
        self.telemetry = telemetry or NullTelemetry()
        self.progress = progress or passthrough_progress
        early_cfg = cfg.get("early_stop") or {}
        self.early_stop_enabled = bool(early_cfg.get("enabled", False))
        self.early_stop_patience = int(early_cfg.get("patience", 0))
        self.early_stop_min_delta = float(early_cfg.get("min_delta", 0.0))
        self.early_stop_require_min_unique = bool(early_cfg.get("require_min_unique", False))
        self.early_stop_min_unique = int(early_cfg.get("min_unique", 0))
        self.early_stop_success_min_norm = float(early_cfg.get("success_min_per_tf_norm", 0.0))
        self.unique_successes: int | None = 0 if self.early_stop_require_min_unique else None
        self._unique_success_set: set[str] | None = set() if self.early_stop_require_min_unique else None
        if self.early_stop_patience <= 0:
            self.early_stop_enabled = False

        # β‑ladder (base)
        cooling_cfg = {"kind": cfg["kind"], "beta": cfg["beta"]}
        self.beta_ladder_base: List[float] = make_beta_ladder(cooling_cfg)
        if len(self.beta_ladder_base) != self.chains:
            raise ValueError(
                "Length of beta ladder (%d) must match number of chains (%d)"
                % (len(self.beta_ladder_base), self.chains)
            )
        self.beta_ladder: List[float] = list(self.beta_ladder_base)
        self.sweeps_per_chain = self.tune + self.draws
        self.total_sweeps = self.sweeps_per_chain

        # Move configuration
        self.move_cfg: Dict[str, Tuple[int, int] | int] = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }
        self._base_block_len_range: tuple[int, int] = tuple(int(v) for v in cfg["block_len_range"])
        self._base_multi_k_range: tuple[int, int] = tuple(int(v) for v in cfg["multi_k_range"])
        self.move_probs_start = move_probs_array(cfg["move_probs"])
        move_sched_cfg = cfg.get("move_schedule") or {}
        if move_sched_cfg.get("enabled"):
            end_probs = move_probs_array(move_sched_cfg["end"])
        else:
            end_probs = None
        self.move_schedule = MoveSchedule(start=self.move_probs_start, end=end_probs)
        adaptive_moves_cfg = cfg.get("adaptive_weights") or {}
        self.adaptive_moves_enabled = bool(adaptive_moves_cfg.get("enabled", False))
        self.move_controller = AdaptiveMoveController(
            enabled=self.adaptive_moves_enabled,
            window=int(adaptive_moves_cfg.get("window", 250)),
            k=float(adaptive_moves_cfg.get("k", 0.5)),
            min_prob=float(adaptive_moves_cfg.get("min_prob", 0.01)),
            max_prob=float(adaptive_moves_cfg.get("max_prob", 0.95)),
            targets=dict(adaptive_moves_cfg.get("targets") or {"S": 0.95, "B": 0.40, "M": 0.35, "I": 0.35}),
            kinds=tuple(adaptive_moves_cfg.get("kinds") or ("S", "B", "M", "I")),
        )
        proposal_adapt_cfg = cfg.get("proposal_adapt") or {}
        self.proposal_controller = AdaptiveProposalController(
            enabled=bool(proposal_adapt_cfg.get("enabled", False)),
            window=int(proposal_adapt_cfg.get("window", 250)),
            step=float(proposal_adapt_cfg.get("step", 0.10)),
            min_scale=float(proposal_adapt_cfg.get("min_scale", 0.50)),
            max_scale=float(proposal_adapt_cfg.get("max_scale", 2.0)),
            target_low=float(proposal_adapt_cfg.get("target_low", 0.25)),
            target_high=float(proposal_adapt_cfg.get("target_high", 0.75)),
        )
        target_prob = float(cfg.get("target_worst_tf_prob", 0.0))
        target_pad = int(cfg.get("target_window_pad", 0))
        self.targeting = TargetingPolicy(
            enabled=target_prob > 0.0,
            worst_tf_prob=target_prob,
            window_pad=target_pad,
        )
        self.insertion_consensus_prob = float(cfg.get("insertion_consensus_prob", 0.5))

        # Soft-min schedule (independent of PT temperature ladder)
        softmin_cfg = cfg.get("softmin") or {}
        self.softmin_enabled = bool(softmin_cfg.get("enabled", False))
        if self.softmin_enabled:
            softmin_sched = {k: v for k, v in softmin_cfg.items() if k in ("kind", "beta", "stages")}
            self.softmin_of = make_beta_scheduler(softmin_sched, self.total_sweeps)
        else:
            self.softmin_of = None

        # Adaptive swap controller
        self.adaptive_swap_cfg = cfg.get("adaptive_swap") or {}
        self.adaptive_swap_enabled = bool(self.adaptive_swap_cfg.get("enabled", False))
        if self.adaptive_swap_enabled:
            self.swap_controller = AdaptiveSwapPairController(
                n_pairs=max(0, self.chains - 1),
                target=float(self.adaptive_swap_cfg.get("target_swap", 0.25)),
                window=int(self.adaptive_swap_cfg.get("window", 50)),
                k=float(self.adaptive_swap_cfg.get("k", 0.5)),
                min_scale=float(self.adaptive_swap_cfg.get("min_scale", 0.25)),
                max_scale=float(self.adaptive_swap_cfg.get("max_scale", 4.0)),
                strict=bool(self.adaptive_swap_cfg.get("strict", False)),
                saturation_windows=int(self.adaptive_swap_cfg.get("saturation_windows", 5)),
                enabled=True,
            )
        else:
            self.swap_controller = None

        # References needed during optimisation
        self.pwms = pwms
        self.init_cfg = init_cfg
        if self.swap_stride < 1:
            raise ValueError("swap_stride must be >= 1")

        # Cache consensus and per-row probabilities for insertion moves
        self._insertion_consensus: Dict[str, np.ndarray] = {}
        self._insertion_row_probs: Dict[str, List[np.ndarray]] = {}
        for name, pwm in self.pwms.items():
            consensus = np.argmax(pwm.matrix, axis=1).astype(np.int8)
            self._insertion_consensus[name] = consensus
            row_probs = []
            for row in pwm.matrix:
                probs = row / row.sum()
                row_probs.append(np.asarray(probs, dtype=float))
            self._insertion_row_probs[name] = row_probs

        # Book‑keeping
        self.move_tally: Counter = Counter()
        self.accept_tally: Counter = Counter()
        self.swap_attempts: int = 0
        self.swap_accepts: int = 0
        self.swap_attempts_by_pair: List[int] = [0 for _ in range(max(0, self.chains - 1))]
        self.swap_accepts_by_pair: List[int] = [0 for _ in range(max(0, self.chains - 1))]
        self.move_stats: List[Dict[str, object]] = []
        self.swap_events: List[Dict[str, object]] = []
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_trace_meta: List[Dict[str, object]] = []
        self.all_scores: List[Dict[str, float]] = []
        self.elites_meta: List[Tuple[int, int]] = []
        self.trace_idata = None  # filled after optimise()
        self.best_score: float | None = None
        self.best_meta: Tuple[int, int] | None = None

    # Public API
    def _current_beta_ladder(self) -> list[float]:
        if self.swap_controller is None:
            return list(self.beta_ladder_base)
        return self.swap_controller.ladder_from_base(self.beta_ladder_base)

    def _scaled_beta_ladder(self, scale: float) -> list[float]:
        if not self.beta_ladder_base:
            return []
        beta_min = float(self.beta_ladder_base[0])
        beta_max = float(self.beta_ladder_base[-1])
        if beta_min <= 0:
            return [float(b) * float(scale) for b in self.beta_ladder_base]
        effective_beta_max = max(beta_min, beta_max * float(scale))
        if self.chains <= 1:
            return [beta_min]
        return list(np.geomspace(beta_min, effective_beta_max, self.chains))

    def optimise(self) -> List[SequenceState]:  # noqa: C901  (long but readable)
        """Run PT-MCMC and return *k* diverse elite sequences."""

        rng = self.rng
        evaluator = self.scorer  # SequenceEvaluator
        C, T, D = self.chains, self.tune, self.draws
        logger.debug("Starting PT optimisation: chains=%d  tune=%d  draws=%d", C, T, D)
        self.all_samples.clear()
        self.all_meta.clear()
        self.all_trace_meta.clear()
        self.all_scores.clear()
        self.move_stats.clear()
        self.swap_events.clear()
        self.move_tally.clear()
        self.accept_tally.clear()
        self.swap_attempts = 0
        self.swap_accepts = 0
        self.swap_attempts_by_pair = [0 for _ in range(max(0, C - 1))]
        self.swap_accepts_by_pair = [0 for _ in range(max(0, C - 1))]
        if self._unique_success_set is not None:
            self._unique_success_set.clear()
            self.unique_successes = 0

        def _perturb_seed(seed_arr: np.ndarray, n_mutations: int) -> np.ndarray:
            """Return a lightly perturbed copy of the seed for swap-friendly PT starts."""
            mutated = seed_arr.copy()
            if n_mutations <= 0 or mutated.size == 0:
                return mutated
            n_mutations = min(n_mutations, mutated.size)
            positions = rng.choice(mutated.size, size=n_mutations, replace=False)
            for pos in positions:
                current = int(mutated[pos])
                # Sample from {0,1,2,3} \ {current} uniformly.
                pick = int(rng.integers(0, 3))
                if pick >= current:
                    pick += 1
                mutated[pos] = np.int8(pick)
            return mutated

        # Seed each chain (shared seed + small perturbations for swap-friendly PT starts).
        if self.init_cfg is None:
            base_seed = SequenceState.random(self.sequence_length, rng).seq.copy()
        else:
            base_seed = make_seed(self.init_cfg, self.pwms, rng, sequence_length=self.sequence_length).seq.copy()
        chain_states = [base_seed.copy() for _ in range(C)]
        if C > 1:
            n_mutations = max(1, int(round(base_seed.size * 0.02)))
            for c in range(1, C):
                chain_states[c] = _perturb_seed(base_seed, n_mutations)
        chain_state_objs: List[SequenceState] = [SequenceState(chain_states[c], particle_id=c) for c in range(C)]
        scan_caches: List[LocalScanCache | None] = []
        scorer = getattr(evaluator, "scorer", None)
        for c in range(C):
            cache = None
            if scorer is not None and getattr(scorer, "scale", None) in LocalScanCache.SUPPORTED_SCALES:
                cache = scorer.make_local_cache(chain_states[c])
            scan_caches.append(cache)
        current_per_tf_maps: List[Dict[str, float]] = [evaluator(chain_state_objs[c]) for c in range(C)]

        # For trace: only *draw* phase (not tune) is stored per ArviZ convention.
        chain_scores: List[List[float]] = [[] for _ in range(C)]

        # Helper to record a state
        def _record(
            slot_id: int,
            sweep_idx: int,
            seq_arr: np.ndarray,
            per_tf: Dict[str, float],
            *,
            phase: str,
            beta: float,
            particle_id: int | None,
        ) -> Dict[str, float]:
            if particle_id is None:
                raise RuntimeError(f"PT trace invariant violated: missing particle_id for slot {slot_id}.")
            self.all_samples.append(seq_arr.copy())
            self.all_scores.append(per_tf)
            self.all_meta.append((slot_id, sweep_idx))
            self.all_trace_meta.append(
                {
                    "slot_id": int(slot_id),
                    "particle_id": int(particle_id),
                    "beta": float(beta),
                    "sweep_idx": int(sweep_idx),
                    "phase": str(phase),
                }
            )
            return per_tf

        def _record_unique_success(seq_arr: np.ndarray, per_tf: Dict[str, float]) -> None:
            if self._unique_success_set is None:
                return
            if self.score_scale == "normalized-llr":
                norm_values = list(per_tf.values())
            else:
                scorer = getattr(evaluator, "scorer", None)
                if scorer is None:
                    raise RuntimeError("early_stop.require_min_unique requires a scorer to compute normalized scores.")
                norm_values = list(scorer.normalized_llr_map(seq_arr).values())
            min_norm = min(norm_values) if norm_values else 0.0
            if min_norm < self.early_stop_success_min_norm:
                return
            if self.dsdna_canonicalize:
                key = SequenceState(canon_int(seq_arr)).to_string()
            else:
                key = SequenceState(seq_arr).to_string()
            if key not in self._unique_success_set:
                self._unique_success_set.add(key)
                self.unique_successes = len(self._unique_success_set)

        def _maybe_raise_tuning_limited(*, phase: str, sweep_idx: int) -> None:
            if self.swap_controller is None:
                return
            if self.swap_controller.tuning_limited():
                raise RuntimeError(
                    "PT swap adaptation tuning-limited: ladder scale saturated at max_scale "
                    f"for {self.swap_controller.saturated_windows_seen} windows "
                    f"(phase={phase}, sweep={sweep_idx})."
                )

        stop_after_tune = bool(self.adaptive_swap_cfg.get("stop_after_tune", True))

        # Burn‑in sweeps (record like Gibbs for consistency)
        for t in self.progress(range(T), desc="burn-in", leave=False, disable=not self.progress_bar):
            sweep_idx = t
            beta_softmin = self.softmin_of(sweep_idx) if self.softmin_of else None
            base_move_probs = self.move_schedule.probs(sweep_idx / max(self.total_sweeps - 1, 1))
            move_probs = self.move_controller.adapt(base_move_probs)
            block_range, multi_range = self.proposal_controller.current_ranges(
                self._base_block_len_range,
                self._base_multi_k_range,
                sequence_length=self.sequence_length,
            )
            self.move_cfg["block_len_range"] = block_range
            self.move_cfg["multi_k_range"] = multi_range
            self.beta_ladder = self._current_beta_ladder()
            current_scores: list[float] = []
            for c in range(C):
                per_tf_map = current_per_tf_maps[c]
                current_score = evaluator.combined_from_scores(
                    per_tf_map,
                    beta=beta_softmin,
                    length=chain_states[c].size,
                )
                move_kind, accepted, per_tf_map, new_score, move_detail = self._single_chain_move(
                    chain_states[c],
                    current_score,
                    self.beta_ladder[c],
                    beta_softmin,
                    evaluator,
                    rng,
                    move_probs,
                    state=chain_state_objs[c],
                    scan_cache=scan_caches[c],
                    per_tf=per_tf_map,
                )
                current_per_tf_maps[c] = per_tf_map
                current_scores.append(new_score)
                self.move_controller.record(move_kind, accepted=accepted)
                self.proposal_controller.record(move_kind, accepted=accepted)
                self.move_stats.append(
                    {
                        "sweep_idx": int(sweep_idx),
                        "phase": "tune",
                        "chain": int(c),
                        "move_kind": move_kind,
                        "attempted": 1,
                        "accepted": int(bool(accepted)),
                        "delta": move_detail.get("delta"),
                        "score_old": move_detail.get("score_old"),
                        "score_new": move_detail.get("score_new"),
                    }
                )

            # Optional swap attempts during tune (for adaptive swap calibration)
            if self.adaptive_swap_enabled and (sweep_idx % self.swap_stride) == 0:
                for c in range(C - 1):
                    self.swap_attempts += 1
                    self.swap_attempts_by_pair[c] += 1
                    s0, s1 = chain_states[c], chain_states[c + 1]
                    β0, β1 = self.beta_ladder[c], self.beta_ladder[c + 1]
                    f0 = current_scores[c]
                    f1 = current_scores[c + 1]
                    Δ = (β1 - β0) * (f0 - f1)
                    particle_lo_before = chain_state_objs[c].particle_id
                    particle_hi_before = chain_state_objs[c + 1].particle_id
                    accepted = False
                    log_u = None
                    if Δ >= 0:
                        accepted = True
                    else:
                        log_u = float(np.log(rng.random()))
                        accepted = bool(log_u < Δ)
                    if accepted:
                        chain_states[c], chain_states[c + 1] = s1, s0
                        chain_state_objs[c], chain_state_objs[c + 1] = (
                            chain_state_objs[c + 1],
                            chain_state_objs[c],
                        )
                        scan_caches[c], scan_caches[c + 1] = scan_caches[c + 1], scan_caches[c]
                        current_scores[c], current_scores[c + 1] = current_scores[c + 1], current_scores[c]
                        current_per_tf_maps[c], current_per_tf_maps[c + 1] = (
                            current_per_tf_maps[c + 1],
                            current_per_tf_maps[c],
                        )
                        self.swap_accepts += 1
                        self.swap_accepts_by_pair[c] += 1
                    if self.swap_controller is not None:
                        self.swap_controller.record(pair_idx=c, accepted=accepted)
                        if not stop_after_tune:
                            self.swap_controller.update()
                            _maybe_raise_tuning_limited(phase="tune", sweep_idx=sweep_idx)
                    self.swap_events.append(
                        {
                            "sweep_idx": int(sweep_idx),
                            "phase": "tune",
                            "slot_lo": int(c),
                            "slot_hi": int(c + 1),
                            "beta_lo": float(β0),
                            "beta_hi": float(β1),
                            "particle_lo_before": int(particle_lo_before),
                            "particle_hi_before": int(particle_hi_before),
                            "accepted": bool(accepted),
                            "delta": float(Δ),
                            "log_u": float(log_u) if log_u is not None else None,
                        }
                    )
                if self.swap_controller is not None and stop_after_tune:
                    self.swap_controller.update()
                    _maybe_raise_tuning_limited(phase="tune", sweep_idx=sweep_idx)

            if self.record_tune:
                for c in range(C):
                    _record(
                        c,
                        t,
                        chain_states[c],
                        current_per_tf_maps[c],
                        phase="tune",
                        beta=self.beta_ladder[c],
                        particle_id=chain_state_objs[c].particle_id,
                    )

            self._maybe_log_progress("burn-in", t + 1, T)

        # Sampling sweeps + swap attempts
        no_improve = 0
        best_global: float | None = None
        for d in self.progress(range(D), desc="sampling", leave=False, disable=not self.progress_bar):
            sweep_idx = T + d
            beta_softmin = self.softmin_of(sweep_idx) if self.softmin_of else None
            base_move_probs = self.move_schedule.probs(sweep_idx / max(self.total_sweeps - 1, 1))
            move_probs = self.move_controller.adapt(base_move_probs)
            block_range, multi_range = self.proposal_controller.current_ranges(
                self._base_block_len_range,
                self._base_multi_k_range,
                sequence_length=self.sequence_length,
            )
            self.move_cfg["block_len_range"] = block_range
            self.move_cfg["multi_k_range"] = multi_range
            self.beta_ladder = self._current_beta_ladder()

            # Within‑chain proposals
            current_scores: list[float] = []
            for c in range(C):
                per_tf_map = current_per_tf_maps[c]
                current_score = evaluator.combined_from_scores(
                    per_tf_map,
                    beta=beta_softmin,
                    length=chain_states[c].size,
                )
                move_kind, accepted, per_tf_map, new_score, move_detail = self._single_chain_move(
                    chain_states[c],
                    current_score,
                    self.beta_ladder[c],
                    beta_softmin,
                    evaluator,
                    rng,
                    move_probs,
                    state=chain_state_objs[c],
                    scan_cache=scan_caches[c],
                    per_tf=per_tf_map,
                )
                current_per_tf_maps[c] = per_tf_map
                current_scores.append(new_score)
                self.move_controller.record(move_kind, accepted=accepted)
                self.proposal_controller.record(move_kind, accepted=accepted)
                self.move_stats.append(
                    {
                        "sweep_idx": int(sweep_idx),
                        "phase": "draw",
                        "chain": int(c),
                        "move_kind": move_kind,
                        "attempted": 1,
                        "accepted": int(bool(accepted)),
                        "delta": move_detail.get("delta"),
                        "score_old": move_detail.get("score_old"),
                        "score_new": move_detail.get("score_new"),
                    }
                )

            # Pair‑wise swaps
            if (sweep_idx % self.swap_stride) == 0:
                for c in range(C - 1):
                    self.swap_attempts += 1
                    self.swap_attempts_by_pair[c] += 1
                    s0, s1 = chain_states[c], chain_states[c + 1]
                    β0, β1 = self.beta_ladder[c], self.beta_ladder[c + 1]
                    f0 = current_scores[c]
                    f1 = current_scores[c + 1]
                    Δ = (β1 - β0) * (f0 - f1)
                    particle_lo_before = chain_state_objs[c].particle_id
                    particle_hi_before = chain_state_objs[c + 1].particle_id
                    accepted = False
                    log_u = None
                    if Δ >= 0:
                        accepted = True
                    else:
                        log_u = float(np.log(rng.random()))
                        accepted = bool(log_u < Δ)
                    if accepted:
                        chain_states[c], chain_states[c + 1] = s1, s0  # swap in‑place
                        chain_state_objs[c], chain_state_objs[c + 1] = (
                            chain_state_objs[c + 1],
                            chain_state_objs[c],
                        )
                        scan_caches[c], scan_caches[c + 1] = scan_caches[c + 1], scan_caches[c]
                        current_scores[c], current_scores[c + 1] = current_scores[c + 1], current_scores[c]
                        current_per_tf_maps[c], current_per_tf_maps[c + 1] = (
                            current_per_tf_maps[c + 1],
                            current_per_tf_maps[c],
                        )
                        self.swap_accepts += 1
                        self.swap_accepts_by_pair[c] += 1
                    if self.swap_controller is not None:
                        self.swap_controller.record(pair_idx=c, accepted=accepted)
                        if not stop_after_tune:
                            self.swap_controller.update()
                            _maybe_raise_tuning_limited(phase="draw", sweep_idx=sweep_idx)
                    self.swap_events.append(
                        {
                            "sweep_idx": int(sweep_idx),
                            "phase": "draw",
                            "slot_lo": int(c),
                            "slot_hi": int(c + 1),
                            "beta_lo": float(β0),
                            "beta_hi": float(β1),
                            "particle_lo_before": int(particle_lo_before),
                            "particle_hi_before": int(particle_hi_before),
                            "accepted": bool(accepted),
                            "delta": float(Δ),
                            "log_u": float(log_u) if log_u is not None else None,
                        }
                    )
            for c in range(C):
                _record(
                    c,
                    T + d,
                    chain_states[c],
                    current_per_tf_maps[c],
                    phase="draw",
                    beta=self.beta_ladder[c],
                    particle_id=chain_state_objs[c].particle_id,
                )
                if self._unique_success_set is not None:
                    _record_unique_success(chain_states[c], current_per_tf_maps[c])
                comb = current_scores[c]
                if self.build_trace:
                    chain_scores[c].append(comb)
                if self.best_score is None or comb > self.best_score:
                    self.best_score = comb
                    self.best_meta = (c, T + d)

            score_mean = float(np.mean(current_scores)) if current_scores else None
            score_std = float(np.std(current_scores)) if current_scores else None
            current_best = float(max(current_scores)) if current_scores else None
            self._maybe_log_progress(
                "sampling",
                d + 1,
                D,
                current_score=current_best,
                score_mean=score_mean,
                score_std=score_std,
                beta_softmin=beta_softmin,
            )
            if self.early_stop_enabled and current_best is not None:
                if best_global is None or current_best > best_global + self.early_stop_min_delta:
                    best_global = current_best
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.early_stop_patience:
                        if (
                            self.early_stop_require_min_unique
                            and (self.unique_successes or 0) < self.early_stop_min_unique
                        ):
                            continue
                        logger.info(
                            "Early-stop: stalled for %d sweeps (min_delta=%.3f).",
                            self.early_stop_patience,
                            self.early_stop_min_delta,
                        )
                        self.telemetry.update(
                            status_message="early_stop",
                            early_stop={
                                "patience": self.early_stop_patience,
                                "min_delta": self.early_stop_min_delta,
                                "best_score": best_global,
                                "unique_successes": self.unique_successes,
                            },
                        )
                        break

        logger.debug("PT optimisation finished. Move utilisation: %s", dict(self.move_tally))

        # Build ArviZ trace from draw phase only when trace output is enabled.
        if self.build_trace:
            if chain_scores:
                max_len = max(len(scores) for scores in chain_scores)
                for scores in chain_scores:
                    if len(scores) < max_len:
                        scores.extend([float("nan")] * (max_len - len(scores)))
            score_arr = np.asarray(chain_scores, dtype=float)  # (C, D)
            az = importlib.import_module("arviz")
            self.trace_idata = az.from_dict(posterior={"score": score_arr})
        else:
            self.trace_idata = None

        if self.top_k <= 0:
            self.elites_meta = []
            return []

        # Rank all recorded sequences by combined fitness
        beta_softmin_final = self.softmin_of(self.total_sweeps - 1) if self.softmin_of else None
        ranked: List[Tuple[float, np.ndarray, int]] = []
        for idx, (seq, per_tf_map) in enumerate(zip(self.all_samples, self.all_scores)):
            val = evaluator.combined_from_scores(per_tf_map, beta=beta_softmin_final, length=int(seq.size))
            ranked.append((val, seq.copy(), idx))
        ranked.sort(key=lambda x: x[0], reverse=True)

        dist_fn = hamming_distance
        elites: List[SequenceState] = []
        picked_idx: List[int] = []
        for val, seq, idx in ranked:
            if len(elites) >= self.top_k:
                break
            if any(dist_fn(seq, e.seq) < self.min_dist for e in elites):
                continue
            elites.append(SequenceState(seq))
            picked_idx.append(idx)
        self.elites_meta = [self.all_meta[i] for i in picked_idx]
        return elites

    # Low‑level helpers
    def _single_chain_move(
        self,
        seq: np.ndarray,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        move_probs: np.ndarray,
        *,
        state: SequenceState,
        scan_cache: LocalScanCache | None = None,
        per_tf: Dict[str, float] | None = None,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, float | None]]:
        """One Gibbs-style proposal/accept cycle at inverse-temperature β."""

        L = seq.size
        if per_tf is None:
            per_tf = evaluator(state)
        move_kind = self._sample_move_kind(rng, move_probs)
        self.move_tally[move_kind] += 1
        accepted = False
        score_old = float(current_combined)

        def _detail(score_new: float) -> Dict[str, float | None]:
            return {
                "delta": float(score_new - score_old),
                "score_old": score_old,
                "score_new": float(score_new),
            }

        target = self.targeting.maybe_target(
            seq_len=L,
            state=state,
            evaluator=evaluator,
            rng=rng,
            per_tf=per_tf,
        )
        target_tf = target[0] if target else None
        target_window = (target[1], target[2]) if target else None

        # Single‑base flip
        if move_kind == "S":
            if target_window is not None:
                i = rng.integers(target_window[0], target_window[1])
            else:
                i = rng.integers(L)
            old = int(seq[i])
            lods = np.empty(4, float)
            combined_vals = np.empty(4, float)
            per_tf_candidates: list[Dict[str, float]] = []
            if scan_cache is not None:
                raw_candidates = scan_cache.candidate_raw_llr_maps(i, old)
                for b in range(4):
                    per_tf_b = evaluator.scorer.scaled_from_raw_llr(raw_candidates[b], L)
                    per_tf_candidates.append(per_tf_b)
                    comb_b = evaluator.combined_from_scores(per_tf_b, beta=beta_softmin, length=L)
                    combined_vals[b] = comb_b
                    lods[b] = β * comb_b
            else:
                for b in range(4):
                    seq[i] = b
                    per_tf_b, comb_b = evaluator.evaluate(state, beta=beta_softmin, length=L)
                    per_tf_candidates.append(per_tf_b)
                    combined_vals[b] = comb_b
                    lods[b] = β * comb_b
                seq[i] = old
            lods -= lods.max()
            probs = np.exp(lods)
            new_base = rng.choice(4, p=probs / probs.sum())
            seq[i] = new_base
            if scan_cache is not None:
                scan_cache.apply_base_change(i, old, int(new_base))
            self.accept_tally[move_kind] += 1
            accepted = True
            return (
                move_kind,
                accepted,
                per_tf_candidates[int(new_base)],
                float(combined_vals[new_base]),
                _detail(float(combined_vals[new_base])),
            )

        # Contiguous block replace
        if move_kind == "B":
            mn, mx = self.move_cfg["block_len_range"]
            length = rng.integers(mn, mx + 1)
            if length > L:
                return move_kind, False, per_tf, current_combined, _detail(current_combined)
            start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            proposal = rng.integers(0, 4, size=length)
            old_block = seq[start : start + length].copy()

            _replace_block(seq, start, length, proposal)
            new_per_tf, new_f = evaluator.evaluate(state, beta=beta_softmin, length=L)
            _replace_block(seq, start, length, old_block)
            Δ = β * (new_f - current_combined)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                _replace_block(seq, start, length, proposal)
                self.accept_tally[move_kind] += 1
                accepted = True
                if scan_cache is not None:
                    scan_cache.rebuild(seq)
                return move_kind, accepted, new_per_tf, new_f, _detail(new_f)
            return move_kind, accepted, per_tf, current_combined, _detail(new_f)

        # Multi‑site flip
        if move_kind == "M":
            kmin, kmax = self.move_cfg["multi_k_range"]
            k = rng.integers(kmin, kmax + 1)
            if k > L:
                k = L
            if target_window is not None and (target_window[1] - target_window[0]) >= k:
                idxs = rng.choice(np.arange(target_window[0], target_window[1]), size=k, replace=False)
            else:
                idxs = rng.choice(L, size=k, replace=False)
            old_bases = seq[idxs].copy()
            proposal = rng.integers(0, 4, size=k)

            seq[idxs] = proposal
            new_per_tf, new_f = evaluator.evaluate(state, beta=beta_softmin, length=L)
            seq[idxs] = old_bases
            Δ = β * (new_f - current_combined)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                seq[idxs] = proposal
                self.accept_tally[move_kind] += 1
                accepted = True
                if scan_cache is not None:
                    scan_cache.rebuild(seq)
                return move_kind, accepted, new_per_tf, new_f, _detail(new_f)
            return move_kind, accepted, per_tf, current_combined, _detail(new_f)

        if move_kind == "L":
            min_len, max_len = self.move_cfg["swap_len_range"]
            length = rng.integers(min_len, max_len + 1)
            max_shift = int(self.move_cfg["slide_max_shift"])
            if max_shift < 1 or length >= L:
                return move_kind, False, per_tf, current_combined, _detail(current_combined)
            shift = rng.integers(-max_shift, max_shift + 1)
            if shift == 0:
                shift = 1 if rng.random() < 0.5 else -1
            if shift > 0:
                min_start, max_start = 0, L - length - shift
            else:
                min_start, max_start = -shift, L - length
            if max_start < min_start:
                return move_kind, False, per_tf, current_combined, _detail(current_combined)
            start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            start = max(min_start, min(max_start, start))
            slide_window(seq, start, length, int(shift))
            new_per_tf, new_f = evaluator.evaluate(state, beta=beta_softmin, length=L)
            slide_window(seq, start + int(shift), length, int(-shift))
            Δ = β * (new_f - current_combined)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                slide_window(seq, start, length, int(shift))
                self.accept_tally[move_kind] += 1
                accepted = True
                if scan_cache is not None:
                    scan_cache.rebuild(seq)
                return move_kind, accepted, new_per_tf, new_f, _detail(new_f)
            return move_kind, accepted, per_tf, current_combined, _detail(new_f)

        if move_kind == "W":
            min_len, max_len = self.move_cfg["swap_len_range"]
            length = rng.integers(min_len, max_len + 1)
            if length >= L:
                return move_kind, False, per_tf, current_combined, _detail(current_combined)
            start_a = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            max_tries = 10
            start_b = start_a
            for _ in range(max_tries):
                start_b = rng.integers(0, L - length + 1)
                if abs(start_b - start_a) >= length:
                    break
            if abs(start_b - start_a) < length:
                return move_kind, False, per_tf, current_combined, _detail(current_combined)
            swap_block(seq, start_a, start_b, length)
            new_per_tf, new_f = evaluator.evaluate(state, beta=beta_softmin, length=L)
            swap_block(seq, start_a, start_b, length)
            Δ = β * (new_f - current_combined)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                swap_block(seq, start_a, start_b, length)
                self.accept_tally[move_kind] += 1
                accepted = True
                if scan_cache is not None:
                    scan_cache.rebuild(seq)
                return move_kind, accepted, new_per_tf, new_f, _detail(new_f)
            return move_kind, accepted, per_tf, current_combined, _detail(new_f)

        # Motif insertion proposal
        tf_names = list(self.pwms.keys())
        if not tf_names:
            return move_kind, False, per_tf, current_combined, _detail(current_combined)
        tf_name = target_tf if target_tf in self.pwms else rng.choice(tf_names)
        pwm = self.pwms[tf_name]
        width = pwm.length
        if width > L:
            return move_kind, False, per_tf, current_combined, _detail(current_combined)
        start = targeted_start(seq_len=L, block_len=width, target=target_window, rng=rng)
        if rng.random() < self.insertion_consensus_prob:
            proposal = self._insertion_consensus[tf_name].copy()
        else:
            row_probs = self._insertion_row_probs[tf_name]
            proposal = np.array([rng.choice(4, p=row) for row in row_probs], dtype=np.int8)
        if self.bidirectional and rng.random() < 0.5:
            proposal = revcomp_int(proposal)
        old_block = seq[start : start + width].copy()
        _replace_block(seq, start, width, proposal)
        new_per_tf, new_f = evaluator.evaluate(state, beta=beta_softmin, length=L)
        _replace_block(seq, start, width, old_block)
        Δ = β * (new_f - current_combined)
        if Δ >= 0 or np.log(rng.random()) < Δ:
            _replace_block(seq, start, width, proposal)
            self.accept_tally[move_kind] += 1
            accepted = True
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            return move_kind, accepted, new_per_tf, new_f, _detail(new_f)
        return move_kind, accepted, per_tf, current_combined, _detail(new_f)

    def _sample_move_kind(self, rng: np.random.Generator, move_probs: np.ndarray) -> str:
        return str(rng.choice(MOVE_KINDS, p=move_probs))

    def _maybe_log_progress(
        self,
        phase: str,
        step: int,
        total: int,
        *,
        beta_softmin: float | None = None,
        current_score: float | None = None,
        score_mean: float | None = None,
        score_std: float | None = None,
    ) -> None:
        if not self.progress_every:
            return
        if step % self.progress_every != 0 and step != total:
            return
        totals = dict(self.move_tally)
        accepted = dict(self.accept_tally)
        acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
        acc_label = ", ".join(f"{k}={acceptance_rate[k]:.2f}" for k in sorted(acceptance_rate))
        mh_kinds = [k for k in totals if k != "S"]
        mh_total = sum(totals.get(k, 0) for k in mh_kinds)
        mh_accept = sum(accepted.get(k, 0) for k in mh_kinds)
        acceptance_rate_mh = (mh_accept / mh_total) if mh_total else 0.0
        all_total = sum(totals.values())
        all_accept = sum(accepted.values())
        acceptance_rate_all = (all_accept / all_total) if all_total else 0.0
        pct = (step / total) * 100 if total else 100.0
        swap_rate = self.swap_accepts / self.swap_attempts if self.swap_attempts else 0.0
        score_blob = ""
        if current_score is not None:
            score_blob = f" score={current_score:.3f}"
        if score_mean is not None and score_std is not None:
            score_blob += f" mean={score_mean:.3f}±{score_std:.3f}"
        if self.best_score is not None:
            score_blob += f" best={self.best_score:.3f}"
        logger.info(
            "Progress: %s %d/%d (%.1f%%) accept={%s} swap_acc=%.2f%s",
            phase,
            step,
            total,
            pct,
            acc_label,
            swap_rate,
            score_blob,
        )
        self.telemetry.update(
            phase=phase,
            step=step,
            total=total,
            progress_pct=round(pct, 2),
            acceptance_rate=acceptance_rate,
            acceptance_rate_mh=acceptance_rate_mh,
            acceptance_rate_all=acceptance_rate_all,
            swap_accepts=self.swap_accepts,
            swap_attempts=self.swap_attempts,
            swap_rate=swap_rate,
            current_score=current_score,
            score_mean=score_mean,
            score_std=score_std,
            beta_softmin=beta_softmin,
            beta_min=min(self.beta_ladder) if self.beta_ladder else None,
            beta_max=max(self.beta_ladder) if self.beta_ladder else None,
            best_score=self.best_score,
            best_chain=(self.best_meta[0] + 1) if self.best_meta else None,
            best_draw=(self.best_meta[1]) if self.best_meta else None,
        )

    def stats(self) -> Dict[str, object]:
        totals = dict(self.move_tally)
        accepted = dict(self.accept_tally)
        acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
        mh_kinds = [k for k in totals if k != "S"]
        mh_total = sum(totals.get(k, 0) for k in mh_kinds)
        mh_accept = sum(accepted.get(k, 0) for k in mh_kinds)
        acceptance_rate_mh = (mh_accept / mh_total) if mh_total else 0.0
        all_total = sum(totals.values())
        all_accept = sum(accepted.values())
        acceptance_rate_all = (all_accept / all_total) if all_total else 0.0
        swap_rate = self.swap_accepts / self.swap_attempts if self.swap_attempts else 0.0
        eps = 1.0e-12
        mh_deltas = [
            abs(float(ms.get("delta", 0.0)))
            for ms in self.move_stats
            if ms.get("move_kind") in mh_kinds and ms.get("delta") is not None
        ]
        if mh_deltas:
            delta_abs_median_mh = float(np.median(mh_deltas))
            delta_frac_zero_mh = float(np.mean([d <= eps for d in mh_deltas]))
            score_change_rate_mh = float(np.mean([d > eps for d in mh_deltas]))
        else:
            delta_abs_median_mh = None
            delta_frac_zero_mh = None
            score_change_rate_mh = None
        beta_min = float(self.beta_ladder_base[0]) if self.beta_ladder_base else None
        beta_max_base = float(self.beta_ladder_base[-1]) if self.beta_ladder_base else None
        beta_max_final = float(max(self.beta_ladder)) if self.beta_ladder else None
        beta_mode = "pair_adaptive_log_gap" if self.swap_controller is not None else "fixed"
        swap_pair_scales = self.swap_controller.pair_scales if self.swap_controller is not None else None
        tuning_limited = self.swap_controller.tuning_limited() if self.swap_controller is not None else False
        swap_saturation_windows = self.swap_controller.saturated_windows_seen if self.swap_controller is not None else 0
        return {
            "moves": totals,
            "accepted": accepted,
            "acceptance_rate": acceptance_rate,
            "acceptance_rate_mh": acceptance_rate_mh,
            "acceptance_rate_all": acceptance_rate_all,
            "delta_abs_median_mh": delta_abs_median_mh,
            "delta_frac_zero_mh": delta_frac_zero_mh,
            "score_change_rate_mh": score_change_rate_mh,
            "swap_attempts": self.swap_attempts,
            "swap_accepts": self.swap_accepts,
            "swap_acceptance_rate": swap_rate,
            "swap_attempts_by_pair": list(self.swap_attempts_by_pair),
            "swap_accepts_by_pair": list(self.swap_accepts_by_pair),
            "beta_ladder_base": list(self.beta_ladder_base),
            "beta_ladder_final": list(self.beta_ladder),
            "beta_ladder_scale_final": float(self.swap_controller.scale) if self.swap_controller else 1.0,
            "beta_ladder_scale_mode": beta_mode,
            "swap_pair_scales_final": swap_pair_scales,
            "swap_saturation_windows": swap_saturation_windows,
            "swap_tuning_limited": tuning_limited,
            "beta_min": beta_min,
            "beta_max_base": beta_max_base,
            "beta_max_final": beta_max_final,
            "adaptive_moves_enabled": bool(self.move_controller.enabled),
            "proposal_adapt_enabled": bool(self.proposal_controller.enabled),
            "proposal_block_len_range_final": list(self.move_cfg["block_len_range"]),
            "proposal_multi_k_range_final": list(self.move_cfg["multi_k_range"]),
            "unique_successes": self.unique_successes,
            "move_stats": list(self.move_stats),
            "swap_events": list(self.swap_events),
            "final_softmin_beta": self.final_softmin_beta(),
            "final_mcmc_beta": self.final_mcmc_beta(),
        }

    def final_softmin_beta(self) -> float | None:
        if self.softmin_of is None:
            return None
        return float(self.softmin_of(self.total_sweeps - 1))

    def final_mcmc_beta(self) -> float | None:
        if not self.beta_ladder:
            return None
        return float(max(self.beta_ladder))

    def objective_schedule_summary(self) -> Dict[str, object]:
        beta_mode = "pair_adaptive_log_gap" if self.swap_controller is not None else "fixed"
        return {
            "total_sweeps": self.total_sweeps,
            "beta_ladder_base": list(self.beta_ladder_base),
            "beta_ladder_final": list(self.beta_ladder),
            "beta_ladder_scale_final": float(self.swap_controller.scale) if self.swap_controller else 1.0,
            "beta_ladder_scale_mode": beta_mode,
            "swap_pair_scales_final": self.swap_controller.pair_scales if self.swap_controller is not None else None,
        }
