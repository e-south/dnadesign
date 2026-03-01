"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/gibbs_anneal.py

Gibbs annealing optimizer for sequence design.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from dnadesign.cruncher.core.optimizers.base import Optimizer
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.optimizers.gibbs_postprocess import (
    build_trace_idata,
    make_move_stat_entry,
    select_diverse_elites,
)
from dnadesign.cruncher.core.optimizers.gibbs_runtime import (
    accept_metropolis,
    gibbs_stay_probability,
    maybe_log_progress,
    move_adaptation_frozen,
    move_detail,
    proposal_adaptation_frozen,
    sample_move_kind,
)
from dnadesign.cruncher.core.optimizers.gibbs_summary import (
    build_objective_schedule_summary,
    build_stats,
    final_mcmc_beta,
    final_softmin_beta,
)
from dnadesign.cruncher.core.optimizers.helpers import _replace_block, slide_window, swap_block
from dnadesign.cruncher.core.optimizers.policies import (
    AdaptiveMoveController,
    AdaptiveProposalController,
    MoveSchedule,
    TargetingPolicy,
    move_probs_array,
    targeted_start,
)
from dnadesign.cruncher.core.optimizers.progress import ProgressAdapter, passthrough_progress
from dnadesign.cruncher.core.optimizers.telemetry import NullTelemetry, OptimizerTelemetry
from dnadesign.cruncher.core.scoring import LocalScanCache
from dnadesign.cruncher.core.sequence import canon_int, revcomp_int
from dnadesign.cruncher.core.state import SequenceState, make_seed

logger = logging.getLogger(__name__)


class GibbsAnnealOptimizer(Optimizer):
    """Gibbs/Metropolis sampler with an annealing schedule."""

    _UNSUPPORTED_REPLICA_EXCHANGE_KEYS: tuple[str, ...] = (
        "adaptive_swap",
        "swap_stride",
        "n_temps",
        "temp_max",
    )

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

        unexpected = [key for key in self._UNSUPPORTED_REPLICA_EXCHANGE_KEYS if key in cfg]
        if unexpected:
            keys = ", ".join(sorted(unexpected))
            raise ValueError(
                f"Replica-exchange optimizer settings are unsupported for optimizer kind='gibbs_anneal': {keys}"
            )

        # Core dimensions
        self.draws: int = int(cfg["draws"])
        self.tune: int = int(cfg["tune"])
        self.chains: int = int(cfg["chains"])
        self.min_dist: int = int(cfg["min_dist"])
        self.top_k: int = int(cfg["top_k"])
        self.sequence_length: int = int(cfg["sequence_length"])
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

        # MCMC annealing schedule (shared across independent chains).
        self.sweeps_per_chain = self.tune + self.draws
        self.total_sweeps = self.sweeps_per_chain
        cooling_cfg_raw = cfg.get("mcmc_cooling") or {"kind": "fixed", "beta": 1.0}
        cooling_kind = str(cooling_cfg_raw.get("kind") or "").strip().lower()
        if cooling_kind == "fixed":
            beta = float(cooling_cfg_raw["beta"])
            self.mcmc_cooling_cfg = {"kind": "fixed", "beta": beta}
            self.mcmc_cooling_summary = {"kind": "fixed", "beta": beta}
        elif cooling_kind == "linear":
            beta_start = float(cooling_cfg_raw["beta_start"])
            beta_end = float(cooling_cfg_raw["beta_end"])
            self.mcmc_cooling_cfg = {
                "kind": "linear",
                "beta": (beta_start, beta_end),
            }
            self.mcmc_cooling_summary = {"kind": "linear", "beta_start": beta_start, "beta_end": beta_end}
        elif cooling_kind == "piecewise":
            stages_raw = cooling_cfg_raw.get("stages")
            if not isinstance(stages_raw, list) or not stages_raw:
                raise ValueError("mcmc_cooling.stages must be a non-empty list when kind='piecewise'")
            stages: list[dict[str, float | int]] = []
            for stage in stages_raw:
                if not isinstance(stage, dict):
                    raise ValueError("mcmc_cooling.stages must contain dictionaries with sweeps and beta")
                stages.append(
                    {
                        "sweeps": int(stage["sweeps"]),
                        "beta": float(stage["beta"]),
                    }
                )
            self.mcmc_cooling_cfg = {"kind": "piecewise", "stages": stages}
            self.mcmc_cooling_summary = {"kind": "piecewise", "stages": list(stages)}
        else:
            raise ValueError(f"Unsupported mcmc_cooling kind: {cooling_kind!r}")
        self.mcmc_beta_of = make_beta_scheduler(self.mcmc_cooling_cfg, self.total_sweeps)
        beta_start = float(self.mcmc_beta_of(0))
        self.beta_ladder_base: List[float] = [beta_start for _ in range(self.chains)]
        self.beta_ladder: List[float] = list(self.beta_ladder_base)

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
        freeze_moves_sweep = adaptive_moves_cfg.get("freeze_after_sweep")
        freeze_moves_beta = adaptive_moves_cfg.get("freeze_after_beta")
        self.move_adapt_freeze_after_sweep: int | None = (
            int(freeze_moves_sweep) if freeze_moves_sweep is not None else None
        )
        self.move_adapt_freeze_after_beta: float | None = (
            float(freeze_moves_beta) if freeze_moves_beta is not None else None
        )
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
        freeze_prop_sweep = proposal_adapt_cfg.get("freeze_after_sweep")
        freeze_prop_beta = proposal_adapt_cfg.get("freeze_after_beta")
        self.proposal_adapt_freeze_after_sweep: int | None = (
            int(freeze_prop_sweep) if freeze_prop_sweep is not None else None
        )
        self.proposal_adapt_freeze_after_beta: float | None = (
            float(freeze_prop_beta) if freeze_prop_beta is not None else None
        )
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
        gibbs_inertia_cfg = cfg.get("gibbs_inertia") or {}
        self.gibbs_inertia_enabled = bool(gibbs_inertia_cfg.get("enabled", False))
        self.gibbs_inertia_kind = str(gibbs_inertia_cfg.get("kind") or "linear").strip().lower()
        self.gibbs_inertia_p_stay_start = float(gibbs_inertia_cfg.get("p_stay_start", 0.0))
        self.gibbs_inertia_p_stay_end = float(gibbs_inertia_cfg.get("p_stay_end", 0.0))
        if self.gibbs_inertia_kind not in {"fixed", "linear"}:
            raise ValueError(f"Unsupported gibbs_inertia kind: {self.gibbs_inertia_kind!r}")

        # Soft-min schedule (independent of MCMC cooling schedule)
        softmin_cfg = cfg.get("softmin") or {}
        self.softmin_enabled = bool(softmin_cfg.get("enabled", False))
        if self.softmin_enabled:
            softmin_sched = {k: v for k, v in softmin_cfg.items() if k in ("kind", "beta", "stages")}
            self.softmin_of = make_beta_scheduler(softmin_sched, self.total_sweeps)
        else:
            self.softmin_of = None

        # References needed during optimisation
        self.pwms = pwms
        self.init_cfg = init_cfg
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
        self.move_stats: List[Dict[str, object]] = []
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_trace_meta: List[Dict[str, object]] = []
        self.all_scores: List[Dict[str, float]] = []
        self.elites_meta: List[Tuple[int, int]] = []
        self.trace_idata = None  # filled after optimise()
        self.best_score: float | None = None
        self.best_meta: Tuple[int, int] | None = None

    # Public API
    def _current_beta_ladder(self, sweep_idx: int) -> list[float]:
        beta_now = float(self.mcmc_beta_of(int(sweep_idx)))
        return [beta_now for _ in range(self.chains)]

    def _reset_optimise_state(self) -> None:
        self.all_samples.clear()
        self.all_meta.clear()
        self.all_trace_meta.clear()
        self.all_scores.clear()
        self.move_stats.clear()
        self.move_tally.clear()
        self.accept_tally.clear()
        if self._unique_success_set is not None:
            self._unique_success_set.clear()
            self.unique_successes = 0

    def _perturb_seed(
        self,
        seed_arr: np.ndarray,
        *,
        n_mutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mutated = seed_arr.copy()
        if n_mutations <= 0 or mutated.size == 0:
            return mutated
        n_mutations = min(n_mutations, mutated.size)
        positions = rng.choice(mutated.size, size=n_mutations, replace=False)
        for pos in positions:
            current = int(mutated[pos])
            pick = int(rng.integers(0, 3))
            if pick >= current:
                pick += 1
            mutated[pos] = np.int8(pick)
        return mutated

    def _initialize_chain_runtime(
        self,
        *,
        rng: np.random.Generator,
        evaluator: Any,
    ) -> tuple[
        list[np.ndarray],
        list[SequenceState],
        list[LocalScanCache | None],
        list[Dict[str, float]],
    ]:
        if self.init_cfg is None:
            base_seed = SequenceState.random(self.sequence_length, rng).seq.copy()
        else:
            base_seed = make_seed(self.init_cfg, self.pwms, rng, sequence_length=self.sequence_length).seq.copy()
        chain_states = [base_seed.copy() for _ in range(self.chains)]
        if self.chains > 1:
            n_mutations = max(1, int(round(base_seed.size * 0.02)))
            for c in range(1, self.chains):
                chain_states[c] = self._perturb_seed(base_seed, n_mutations=n_mutations, rng=rng)
        chain_state_objs: List[SequenceState] = [
            SequenceState(chain_states[c], particle_id=c) for c in range(self.chains)
        ]
        scan_caches: List[LocalScanCache | None] = []
        scorer = getattr(evaluator, "scorer", None)
        for c in range(self.chains):
            cache = None
            if scorer is not None and getattr(scorer, "scale", None) in LocalScanCache.SUPPORTED_SCALES:
                cache = scorer.make_local_cache(chain_states[c])
            scan_caches.append(cache)
        current_per_tf_maps: List[Dict[str, float]] = [evaluator(chain_state_objs[c]) for c in range(self.chains)]
        return chain_states, chain_state_objs, scan_caches, current_per_tf_maps

    def _record_state(
        self,
        *,
        slot_id: int,
        sweep_idx: int,
        seq_arr: np.ndarray,
        per_tf: Dict[str, float],
        phase: str,
        beta: float,
        particle_id: int | None,
    ) -> Dict[str, float]:
        if particle_id is None:
            raise RuntimeError(f"Gibbs trace invariant violated: missing particle_id for slot {slot_id}.")
        self.all_samples.append(seq_arr.copy())
        self.all_scores.append(per_tf)
        self.all_meta.append((slot_id, sweep_idx))
        self.all_trace_meta.append(
            {
                "chain": int(slot_id),
                "slot_id": int(slot_id),
                "particle_id": int(particle_id),
                "beta": float(beta),
                "sweep_idx": int(sweep_idx),
                "phase": str(phase),
            }
        )
        return per_tf

    def _record_unique_success(
        self,
        *,
        seq_arr: np.ndarray,
        per_tf: Dict[str, float],
        evaluator: Any,
    ) -> None:
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

    def _configure_sweep(
        self,
        *,
        sweep_idx: int,
    ) -> tuple[float | None, np.ndarray, bool, bool]:
        beta_now = float(self.mcmc_beta_of(sweep_idx))
        beta_softmin = self.softmin_of(sweep_idx) if self.softmin_of else None
        base_move_probs = self.move_schedule.probs(sweep_idx / max(self.total_sweeps - 1, 1))
        move_adapt_frozen = move_adaptation_frozen(
            sweep_idx=sweep_idx,
            beta_now=beta_now,
            freeze_after_sweep=self.move_adapt_freeze_after_sweep,
            freeze_after_beta=self.move_adapt_freeze_after_beta,
        )
        proposal_adapt_frozen = proposal_adaptation_frozen(
            sweep_idx=sweep_idx,
            beta_now=beta_now,
            freeze_after_sweep=self.proposal_adapt_freeze_after_sweep,
            freeze_after_beta=self.proposal_adapt_freeze_after_beta,
        )
        move_probs = base_move_probs if move_adapt_frozen else self.move_controller.adapt(base_move_probs)
        if proposal_adapt_frozen:
            block_range, multi_range = self._base_block_len_range, self._base_multi_k_range
        else:
            block_range, multi_range = self.proposal_controller.current_ranges(
                self._base_block_len_range,
                self._base_multi_k_range,
                sequence_length=self.sequence_length,
            )
        self.move_cfg["block_len_range"] = block_range
        self.move_cfg["multi_k_range"] = multi_range
        self.beta_ladder = self._current_beta_ladder(sweep_idx)
        return beta_softmin, move_probs, move_adapt_frozen, proposal_adapt_frozen

    def _run_chain_sweep_moves(
        self,
        *,
        phase: str,
        sweep_idx: int,
        beta_softmin: float | None,
        move_probs: np.ndarray,
        move_adapt_frozen: bool,
        proposal_adapt_frozen: bool,
        evaluator: Any,
        rng: np.random.Generator,
        chain_states: list[np.ndarray],
        chain_state_objs: list[SequenceState],
        scan_caches: list[LocalScanCache | None],
        current_per_tf_maps: list[Dict[str, float]],
    ) -> list[float]:
        current_scores: list[float] = []
        for c in range(self.chains):
            per_tf_map = current_per_tf_maps[c]
            current_score = evaluator.combined_from_scores(
                per_tf_map,
                beta=beta_softmin,
                length=chain_states[c].size,
            )
            move_kind, accepted, per_tf_map, new_score, move_detail_payload = self._single_chain_move(
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
                sweep_idx=sweep_idx,
            )
            current_per_tf_maps[c] = per_tf_map
            current_scores.append(new_score)
            if not move_adapt_frozen:
                self.move_controller.record(move_kind, accepted=accepted)
            if not proposal_adapt_frozen:
                self.proposal_controller.record(move_kind, accepted=accepted)
            self.move_stats.append(
                make_move_stat_entry(
                    sweep_idx=sweep_idx,
                    phase=phase,
                    chain_index=c,
                    move_kind=move_kind,
                    accepted=accepted,
                    move_detail_payload=move_detail_payload,
                )
            )
        return current_scores

    def optimise(self) -> List[SequenceState]:  # noqa: C901  (long but readable)
        """Run Gibbs annealing and return *k* diverse elite sequences."""

        rng = self.rng
        evaluator = self.scorer  # SequenceEvaluator
        C, T, D = self.chains, self.tune, self.draws
        logger.debug("Starting Gibbs annealing: chains=%d  tune=%d  draws=%d", C, T, D)
        self._reset_optimise_state()
        chain_states, chain_state_objs, scan_caches, current_per_tf_maps = self._initialize_chain_runtime(
            rng=rng,
            evaluator=evaluator,
        )
        chain_scores: List[List[float]] = [[] for _ in range(C)]

        # Burn‑in sweeps (record like Gibbs for consistency)
        for t in self.progress(range(T), desc="burn-in", leave=False, disable=not self.progress_bar):
            sweep_idx = t
            beta_softmin, move_probs, move_adapt_frozen, proposal_adapt_frozen = self._configure_sweep(
                sweep_idx=sweep_idx
            )
            self._run_chain_sweep_moves(
                phase="tune",
                sweep_idx=sweep_idx,
                beta_softmin=beta_softmin,
                move_probs=move_probs,
                move_adapt_frozen=move_adapt_frozen,
                proposal_adapt_frozen=proposal_adapt_frozen,
                evaluator=evaluator,
                rng=rng,
                chain_states=chain_states,
                chain_state_objs=chain_state_objs,
                scan_caches=scan_caches,
                current_per_tf_maps=current_per_tf_maps,
            )

            if self.record_tune:
                for c in range(C):
                    self._record_state(
                        slot_id=c,
                        sweep_idx=t,
                        seq_arr=chain_states[c],
                        per_tf=current_per_tf_maps[c],
                        phase="tune",
                        beta=self.beta_ladder[c],
                        particle_id=chain_state_objs[c].particle_id,
                    )

            maybe_log_progress(
                logger=logger,
                telemetry=self.telemetry,
                progress_every=self.progress_every,
                phase="burn-in",
                step=t + 1,
                total=T,
                move_tally=self.move_tally,
                accept_tally=self.accept_tally,
                best_score=self.best_score,
                best_meta=self.best_meta,
                beta_ladder=self.beta_ladder,
            )

        # Sampling sweeps + swap attempts
        no_improve = 0
        best_global: float | None = None
        for d in self.progress(range(D), desc="sampling", leave=False, disable=not self.progress_bar):
            sweep_idx = T + d
            beta_softmin, move_probs, move_adapt_frozen, proposal_adapt_frozen = self._configure_sweep(
                sweep_idx=sweep_idx
            )
            current_scores = self._run_chain_sweep_moves(
                phase="draw",
                sweep_idx=sweep_idx,
                beta_softmin=beta_softmin,
                move_probs=move_probs,
                move_adapt_frozen=move_adapt_frozen,
                proposal_adapt_frozen=proposal_adapt_frozen,
                evaluator=evaluator,
                rng=rng,
                chain_states=chain_states,
                chain_state_objs=chain_state_objs,
                scan_caches=scan_caches,
                current_per_tf_maps=current_per_tf_maps,
            )

            for c in range(C):
                self._record_state(
                    slot_id=c,
                    sweep_idx=T + d,
                    seq_arr=chain_states[c],
                    per_tf=current_per_tf_maps[c],
                    phase="draw",
                    beta=self.beta_ladder[c],
                    particle_id=chain_state_objs[c].particle_id,
                )
                if self._unique_success_set is not None:
                    self._record_unique_success(
                        seq_arr=chain_states[c],
                        per_tf=current_per_tf_maps[c],
                        evaluator=evaluator,
                    )
                comb = current_scores[c]
                if self.build_trace:
                    chain_scores[c].append(comb)
                if self.best_score is None or comb > self.best_score:
                    self.best_score = comb
                    self.best_meta = (c, T + d)

            score_mean = float(np.mean(current_scores)) if current_scores else None
            score_std = float(np.std(current_scores)) if current_scores else None
            current_best = float(max(current_scores)) if current_scores else None
            maybe_log_progress(
                logger=logger,
                telemetry=self.telemetry,
                progress_every=self.progress_every,
                phase="sampling",
                step=d + 1,
                total=D,
                move_tally=self.move_tally,
                accept_tally=self.accept_tally,
                best_score=self.best_score,
                best_meta=self.best_meta,
                beta_ladder=self.beta_ladder,
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

        logger.debug("Gibbs annealing finished. Move utilisation: %s", dict(self.move_tally))
        self.trace_idata = build_trace_idata(build_trace=self.build_trace, chain_scores=chain_scores)
        beta_softmin_final = self.softmin_of(self.total_sweeps - 1) if self.softmin_of else None
        elites, elites_meta = select_diverse_elites(
            top_k=self.top_k,
            min_dist=self.min_dist,
            all_samples=self.all_samples,
            all_scores=self.all_scores,
            all_meta=self.all_meta,
            beta_softmin_final=beta_softmin_final,
            combined_from_scores=lambda per_tf_map, beta, length: evaluator.combined_from_scores(
                per_tf_map,
                beta=beta,
                length=length,
            ),
        )
        self.elites_meta = elites_meta
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
        sweep_idx: int,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        """One Gibbs-style proposal/accept cycle at inverse-temperature β."""

        L = seq.size
        seq_before = seq.copy()
        if per_tf is None:
            per_tf = evaluator(state)
        move_kind = sample_move_kind(rng=rng, move_probs=move_probs)
        self.move_tally[move_kind] += 1
        score_old = float(current_combined)

        target = self.targeting.maybe_target(
            seq_len=L,
            state=state,
            evaluator=evaluator,
            rng=rng,
            per_tf=per_tf,
        )
        target_tf = target[0] if target else None
        target_window = (target[1], target[2]) if target else None

        if move_kind == "S":
            return self._run_single_base_move(
                seq=seq,
                score_old=score_old,
                β=β,
                beta_softmin=beta_softmin,
                evaluator=evaluator,
                rng=rng,
                state=state,
                scan_cache=scan_cache,
                sweep_idx=sweep_idx,
                target_window=target_window,
                seq_before=seq_before,
            )
        if move_kind == "B":
            return self._run_block_move(
                seq=seq,
                score_old=score_old,
                current_combined=current_combined,
                β=β,
                beta_softmin=beta_softmin,
                evaluator=evaluator,
                rng=rng,
                state=state,
                scan_cache=scan_cache,
                per_tf=per_tf,
                target_window=target_window,
                seq_before=seq_before,
            )
        if move_kind == "M":
            return self._run_multi_move(
                seq=seq,
                score_old=score_old,
                current_combined=current_combined,
                β=β,
                beta_softmin=beta_softmin,
                evaluator=evaluator,
                rng=rng,
                state=state,
                scan_cache=scan_cache,
                per_tf=per_tf,
                target_window=target_window,
                seq_before=seq_before,
            )
        if move_kind == "L":
            return self._run_slide_move(
                seq=seq,
                score_old=score_old,
                current_combined=current_combined,
                β=β,
                beta_softmin=beta_softmin,
                evaluator=evaluator,
                rng=rng,
                state=state,
                scan_cache=scan_cache,
                per_tf=per_tf,
                target_window=target_window,
                seq_before=seq_before,
            )
        if move_kind == "W":
            return self._run_swap_move(
                seq=seq,
                score_old=score_old,
                current_combined=current_combined,
                β=β,
                beta_softmin=beta_softmin,
                evaluator=evaluator,
                rng=rng,
                state=state,
                scan_cache=scan_cache,
                per_tf=per_tf,
                target_window=target_window,
                seq_before=seq_before,
            )
        return self._run_insertion_move(
            seq=seq,
            score_old=score_old,
            current_combined=current_combined,
            β=β,
            beta_softmin=beta_softmin,
            evaluator=evaluator,
            rng=rng,
            state=state,
            scan_cache=scan_cache,
            per_tf=per_tf,
            target_tf=target_tf,
            target_window=target_window,
            seq_before=seq_before,
        )

    def _run_single_base_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        sweep_idx: int,
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
        i = rng.integers(target_window[0], target_window[1]) if target_window is not None else rng.integers(L)
        old = int(seq[i])
        lods = np.empty(4, float)
        combined_vals = np.empty(4, float)
        raw_candidates: list[Dict[str, float]] | None = None
        per_tf_candidates: list[Dict[str, float]] = []
        if scan_cache is not None:
            raw_candidates = scan_cache.candidate_raw_llr_maps(i, old)
            for b in range(4):
                comb_b = evaluator.combined_from_raw_llr(raw_candidates[b], beta=beta_softmin, length=L)
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
        probs = probs / probs.sum()
        p_stay = gibbs_stay_probability(
            sweep_idx=sweep_idx,
            enabled=self.gibbs_inertia_enabled,
            inertia_kind=self.gibbs_inertia_kind,
            total_sweeps=self.total_sweeps,
            p_stay_start=self.gibbs_inertia_p_stay_start,
            p_stay_end=self.gibbs_inertia_p_stay_end,
        )
        if p_stay > 0:
            one_hot = np.zeros(4, dtype=float)
            one_hot[old] = 1.0
            probs = (1.0 - p_stay) * probs + p_stay * one_hot
            probs = probs / probs.sum()
        new_base = int(rng.choice(4, p=probs))
        seq[i] = new_base
        if scan_cache is not None:
            scan_cache.apply_base_change(i, old, new_base)
            assert raw_candidates is not None
            selected_per_tf = evaluator.scorer.scaled_from_raw_llr(raw_candidates[new_base], L)
        else:
            selected_per_tf = per_tf_candidates[new_base]
        self.accept_tally["S"] += 1
        score_new = float(combined_vals[new_base])
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=score_new,
            accepted=True,
            gibbs_changed=(new_base != old),
        )
        return "S", True, selected_per_tf, score_new, detail

    def _run_block_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        per_tf: Dict[str, float],
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
        mn, mx = self.move_cfg["block_len_range"]
        length = rng.integers(mn, mx + 1)
        if length > L:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "B", False, per_tf, current_combined, detail
        start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
        proposal = rng.integers(0, 4, size=length)
        old_block = seq[start : start + length].copy()
        _replace_block(seq, start, length, proposal)
        new_per_tf, new_score = evaluator.evaluate(state, beta=beta_softmin, length=L)
        _replace_block(seq, start, length, old_block)
        if accept_metropolis(beta=β, new_score=float(new_score), current_combined=current_combined, rng=rng):
            _replace_block(seq, start, length, proposal)
            self.accept_tally["B"] += 1
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(new_score),
                accepted=True,
            )
            return "B", True, new_per_tf, new_score, detail
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=float(new_score),
            accepted=False,
        )
        return "B", False, per_tf, current_combined, detail

    def _run_multi_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        per_tf: Dict[str, float],
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
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
        new_per_tf, new_score = evaluator.evaluate(state, beta=beta_softmin, length=L)
        seq[idxs] = old_bases
        if accept_metropolis(beta=β, new_score=float(new_score), current_combined=current_combined, rng=rng):
            seq[idxs] = proposal
            self.accept_tally["M"] += 1
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(new_score),
                accepted=True,
            )
            return "M", True, new_per_tf, new_score, detail
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=float(new_score),
            accepted=False,
        )
        return "M", False, per_tf, current_combined, detail

    def _run_slide_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        per_tf: Dict[str, float],
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
        min_len, max_len = self.move_cfg["swap_len_range"]
        length = rng.integers(min_len, max_len + 1)
        max_shift = int(self.move_cfg["slide_max_shift"])
        if max_shift < 1 or length >= L:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "L", False, per_tf, current_combined, detail
        shift = rng.integers(-max_shift, max_shift + 1)
        if shift == 0:
            shift = 1 if rng.random() < 0.5 else -1
        if shift > 0:
            min_start, max_start = 0, L - length - shift
        else:
            min_start, max_start = -shift, L - length
        if max_start < min_start:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "L", False, per_tf, current_combined, detail
        start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
        start = max(min_start, min(max_start, start))
        slide_window(seq, start, length, int(shift))
        new_per_tf, new_score = evaluator.evaluate(state, beta=beta_softmin, length=L)
        slide_window(seq, start + int(shift), length, int(-shift))
        if accept_metropolis(beta=β, new_score=float(new_score), current_combined=current_combined, rng=rng):
            slide_window(seq, start, length, int(shift))
            self.accept_tally["L"] += 1
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(new_score),
                accepted=True,
            )
            return "L", True, new_per_tf, new_score, detail
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=float(new_score),
            accepted=False,
        )
        return "L", False, per_tf, current_combined, detail

    def _run_swap_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        per_tf: Dict[str, float],
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
        min_len, max_len = self.move_cfg["swap_len_range"]
        length = rng.integers(min_len, max_len + 1)
        if length >= L:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "W", False, per_tf, current_combined, detail
        start_a = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
        max_tries = 10
        start_b = start_a
        for _ in range(max_tries):
            start_b = rng.integers(0, L - length + 1)
            if abs(start_b - start_a) >= length:
                break
        if abs(start_b - start_a) < length:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "W", False, per_tf, current_combined, detail
        swap_block(seq, start_a, start_b, length)
        new_per_tf, new_score = evaluator.evaluate(state, beta=beta_softmin, length=L)
        swap_block(seq, start_a, start_b, length)
        if accept_metropolis(beta=β, new_score=float(new_score), current_combined=current_combined, rng=rng):
            swap_block(seq, start_a, start_b, length)
            self.accept_tally["W"] += 1
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(new_score),
                accepted=True,
            )
            return "W", True, new_per_tf, new_score, detail
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=float(new_score),
            accepted=False,
        )
        return "W", False, per_tf, current_combined, detail

    def _run_insertion_move(
        self,
        *,
        seq: np.ndarray,
        score_old: float,
        current_combined: float,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        state: SequenceState,
        scan_cache: LocalScanCache | None,
        per_tf: Dict[str, float],
        target_tf: str | None,
        target_window: tuple[int, int] | None,
        seq_before: np.ndarray,
    ) -> tuple[str, bool, Dict[str, float], float, Dict[str, object]]:
        L = int(seq.size)
        tf_names = list(self.pwms.keys())
        if not tf_names:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "I", False, per_tf, current_combined, detail
        tf_name = target_tf if target_tf in self.pwms else rng.choice(tf_names)
        pwm = self.pwms[tf_name]
        width = pwm.length
        if width > L:
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(current_combined),
                accepted=False,
            )
            return "I", False, per_tf, current_combined, detail
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
        new_per_tf, new_score = evaluator.evaluate(state, beta=beta_softmin, length=L)
        _replace_block(seq, start, width, old_block)
        if accept_metropolis(beta=β, new_score=float(new_score), current_combined=current_combined, rng=rng):
            _replace_block(seq, start, width, proposal)
            self.accept_tally["I"] += 1
            if scan_cache is not None:
                scan_cache.rebuild(seq)
            detail = move_detail(
                seq=seq,
                seq_before=seq_before,
                score_old=score_old,
                score_new=float(new_score),
                accepted=True,
            )
            return "I", True, new_per_tf, new_score, detail
        detail = move_detail(
            seq=seq,
            seq_before=seq_before,
            score_old=score_old,
            score_new=float(new_score),
            accepted=False,
        )
        return "I", False, per_tf, current_combined, detail

    def stats(self) -> Dict[str, object]:
        return build_stats(self)

    def final_softmin_beta(self) -> float | None:
        return final_softmin_beta(self)

    def final_mcmc_beta(self) -> float | None:
        return final_mcmc_beta(self)

    def objective_schedule_summary(self) -> Dict[str, object]:
        return build_objective_schedule_summary(self)
