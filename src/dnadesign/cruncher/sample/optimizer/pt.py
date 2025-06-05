"""
<dnadesign project>
dnadesign/cruncher/sample/optimizer/pt.py

Parallel-Tempered Gibbs/Metropolis sampler.

*   Each chain is run at a fixed inverse-temperature β taken from the
    geometric ladder supplied in the cooling block of the YAML config.
*   Proposal kernel - S / B / M - and acceptance rules are identical to the
    single-chain Gibbs implementation, except that log-accept ratios are
    multiplied by β.
*   After every recorded sweep we perform pair-wise nearest-neighbour swaps in
    the β-ladder with probability *swap_prob*.
*   The optimiser records:
        • `all_samples`  - np.ndarray of each accepted sequence (numeric)
        • `all_meta`     - tuples (chain_id, draw_index)
        • `all_scores`   - per-TF scaled score dicts
        • `trace_idata`  - ArviZ InferenceData of combined fitness
        • `elites_meta`  - (chain, draw) pointers for the top-k diverse elites

The public API mirrors GibbsOptimizer: call `.optimise()` with no
arguments and receive a list of elite SequenceState objects.

Module Author(s): Eric J. South
Dunlop Lab
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np
from tqdm import tqdm

from dnadesign.cruncher.sample.optimizer.base import Optimizer
from dnadesign.cruncher.sample.optimizer.cooling import make_beta_ladder
from dnadesign.cruncher.sample.optimizer.helpers import _replace_block
from dnadesign.cruncher.sample.state import SequenceState, make_seed

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
        init_cfg: Any,
    ) -> None:
        super().__init__(evaluator, cfg, rng)

        # Core dimensions
        self.draws: int = int(cfg["draws"])
        self.tune: int = int(cfg["tune"])
        self.chains: int = int(cfg["chains"])
        self.min_dist: int = int(cfg["min_dist"])
        self.top_k: int = int(cfg["top_k"])
        self.swap_prob: float = float(cfg["swap_prob"])

        # β‑ladder
        cooling_cfg = {"kind": cfg["kind"], "beta": cfg["beta"]}
        self.beta_ladder: List[float] = make_beta_ladder(cooling_cfg)
        if len(self.beta_ladder) != self.chains:
            raise ValueError(
                "Length of beta ladder (%d) must match number of chains (%d)" % (len(self.beta_ladder), self.chains)
            )

        # Move configuration
        self.move_cfg: Dict[str, Tuple[int, int] | int] = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }
        self.move_probs = np.array(
            [cfg["move_probs"]["S"], cfg["move_probs"]["B"], cfg["move_probs"]["M"]],
            dtype=float,
        )

        # References needed during optimisation
        self.pwms = pwms
        self.init_cfg = init_cfg

        # Book‑keeping
        self.move_tally: Counter = Counter()
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_scores: List[Dict[str, float]] = []
        self.elites_meta: List[Tuple[int, int]] = []
        self.trace_idata = None  # filled after optimise()

    # Public API
    def optimise(self) -> List[SequenceState]:  # noqa: C901  (long but readable)
        """Run PT-MCMC and return *k* diverse elite sequences."""

        rng = self.rng
        evaluator = self.scorer  # SequenceEvaluator
        C, T, D = self.chains, self.tune, self.draws
        βs = self.beta_ladder
        logger.info("Starting PT optimisation: chains=%d  tune=%d  draws=%d", C, T, D)

        # Seed each chain independently
        chain_states: List[np.ndarray] = [make_seed(self.init_cfg, self.pwms, rng).seq.copy() for _ in range(C)]

        # For trace: only *draw* phase (not tune) is stored per ArviZ convention.
        chain_scores: List[List[float]] = [[] for _ in range(C)]

        # Helper to record a state
        def _record(chain_id: int, draw_idx: int, seq_arr: np.ndarray):
            state = SequenceState(seq_arr.copy())
            per_tf = evaluator(state)
            self.all_samples.append(seq_arr.copy())
            self.all_scores.append(per_tf)
            self.all_meta.append((chain_id, draw_idx))

        # Burn‑in sweeps (record like Gibbs for consistency)
        for t in tqdm(range(T), desc="burn‑in", leave=False):
            for c in range(C):
                self._single_chain_move(chain_states[c], βs[c], evaluator, rng)
                _record(c, t, chain_states[c])

        # Sampling sweeps + swap attempts
        for d in tqdm(range(D), desc="sampling", leave=False):
            # Within‑chain proposals
            for c in range(C):
                self._single_chain_move(chain_states[c], βs[c], evaluator, rng)
                _record(c, T + d, chain_states[c])

                # store combined fitness only for draw phase
                comb = evaluator.combined(SequenceState(chain_states[c].copy()))
                chain_scores[c].append(comb)

            # Pair‑wise swaps
            for c in range(C - 1):
                if rng.random() >= self.swap_prob:
                    continue
                s0, s1 = chain_states[c], chain_states[c + 1]
                β0, β1 = βs[c], βs[c + 1]
                f0 = evaluator.combined(SequenceState(s0.copy()))
                f1 = evaluator.combined(SequenceState(s1.copy()))
                Δ = (β1 - β0) * (f0 - f1)
                if Δ >= 0 or np.log(rng.random()) < Δ:
                    chain_states[c], chain_states[c + 1] = s1, s0  # swap in‑place

        logger.info("PT optimisation finished. Move utilisation: %s", dict(self.move_tally))

        # Build ArviZ trace from draw phase only
        score_arr = np.asarray(chain_scores)  # (C, D)
        self.trace_idata = az.from_dict(posterior={"score": score_arr})

        # Rank all recorded sequences by combined fitness
        ranked: List[Tuple[float, np.ndarray, int]] = []
        for idx, seq in enumerate(self.all_samples):
            val = evaluator.combined(SequenceState(seq.copy()))
            ranked.append((val, seq.copy(), idx))
        ranked.sort(key=lambda x: x[0], reverse=True)

        elites: List[SequenceState] = []
        picked_idx: List[int] = []
        for val, seq, idx in ranked:
            if len(elites) >= self.top_k:
                break
            if any(np.sum(seq != e.seq) < self.min_dist for e in elites):
                continue
            elites.append(SequenceState(seq))
            picked_idx.append(idx)
        self.elites_meta = [self.all_meta[i] for i in picked_idx]
        return elites

    # Low‑level helpers
    def _single_chain_move(
        self,
        seq: np.ndarray,
        β: float,
        evaluator: Any,
        rng: np.random.Generator,
    ) -> None:
        """One Gibbs-style proposal/accept cycle at inverse-temperature β."""

        L = seq.size
        move_kind = self._sample_move_kind(rng)
        self.move_tally[move_kind] += 1

        # Single‑base flip
        if move_kind == "S":
            i = rng.integers(L)
            old = seq[i]
            lods = np.empty(4, float)
            for b in range(4):
                seq[i] = b
                lods[b] = β * evaluator.combined(SequenceState(seq.copy()))
            seq[i] = old
            lods -= lods.max()
            probs = np.exp(lods)
            seq[i] = rng.choice(4, p=probs / probs.sum())
            return

        # Contiguous block replace
        if move_kind == "B":
            mn, mx = self.move_cfg["block_len_range"]
            length = rng.integers(mn, mx + 1)
            start = rng.integers(0, L - length + 1)
            proposal = rng.integers(0, 4, size=length)
            old_block = seq[start : start + length].copy()

            _replace_block(seq, start, length, proposal)
            new_f = evaluator.combined(SequenceState(seq.copy()))
            _replace_block(seq, start, length, old_block)
            old_f = evaluator.combined(SequenceState(seq.copy()))

            Δ = β * (new_f - old_f)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                _replace_block(seq, start, length, proposal)
            return

        # Multi‑site flip
        kmin, kmax = self.move_cfg["multi_k_range"]
        k = rng.integers(kmin, kmax + 1)
        idxs = rng.choice(L, size=k, replace=False)
        old_bases = seq[idxs].copy()
        proposal = rng.integers(0, 4, size=k)

        seq[idxs] = proposal
        new_f = evaluator.combined(SequenceState(seq.copy()))
        seq[idxs] = old_bases
        old_f = evaluator.combined(SequenceState(seq.copy()))

        Δ = β * (new_f - old_f)
        if Δ >= 0 or np.log(rng.random()) < Δ:
            seq[idxs] = proposal

    def _sample_move_kind(self, rng: np.random.Generator) -> str:
        kinds = ["S", "B", "M"]
        return rng.choice(kinds, p=self.move_probs)
