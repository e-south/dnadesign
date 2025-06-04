"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/pt.py

Parallel-Tempered Gibbs/Metropolis optimiser.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

# dnadesign/cruncher/sample/optimizer/pt.py

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np

from dnadesign.cruncher.sample.optimizer.base import Optimizer
from dnadesign.cruncher.sample.optimizer.cooling import make_beta_ladder
from dnadesign.cruncher.sample.optimizer.helpers import _replace_block
from dnadesign.cruncher.sample.state import SequenceState, make_seed

logger = logging.getLogger(__name__)


class PTGibbsOptimizer(Optimizer):
    """
    Parallel‐Tempered Gibbs/Metropolis sampler. Identical API to GibbsOptimizer,
    but runs multiple “β‐ladders” & swaps between chains. Stores per‐TF scores
    at each draw, and uses evaluator.combined(...) for acceptance.
    """

    def __init__(self, evaluator: Any, cfg: Dict[str, Any], rng, *, pwms: Dict[str, object], init_cfg: Any):
        """
        evaluator: SequenceEvaluator (returns per‐TF dict + .combined())
        cfg:       flattened config dict
        rng:       numpy.random.Generator
        pwms:      dict of loaded PWMs (names → PWM objects)
        init_cfg:  the init block so that we can seed each chain
        """
        super().__init__(evaluator, cfg, rng)

        self.draws = int(cfg["draws"])
        self.tune = int(cfg["tune"])
        self.chains = int(cfg["chains"])
        self.min_dist = int(cfg["min_dist"])
        self.top_k = int(cfg["top_k"])
        self.swap_prob = float(cfg["swap_prob"])

        # Build β‐ladder from config: list of β values, one per chain
        self.beta_ladder = make_beta_ladder(cfg["beta_ladder"])  # e.g. [0.02,0.05,0.1,...]
        if len(self.beta_ladder) != self.chains:
            raise ValueError("PTGibbs requires chains == len(beta_ladder)")

        # Extract move_cfg (same keys as Gibbs)
        self.move_cfg = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }

        self.pwms = pwms
        self.init_cfg = init_cfg

        self.move_tally: Counter = Counter()

        # Store every sequence-array + (chain, draw) + per‐TF
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_scores: List[Dict[str, float]] = []

        self.elites_meta: List[Tuple[int, int]] = []

        # We'll still build an ArviZ InferenceData of combined fitness per (chain, draw)
        self.trace_idata = None

    def optimise(self, initial: SequenceState) -> List[SequenceState]:
        """
        Parallel‐Tempered MCMC with two phases (burn‐in, sampling). Each chain has its own β.
        Records per‐TF at every draw, and uses evaluator.combined(...) to accept/reject.
        """

        rng = self.rng
        evaluator: Any = self.scorer  # SequenceEvaluator
        tune = self.tune
        draws = self.draws
        chains = self.chains
        min_dist = self.min_dist
        top_k = self.top_k
        beta_ladder = self.beta_ladder
        move_cfg = self.move_cfg

        # Track combined fitness per chain/draw
        chain_scores: List[List[float]] = [[] for _ in range(chains)]

        # Clear old storage
        self.all_samples.clear()
        self.all_meta.clear()
        self.all_scores.clear()

        # Initialize each chain’s state independently
        chain_states: List[np.ndarray] = []
        for c in range(chains):
            seed_state = make_seed(self.init_cfg, self.pwms, rng).seq.copy()
            chain_states.append(seed_state)

        # Phase 1: Burn‐in (no recording)
        total_iters = tune + draws
        for c in range(chains):
            seq = chain_states[c]
            β = beta_ladder[c]
            for _ in range(tune):
                self._single_chain_move(seq, β, evaluator, move_cfg, rng)
                # no recording during burn‐in

        # Phase 2: Sampling + occasional inter‐chain swaps
        for d in range(draws):
            for c in range(chains):
                β = beta_ladder[c]
                seq = chain_states[c]
                self._single_chain_move(seq, β, evaluator, move_cfg, rng)

                # Record this chain’s new sequence, per‐TF, combined
                current_arr = seq.copy()
                current_state = SequenceState(current_arr)
                per_tf_map: Dict[str, float] = evaluator(current_state)
                combined_val = evaluator.combined(current_state)

                self.all_samples.append(current_arr)
                self.all_meta.append((c, d))
                self.all_scores.append(per_tf_map)
                chain_scores[c].append(combined_val)

            # Now attempt swaps between adjacent β‐pairs with probability swap_prob
            for c in range(chains - 1):
                if rng.random() < self.swap_prob:
                    seq_c = chain_states[c]
                    seq_cp1 = chain_states[c + 1]
                    β_c = beta_ladder[c]
                    β_cp1 = beta_ladder[c + 1]

                    # Combined fitness under each chain’s β
                    comb_c = evaluator.combined(SequenceState(seq_c.copy()))
                    comb_cp1 = evaluator.combined(SequenceState(seq_cp1.copy()))

                    # Metropolis‐Hastings swap acceptance
                    Δ = (β_cp1 - β_c) * (comb_c - comb_cp1)
                    if Δ >= 0 or np.log(rng.random()) < Δ:
                        # Swap states
                        chain_states[c], chain_states[c + 1] = seq_cp1, seq_c

        logger.info("Move utilization: %s", dict(self.move_tally))

        # Build ArviZ InferenceData from chain_scores
        scores_arr = np.asarray(chain_scores)  # shape = (C, draws)
        self.trace_idata = az.from_dict(posterior={"score": scores_arr})

        # Rank all sampled sequences by combined fitness (we already stored per‐TF but need combined again)
        scored_list: List[Tuple[float, np.ndarray, int]] = []
        for idx, seq_arr in enumerate(self.all_samples):
            combined_val = evaluator.combined(SequenceState(seq_arr.copy()))
            scored_list.append((combined_val, seq_arr.copy(), idx))
        scored_list.sort(key=lambda x: x[0], reverse=True)

        elites: List[SequenceState] = []
        used_indices: List[int] = []
        for combined_val, seq_arr, idx in scored_list:
            if len(elites) >= top_k:
                break
            if any(np.sum(seq_arr != e.seq) < min_dist for e in elites):
                continue
            elites.append(SequenceState(seq_arr.copy()))
            used_indices.append(idx)

        self.elites_meta = [self.all_meta[i] for i in used_indices]
        return elites

    def _single_chain_move(
        self,
        seq: np.ndarray,
        β: float,
        evaluator: Any,
        move_cfg: Dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """
        Apply exactly one Gibbs-style update (S/B/M) at inverse-temperature β to seq in place.
        Uses evaluator.combined(...) for acceptance.
        """
        L = seq.size
        move_kind = self._sample_move_kind(β, move_cfg, rng)
        self.move_tally[move_kind] += 1

        if move_kind == "S":
            i = rng.integers(L)
            old_base = seq[i]
            lods = np.empty(4, float)
            for b in range(4):
                seq[i] = b
                lods[b] = β * evaluator.combined(SequenceState(seq.copy()))
            seq[i] = old_base
            lods -= lods.max()
            probs = np.exp(lods)
            seq[i] = rng.choice(4, p=probs / probs.sum())

        elif move_kind == "B":
            min_len, max_len = move_cfg["block_len_range"]
            length = rng.integers(min_len, max_len + 1)
            start = rng.integers(0, L - length + 1)
            proposal = rng.integers(0, 4, size=length)

            old_block = seq[start : start + length].copy()
            _replace_block(seq, start, length, proposal)
            new_comb = evaluator.combined(SequenceState(seq.copy()))
            _replace_block(seq, start, length, old_block)
            old_comb = evaluator.combined(SequenceState(seq.copy()))

            Δ = (β * new_comb) - (β * old_comb)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                _replace_block(seq, start, length, proposal)

        else:  # move_kind == "M"
            kmin, kmax = move_cfg["multi_k_range"]
            k = rng.integers(kmin, kmax + 1)
            idxs = rng.choice(L, size=k, replace=False)
            old_bases = seq[idxs].copy()
            proposal = rng.integers(0, 4, size=k)

            seq[idxs] = proposal
            new_comb = evaluator.combined(SequenceState(seq.copy()))
            seq[idxs] = old_bases
            old_comb = evaluator.combined(SequenceState(seq.copy()))

            Δ = (β * new_comb) - (β * old_comb)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                seq[idxs] = proposal

    @staticmethod
    def _sample_move_kind(beta: float, move_cfg: Dict[str, Any], rng: np.random.Generator) -> str:
        """
        Return one of "S","B","M" with β-dependent weights (same as GibbsOptimizer).
        """
        beta_target = 1.0
        r = min(beta / beta_target, 1.0)
        probs = {
            "S": 0.4 + 0.4 * r,
            "B": 0.4 * (1 - r),
            "M": 0.2,
        }
        kinds = list(probs)
        weights = np.fromiter(probs.values(), dtype=float)
        weights /= weights.sum()
        return rng.choice(kinds, p=weights)
