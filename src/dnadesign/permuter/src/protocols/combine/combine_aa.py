"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/combine/combine_aa.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from dnadesign.permuter.src.core.storage import read_parquet
from dnadesign.permuter.src.protocols.base import Protocol, assert_dna
from dnadesign.permuter.src.protocols.combine.builders import random_sample
from dnadesign.permuter.src.protocols.combine.codon_utils import (
    aa_to_best_codon,
    aa_to_weighted_codon,
    load_codon_table,
)
from dnadesign.permuter.src.protocols.combine.selection import (
    select_elite_aa_events,
)

_LOG = logging.getLogger("permuter.protocol.combine_aa")

_TRIPLE = 3


def _preserve_case_apply(
    orig: str, start_0b: int, new_codon_up: str
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Replace 3 nt at [start_0b:start_0b+3) with new_codon_up (uppercase), preserving
    the original per-base case. Returns (new_sequence, nt_change_tokens_info)
    where nt_change_tokens_info is a list of (pos1b, wt, alt) for positions that changed.
    """
    chars = list(orig)
    old = "".join(orig[start_0b + i].upper() for i in range(3))
    nt_tokens: List[Tuple[int, str, str]] = []
    for off in range(3):
        i = start_0b + off
        old_up = old[off]
        new_up = new_codon_up[off]
        if old_up != new_up:
            pos1b = i + 1
            nt_tokens.append((pos1b, old_up, new_up))
        chars[i] = new_up if chars[i].isupper() else new_up.lower()
    return "".join(chars), nt_tokens


class CombineAA(Protocol):
    """
    Build multi‑AA combinations from a prior single‑mutation DMS dataset.
    Emits DNA sequences; encodes AA edits via codon swaps chosen by policy ('top'|'weighted').
    """

    # ------------------------------ Validation ------------------------------ #
    def validate_cfg(self, *, params: Dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("combine_aa: params must be a mapping")

        for req in ("from_dataset", "codon_table", "singles_metric_id"):
            if not params.get(req):
                raise ValueError(f"combine_aa: params.{req} is required")
        p_ds = Path(str(params["from_dataset"])).expanduser().resolve()
        if not p_ds.exists():
            raise ValueError(f"combine_aa: from_dataset not found at {p_ds}")
        p_ct = Path(str(params["codon_table"])).expanduser().resolve()
        if not p_ct.exists():
            raise ValueError(f"combine_aa: codon_table not found at {p_ct}")

        # Combination builder (v0.1: random only)
        comb = params.get("combine") or {}
        if comb.get("strategy", "random") != "random":
            raise ValueError(
                "combine_aa: only strategy='random' is implemented in v0.1"
            )
        try:
            k_min = int(comb.get("k_min", 0))
            k_max = int(comb.get("k_max", 0))
            budget = int(comb.get("budget_total", 0))
        except Exception:
            raise ValueError(
                "combine_aa: combine.k_min/k_max/budget_total must be integers"
            )
        if k_min < 1 or k_max < k_min:
            raise ValueError("combine_aa: k_min must be ≥1 and k_max ≥ k_min")
        if budget <= 0:
            raise ValueError("combine_aa: budget_total must be > 0")

        choice = params.get("codon_choice", "top")
        if choice not in {"top", "weighted"}:
            raise ValueError("combine_aa: codon_choice must be 'top' or 'weighted'")

        # Selection block sanity (optional keys validated in selection)
        _ = params.get("select", {})  # may be empty

        # rng_seed is optional; verify int if present
        if "rng_seed" in params:
            try:
                _ = int(params["rng_seed"])
            except Exception:
                raise ValueError("combine_aa: rng_seed must be an integer")

    # ------------------------------ Generation ------------------------------ #
    def generate(
        self,
        *,
        ref_entry: Dict,
        params: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Dict]:
        # Reference DNA (coding); preserve original case in final output
        orig = str(ref_entry["sequence"])
        assert_dna(orig)
        if len(orig) % _TRIPLE != 0:
            raise ValueError("combine_aa: reference DNA length must be divisible by 3")
        seq_upper = orig.upper()
        n_codons = len(seq_upper) // 3

        # Load singles dataset and validate schema
        metric_id = str(params["singles_metric_id"]).strip()
        metric_col = f"permuter__metric__{metric_id}"
        dms = read_parquet(Path(str(params["from_dataset"])).expanduser().resolve())

        # Select elite single AA events (Top‑K, ruleouts)
        elite = select_elite_aa_events(dms, metric_col=metric_col, cfg=params)

        if elite:
            head = elite[: min(24, len(elite))]
            top_tokens = [
                f"{wt}{pos}{alt}:{score:+.3f}" for (pos, wt, alt, score) in head
            ]
            _LOG.info(
                "[elite] top_global=%d  head=%s", len(elite), "  ".join(top_tokens)
            )

        # RNG: derive a reproducible base seed from outer seed + user seed.
        user_seed = int(params.get("rng_seed", 1234))
        derived = str(params.get("_derived_seed", user_seed))
        base_seed = int(
            hashlib.blake2b(
                f"{derived}-{user_seed}".encode("utf-8"), digest_size=8
            ).hexdigest(),
            16,
        )
        rng = rng or np.random.default_rng(base_seed)
        # Load codon usage table (strict)
        tbl = load_codon_table(params["codon_table"])

        # Informative one‑line plan
        comb = params.get("combine") or {}
        k_min = int(comb.get("k_min"))
        k_max = int(comb.get("k_max"))
        budget_total = int(comb.get("budget_total"))
        choice = params.get("codon_choice", "top")
        _LOG.info(
            "[plan] ref=%s • elite=%d • k=%d..%d • strategy=%s • budget=%d • codon=%s • seed=%d",
            ref_entry.get("ref_name", "<ref>"),
            len(elite),
            k_min,
            k_max,
            str(comb.get("strategy", "random")),
            budget_total,
            choice,
            base_seed,
        )

        # Build combinations deterministically
        combos = random_sample(elite, params, np.random.default_rng(base_seed + 17))
        # Digest
        per_k: Dict[int, int] = {}
        for evs, _ in combos:
            per_k[len(evs)] = per_k.get(len(evs), 0) + 1
        if per_k:
            k_counts = ", ".join(f"k={k}:{per_k[k]}" for k in sorted(per_k))
            _LOG.info(
                "[digest] emitted %d combination proposals (%s)", len(combos), k_counts
            )
        else:
            _LOG.info("[digest] emitted 0 combination proposals")

        # Helper: AA from reference codon
        def wt_aa_at(pos1b: int) -> str:
            idx0 = (int(pos1b) - 1) * 3
            codon = seq_upper[idx0 : idx0 + 3]
            aa = tbl.codon2aa.get(codon)
            if aa is None:
                raise ValueError(
                    f"combine_aa: reference codon {codon!r} at AA pos {pos1b} is not in codon table"
                )
            return aa

        # Emit variants
        for events, proposal_score in combos:
            # Validate and prepare per-event edits
            aa_tokens: List[str] = []
            nt_token_accum: List[Tuple[int, str, str]] = []
            positions_sorted = sorted(e[0] for e in events)
            if any(p < 1 or p > n_codons for p in positions_sorted):
                raise ValueError(
                    f"combine_aa: event position out of bounds (1..{n_codons}): {positions_sorted}"
                )

            # Enforce WT match vs reference and build AA tokens
            for pos, wt, alt, sc in events:
                wt_ref = wt_aa_at(pos)
                if wt_ref != str(wt).upper():
                    raise ValueError(
                        f"combine_aa: WT mismatch at pos {pos}: dataset WT={wt}, ref codon encodes {wt_ref}"
                    )
                aa_tokens.append(
                    f"aa pos={int(pos)} wt={wt_ref} alt={str(alt).upper()}"
                )

            # Mutate DNA codons per policy (preserve per-base case)
            new_seq = orig
            # Apply edits in ascending position order to avoid index confusion
            for pos, wt, alt, sc in sorted(events, key=lambda x: x[0]):
                start0 = (int(pos) - 1) * 3
                if choice == "top":
                    new_codon_up = aa_to_best_codon(tbl, str(alt))
                else:
                    new_codon_up = aa_to_weighted_codon(tbl, str(alt), rng)
                new_seq, nt_changes = _preserve_case_apply(
                    new_seq, start0, new_codon_up
                )
                nt_token_accum.extend(nt_changes)

            # Canonical combo key + lists
            aa_pos_list = [int(p) for p in positions_sorted]
            aa_wt_list = [wt for (_, wt, _, _) in sorted(events, key=lambda x: x[0])]
            aa_alt_list = [alt for (_, _, alt, _) in sorted(events, key=lambda x: x[0])]
            aa_combo_str = "|".join(
                f"{w}{p}{a}" for p, w, a in zip(aa_pos_list, aa_wt_list, aa_alt_list)
            )

            # Header token + AA+NT tokens
            header = f"combo k={len(events)} singles_score={float(proposal_score):.6f} aa=[{aa_combo_str}]"
            nt_tokens_txt = [
                f"nt pos={pos1b} wt={wt} alt={alt}"
                for (pos1b, wt, alt) in nt_token_accum
            ]
            modifications = [header, *aa_tokens, *nt_tokens_txt]

            out = {
                # USR core fields are set by the caller; we emit the sequence and protocol specifics only.
                "sequence": new_seq,
                "modifications": modifications,
                # round '2' (combination generation)
                "round": 2,
                # AA lists
                "aa_pos_list": aa_pos_list,
                "aa_wt_list": aa_wt_list,
                "aa_alt_list": aa_alt_list,
                "aa_combo_str": aa_combo_str,
                "mut_count": len(events),
                "proposal_score": float(proposal_score),
                # expected additive (used by synergy plot)
                f"expected__{metric_id}": float(proposal_score),
            }
            yield out
