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

from dnadesign.permuter.src.core.paths import expand_for_job

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
        # Robust resolution: allow dataset *dir* or explicit records.parquet.
        raw_ds = str(params["from_dataset"])
        job_dir = Path(str(params.get("_job_dir"))) if params.get("_job_dir") else None
        def _coerce_records(p: Path) -> Path:
            if p.is_dir():
                return (p / "records.parquet")
            # if no extension was given and file missing, try dir + records.parquet
            if not p.exists() and p.suffix not in (".parquet", ".pqt"):
                cand = (p / "records.parquet")
                if cand.exists():
                    return cand
            return p
        p_ds = Path(raw_ds).expanduser()
        if job_dir:
            p_ds = expand_for_job(raw_ds, job_dir=job_dir)
        p_ds = _coerce_records(p_ds).resolve()
        if not p_ds.exists():
            raise ValueError(
                "combine_aa: from_dataset not found.\n"
                f"  given: {raw_ds!r}\n"
                f"  resolved: {p_ds}\n"
                "Hint: use job‑relative paths (with ${JOB_DIR}) or pass the dataset directory."
            )
        # codon table (also allow job-relative)
        raw_ct = str(params["codon_table"])
        p_ct = Path(raw_ct).expanduser()
        if job_dir:
            p_ct = expand_for_job(raw_ct, job_dir=job_dir)
        p_ct = p_ct.resolve()
        if not p_ct.exists():
            raise ValueError(
                "combine_aa: codon_table not found.\n"
                f"  given: {raw_ct!r}\n"
                f"  resolved: {p_ct}"
            )

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
        sel = params.get("select", {}) or {}
        mode = str(sel.get("mode", "global")).strip().lower()
        if mode not in {"global", "per_position_best"}:
            raise ValueError("combine_aa: select.mode must be 'global' or 'per_position_best'")
        if "disallow_negative_best" in sel and sel["disallow_negative_best"] not in (True, False):
            raise ValueError("combine_aa: select.disallow_negative_best must be boolean")

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
        # Accept from_dataset as dir or file (already expanded by CLI; still robust here).
        p_ds = Path(str(params["from_dataset"])).expanduser()
        if p_ds.is_dir():
            p_ds = p_ds / "records.parquet"
        dms_path = p_ds.resolve()
        _LOG.info("[combine] singles dataset: %s", dms_path)
        dms = read_parquet(dms_path)

        # Select elite single AA events (Top‑K, ruleouts)
        elite = select_elite_aa_events(dms, metric_col=metric_col, cfg=params)

        # Scoring semantics (LLR): make the contract explicit in stdout
        looks_llr = ("llr" in metric_id.lower()) or ("ratio" in metric_id.lower())
        _LOG.info(
            "[scoring] singles metric_id=%s → expected additive := sum(singles); "
            "observed := evaluator on combined sequence vs REF; synergy := observed − expected",
            metric_id,
        )
        if not looks_llr:
            _LOG.info(
                "[scoring] NOTE: metric_id does not look like an LLR; ensure your evaluator "
                "is a ratio vs reference (e.g., evo2_llr) if you expect zero baseline."
            )

        # RNG: honor user rng_seed (print and use exactly this for sampling).
        user_seed = int(params.get("rng_seed", 42))
        rng = rng or np.random.default_rng(user_seed)
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
            user_seed,
        )

        # Build combinations deterministically
        combos = random_sample(elite, params, np.random.default_rng(user_seed + 17))
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
            # ---- expected (additive) is defined as sum of the selected single‑mutation scores ----
            additive_expected = float(sum(float(sc) for (_, _, _, sc) in events))
            if not np.isfinite(additive_expected):
                raise ValueError("combine_aa: additive expected is non‑finite")
            if not np.isfinite(proposal_score):
                raise ValueError("combine_aa: rank/proposal score is non‑finite")
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
            header = (
                "combo "
                f"k={len(events)} "
                f"additive_{metric_id}={additive_expected:.6f} "
                f"aa=[{aa_combo_str}]"
            )
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
                f"expected__{metric_id}": float(additive_expected),
                "expected_kind": "additive",
            }
            yield out
