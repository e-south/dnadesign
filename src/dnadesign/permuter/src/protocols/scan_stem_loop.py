"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/protocols/scan_stem_loop.py

Stem-loop (hairpin) generator: (re)build or extend a seeded hairpin inside a
region. GC-aware base sampling, optional mismatches, and rare indels on newly
added paired columns. Deterministic RNG and hairpin-only dedupe per length.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import register
from .base import Protocol

_LOG = logging.getLogger("permuter")

DNA = ("A", "C", "G", "T")
RC = {"A": "T", "T": "A", "C": "G", "G": "C"}


def reverse_complement(s: str) -> str:
    return "".join(RC[b] for b in s[::-1])


def draw_gc_aware(rng: np.random.Generator, gc_target: float) -> str:
    # pG=pC=gc/2, pA=pT=(1-gc)/2
    p_at = (1.0 - gc_target) / 2.0
    p_gc = gc_target / 2.0
    return rng.choice(DNA, p=[p_at, p_gc, p_gc, p_at])


def non_complement_of(b: str, rng: np.random.Generator) -> str:
    comp = RC[b]
    choices = tuple(x for x in DNA if x != comp)
    return rng.choice(choices)


def _hp_hash(subseq: str) -> str:
    digest = hashlib.blake2s(subseq.encode("utf-8"), digest_size=10).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=").upper()[:10]


def _pair_counts(up: str, down: str) -> Tuple[int, int, int]:
    """
    Compute paired_columns, mismatch_count, upstream_gc_count.
    Pair from cap outward: upstream end (-1 backward), downstream start (+0 forward).
    Indel extras beyond min lengths are ignored.
    """
    k = min(len(up), len(down))
    mismatches = 0
    gc_count = 0
    for i in range(k):
        u = up[-1 - i]  # walk backward from cap-end
        d = down[i]  # walk forward from cap-start
        if u in ("G", "C"):
            gc_count += 1
        if d != RC[u]:
            mismatches += 1
    return k, mismatches, gc_count


def _longest_match_run(up: str, down: str) -> int:
    k = min(len(up), len(down))
    best = 0
    run = 0
    for i in range(k):
        u = up[-1 - i]
        d = down[i]
        if d == RC[u]:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _attach_pair(up: str, down: str, u: str, d: str, anchor: str) -> Tuple[str, str]:
    if anchor == "cap":
        up = up + u
        down = d + down
    else:  # base
        up = u + up
        down = down + d
    return up, down


def _insert_adjacent(
    up: str, down: str, arm: str, ins: str, anchor: str
) -> Tuple[str, str]:
    if arm == "up":
        if anchor == "cap":
            up = up + ins
        else:  # base
            up = ins + up
    else:  # arm == "down"
        if anchor == "cap":
            down = ins + down
        else:  # base
            down = down + ins
    return up, down


def _normalize_region(region, seq_len: int) -> Tuple[int, int, bool]:
    """
    Returns (start, end, is_insertion). Insertion i is (i,i,True).
    """
    if isinstance(region, int):
        i = int(region)
        if not (0 <= i <= seq_len):
            raise ValueError(
                f"region out of bounds for sequence length {seq_len}: {region}"
            )
        return i, i, True
    if isinstance(region, (list, tuple)) and len(region) == 2:
        s, e = int(region[0]), int(region[1])
        if not (0 <= s <= e <= seq_len):
            raise ValueError(
                f"region out of bounds for sequence length {seq_len}: {region}"
            )
        return s, e, False
    raise ValueError("region must be an integer or [start, end]")


@register
class ScanStemLoop(Protocol):
    id = "scan_stem_loop"
    version = "1.0"

    def validate_cfg(self, *, params: Dict) -> None:
        seed = params.get("seed") or {}
        cap = seed.get("cap")
        if not isinstance(cap, str) or not re.fullmatch(r"[ACGT]+", cap or ""):
            raise ValueError("seed.cap is required and must be A/C/G/T uppercase")
        if len(cap) < 3:
            raise ValueError("seed.cap length must be ≥ 3")

        # Seed fields format
        for fld in ("upstream_stem", "downstream_stem"):
            val = seed.get(fld)
            if val is None:
                continue
            if not isinstance(val, str) or (val and not re.fullmatch(r"[ACGT]+", val)):
                raise ValueError(f"seed fields must be A/C/G/T uppercase: {fld}")

        program = params.get("program") or {}
        mode = program.get("mode")
        if mode not in {"extend", "rebuild"}:
            raise ValueError("program.mode must be 'extend' or 'rebuild'")
        stem_len = program.get("stem_len") or {}
        try:
            start = int(stem_len.get("start"))
            stop = int(stem_len.get("stop"))
            step = int(stem_len.get("step"))
        except Exception:
            raise ValueError(
                "program.stem_len must have positive step and start ≤ stop"
            )
        if not (step > 0 and start <= stop):
            raise ValueError(
                "program.stem_len must have positive step and start ≤ stop"
            )

        anchor = program.get("anchor")
        if anchor not in {"cap", "base"}:
            raise ValueError("program.anchor must be 'cap' or 'base'")

        spl = int(program.get("samples_per_length", 0))
        if spl <= 0:
            raise ValueError("program.samples_per_length must be a positive integer")

        for key in ("gc_target",):
            g = program.get(key, 0.5)
            if not (0.0 <= float(g) <= 1.0):
                raise ValueError("gc_target/mismatch.rate/indel.rate must be in [0,1]")

        mm = (program.get("mismatch") or {}).get("rate", 0.0)
        ind = (program.get("indel") or {}).get("rate", 0.0)
        if not (0.0 <= float(mm) <= 1.0 and 0.0 <= float(ind) <= 1.0):
            raise ValueError("gc_target/mismatch.rate/indel.rate must be in [0,1]")

        dedupe = params.get("dedupe") or {}
        retry = int(dedupe.get("retry_per_length", 1))
        if retry < 1:
            raise ValueError("dedupe.retry_per_length must be ≥ 1")

        # region type-shape only; bounds validated in generate()
        region = params.get("region")
        if not (
            isinstance(region, int)
            or (isinstance(region, (list, tuple)) and len(region) == 2)
        ):
            raise ValueError("region must be an integer or [start, end]")

    def generate(
        self, *, ref_entry: Dict, params: Dict, rng: np.random.Generator
    ) -> Iterable[Dict]:
        original = str(ref_entry["sequence"]).upper()
        name = ref_entry.get("ref_name", "<unknown>")
        if not re.fullmatch(r"[ACGT]+", original or ""):
            raise ValueError(f"[{name}] invalid characters in sequence")

        # region placement
        s, e, is_insert = _normalize_region(params.get("region"), len(original))

        # seed normalization
        seed = params.get("seed") or {}
        up_init = (seed.get("upstream_stem") or "").upper()
        cap = seed["cap"].upper()
        down_init = seed.get("downstream_stem")
        if down_init is None:
            down_init = reverse_complement(up_init)
        down_init = (down_init or "").upper()

        # schedule & knobs
        program = params.get("program") or {}
        mode = str(program["mode"])
        stem_len = program["stem_len"]
        L_start, L_stop, L_step = (
            int(stem_len["start"]),
            int(stem_len["stop"]),
            int(stem_len["step"]),
        )
        anchor = str(program["anchor"])
        samples_per_length = int(program["samples_per_length"])
        gc_target = float(program.get("gc_target", 0.5))
        mismatch_rate = float((program.get("mismatch") or {}).get("rate", 0.0))
        indel_rate = float((program.get("indel") or {}).get("rate", 0.0))

        # dedupe
        retry_per_length = int((params.get("dedupe") or {}).get("retry_per_length", 32))

        # seed paired length
        paired0 = min(len(up_init), len(down_init))
        if mode == "extend" and L_start < paired0:
            raise ValueError(
                "program.stem_len.start must be ≥ seed paired length (extend mode)"
            )

        # derive inclusive schedule
        L_list: List[int] = list(range(L_start, L_stop + 1, L_step))

        # base seed for inner derivation
        base_seed = params.get("_derived_seed")
        # fallback: derive from rng if not provided
        if base_seed is None:
            base_seed = int(rng.integers(0, 2**63 - 1))

        for L in L_list:
            quota = samples_per_length
            found = 0
            attempts_total = 0
            seen_hashes: set[str] = set()

            for r in range(samples_per_length):
                attempts = 0
                while attempts < retry_per_length:
                    attempts += 1
                    attempts_total += 1
                    # per-attempt deterministic RNG
                    seed_bytes = hashlib.blake2b(
                        f"{base_seed}|{L}|{r}|{attempts}".encode("utf-8"), digest_size=8
                    ).digest()
                    inner_seed = int.from_bytes(seed_bytes, "big", signed=False)
                    rgen = np.random.default_rng(inner_seed)

                    up = up_init
                    down = down_init

                    # compute how many new columns to add
                    if L < paired0:
                        # In rebuild mode, allow shrinking paired length to L by trimming near the cap
                        if mode == "rebuild":
                            # trim back to L (remove from cap side)
                            if anchor == "cap":
                                # trimming near cap is symmetric with cap-anchored zipping
                                trim = paired0 - L
                                up = up[: max(0, len(up) - trim)]
                                down = down[trim:]
                            else:
                                # base anchor: still trim at cap-facing ends
                                trim = paired0 - L
                                up = up[trim:]
                                down = down[: max(0, len(down) - trim)]
                        need = 0
                    else:
                        need = L - min(len(up), len(down))

                    # grow to reach paired length L
                    for _ in range(need):
                        u = draw_gc_aware(rgen, gc_target)
                        d = RC[u]
                        if rgen.random() < mismatch_rate:
                            d = non_complement_of(u, rgen)

                        up, down = _attach_pair(up, down, u, d, anchor)

                        if rgen.random() < indel_rate:
                            arm = rgen.choice(["up", "down"])
                            ins = draw_gc_aware(rgen, gc_target)
                            up, down = _insert_adjacent(up, down, arm, ins, anchor)

                    # compute metrics over paired columns only
                    paired_columns, mismatch_count, gc_count = _pair_counts(up, down)
                    gc_frac = gc_count / max(1, paired_columns)
                    mis_frac = mismatch_count / max(1, paired_columns)
                    longest_run = _longest_match_run(up, down)

                    hp_subseq = up + cap + down
                    hp_id = _hp_hash(hp_subseq)

                    if hp_id in seen_hashes:
                        continue  # duplicate; retry
                    seen_hashes.add(hp_id)

                    # assemble into full sequence
                    if is_insert:
                        full = original[:s] + hp_subseq + original[s:]
                    else:
                        full = original[:s] + hp_subseq + original[e:]

                    variant = {
                        "sequence": full,
                        "modifications": [
                            (
                                "hp "
                                f"L={paired_columns} anchor={anchor} mode={mode} "
                                f"region={s}:{e} cap={len(cap)} "
                                f"asym={abs(len(up)-len(down))} "
                                f"mis={mis_frac:.3f} gc={gc_frac:.3f} "
                                f"rep={r} hash={hp_id}"
                            )
                        ],
                        # hp_* namespace
                        "hp_region_start": s,
                        "hp_region_end": e,
                        "hp_mode": mode,
                        "hp_anchor": anchor,
                        "hp_length_paired": paired_columns,
                        "hp_cap_len": len(cap),
                        "hp_up_len": len(up),
                        "hp_down_len": len(down),
                        "hp_asymmetry": abs(len(up) - len(down)),
                        "hp_mismatch_frac": mis_frac,
                        "hp_gc_frac_paired": gc_frac,
                        "hp_longest_match_run": int(longest_run),
                        "hp_upstream": up,
                        "hp_cap": cap,
                        "hp_downstream": down,
                        "hp_subseq": hp_subseq,
                        "hp_hash": hp_id,
                        # gen_* namespace
                        "gen_replicate_idx": r,
                        "gen_gc_target": gc_target,
                        "gen_mismatch_rate": mismatch_rate,
                        "gen_indel_rate": indel_rate,
                    }

                    yield variant
                    found += 1
                    break  # next replicate

            if found < quota:
                _LOG.warning(
                    f"WARNING: length {L}: produced {found}/{quota} unique hairpins "
                    f"after {attempts_total} attempts."
                )
