"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/hairpins/scan_stem_loop.py

Stem-loop (hairpin) generator: (re)build or extend a seeded hairpin inside a
region. GC-aware base sampling and optional mismatches (per stratum) on newly
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

from dnadesign.permuter.src.protocols import register
from dnadesign.permuter.src.protocols.base import Protocol

_LOG = logging.getLogger("permuter.protocol.scan_stem_loop")

DNA = ("A", "C", "G", "T")
RC = {"A": "T", "T": "A", "C": "G", "G": "C"}


def reverse_complement(s: str) -> str:
    return "".join(RC[b] for b in s[::-1])


def draw_gc_aware(rng: np.random.Generator, gc_target: float) -> str:
    """Sample one base with P(A)=P(T)=(1-gc)/2 and P(C)=P(G)=gc/2."""
    p_at = (1.0 - gc_target) / 2.0
    p_gc = gc_target / 2.0
    return rng.choice(DNA, p=[p_at, p_gc, p_gc, p_at])


def non_complement_of(b: str, rng: np.random.Generator) -> str:
    """Return a base that is not the Watson-Crick complement of b."""
    comp = RC[b]
    choices = tuple(x for x in DNA if x != comp)
    return rng.choice(choices)


def _hp_hash(subseq: str) -> str:
    """Stable, short hairpin subsequence hash for de-duplication."""
    digest = hashlib.blake2s(subseq.encode("utf-8"), digest_size=10).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=").upper()[:10]


def _pair_counts(up: str, down: str) -> Tuple[int, int, int]:
    """
    Compute (paired_columns, mismatch_count, upstream_gc_count).
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
    """Longest contiguous run of perfectly matched pairs (cap-outward indexing)."""
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


def _attach_pair(up: str, down: str, u: str, d: str, grow_from: str) -> Tuple[str, str]:
    """Attach a new paired column at the cap or base side."""
    if grow_from == "cap":
        up = up + u
        down = d + down
    else:  # "base"
        up = u + up
        down = down + d
    return up, down


def _normalize_region(region, seq_len: int) -> Tuple[int, int, bool]:
    """
    Returns (start, end, is_insertion). Insertion i is (i, i, True).
    Slice semantics use end-exclusive: original[:start] + NEW + original[end:].
    """
    if isinstance(region, int):
        i = int(region)
        if not (0 <= i <= seq_len):
            raise ValueError(f"region out of bounds for sequence length {seq_len}: {region}")
        return i, i, True
    if isinstance(region, (list, tuple)) and len(region) == 2:
        s, e = int(region[0]), int(region[1])
        if not (0 <= s <= e <= seq_len):
            raise ValueError(f"region out of bounds for sequence length {seq_len}: {region}")
        return s, e, False
    raise ValueError("region must be an integer or [start, end]")


def _mismatch_masks(up: str, down: str) -> Tuple[List[bool], List[bool]]:
    """
    Return (mask_up, mask_rc_up) booleans where True marks a mismatch column.
    • mask_up aligns to 'up' in 5'→3' (mismatches mapped onto the last k bases).
    • mask_rc_up aligns to reverse_complement(up) left→right (cap→base side).
    """
    k = min(len(up), len(down))
    mask_cap_outward = [(down[i] != RC[up[-1 - i]]) for i in range(k)]
    mask_up = [False] * len(up)
    for j in range(len(up) - k, len(up)):
        i = (len(up) - 1) - j  # cap-outward index
        mask_up[j] = mask_cap_outward[i]
    mask_rc_up = [False] * len(up)
    for j in range(k):  # rc(up)[j] pairs with up[-1-j]
        mask_rc_up[j] = mask_cap_outward[j]
    return mask_up, mask_rc_up


def _apply_case(s: str, mask: List[bool]) -> str:
    return "".join((ch.upper() if mask[i] else ch.lower()) for i, ch in enumerate(s))


def _preview_line(original: str, s: int, e: int, up: str, cap: str, down: str) -> str:
    """
    Pretty, compact one-liner per hairpin:
      match: …<left5> | <up (mismatches UPPER)> | <cap> | <revcomp(up) (mismatches UPPER)> | <right5>…
    """
    left5 = original[max(0, s - 5) : s]
    right5 = original[e : e + 5] if e < len(original) else ""
    rc_up = reverse_complement(up)
    m_up, m_rc = _mismatch_masks(up, down)
    up_vis = _apply_case(up, m_up)
    rc_vis = _apply_case(rc_up, m_rc)
    return f"match: …{left5} | {up_vis} | {cap.lower()} | {rc_vis} | {right5}…"


@register
class ScanStemLoop(Protocol):
    id = "scan_stem_loop"
    version = "1.0"

    # ------------------------------ Validation ------------------------------ #
    def validate_cfg(self, *, params: Dict) -> None:
        seed = params.get("seed") or {}
        cap = seed.get("cap")
        if not isinstance(cap, str) or not re.fullmatch(r"[ACGTacgt]+", cap or ""):
            raise ValueError("seed.cap is required and must be DNA (A/C/G/T), case-insensitive")
        if len(cap) < 3:
            raise ValueError("seed.cap length must be ≥ 3")

        # Validate optional seed arms
        for fld in ("upstream_stem", "downstream_stem"):
            val = seed.get(fld)
            if val is None:
                continue
            if not isinstance(val, str) or (val and not re.fullmatch(r"[ACGTacgt]+", val)):
                raise ValueError(f"seed.{fld} must be DNA (A/C/G/T), case-insensitive")

        program = params.get("program") or {}
        mode = program.get("mode")
        if mode not in {"extend", "rebuild"}:
            raise ValueError("program.mode must be 'extend' or 'rebuild'")

        stem_len = program.get("stem_len") or {}
        try:
            start = int(stem_len.get("start"))
            stop = int(stem_len.get("stop"))
            step = int(stem_len.get("step"))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("program.stem_len must have positive step and start ≤ stop") from exc
        if not (step > 0 and start <= stop):
            raise ValueError("program.stem_len must have positive step and start ≤ stop")

        # Explicit growth direction (cap vs base). 'anchor' is deprecated.
        if "anchor" in program:
            raise ValueError("program.anchor is deprecated. Use program.grow_from with 'cap' or 'base'.")
        grow_from = program.get("grow_from")
        if grow_from not in {"cap", "base"}:
            raise ValueError("program.grow_from is required and must be 'cap' or 'base'")

        spl = int(program.get("samples_per_length", 0))
        if spl <= 0:
            raise ValueError("program.samples_per_length must be a positive integer")

        # Scalar knobs
        g = program.get("gc_target", 0.5)
        if not (0.0 <= float(g) <= 1.0):
            raise ValueError("program.gc_target must be in [0, 1]")

        # Dedupe/attempts
        dedupe = params.get("dedupe") or {}
        retry = int(dedupe.get("retry_per_length", 1))
        if retry < 1:
            raise ValueError("dedupe.retry_per_length must be ≥ 1")

        # Deterministic RNG
        rng_seed = program.get("rng_seed", 42)
        try:
            _ = int(rng_seed)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("program.rng_seed must be an integer") from exc

        # Optional category tracks
        strata = program.get("strata")
        if strata is not None:
            if not isinstance(strata, list) or len(strata) == 0:
                raise ValueError("program.strata must be a non-empty list when provided")
            for s in strata:
                if not isinstance(s, dict) or "id" not in s:
                    raise ValueError("each program.strata[] item must be an object with an 'id'")
                ovr = s.get("overrides", {}) or {}
                if "gc_target" in ovr and not (0.0 <= float(ovr["gc_target"]) <= 1.0):
                    raise ValueError("program.strata[].overrides.gc_target must be in [0, 1]")
                mm_s = ovr.get("mismatch_rate", None)
                if mm_s is not None and not (0.0 <= float(mm_s) <= 1.0):
                    raise ValueError("program.strata[].overrides.mismatch.rate must be in [0, 1]")
                spl2 = s.get("samples_per_length")
                if spl2 is not None and int(spl2) <= 0:
                    raise ValueError("program.strata[].samples_per_length must be a positive integer")

        # Placement: either a numeric region OR anchors (left/right), not both
        region = params.get("region", None)
        anchors = params.get("anchors", None)
        if anchors is None:
            # region is required when anchors are not provided
            if not (isinstance(region, int) or (isinstance(region, (list, tuple)) and len(region) == 2)):
                raise ValueError("region must be an integer or [start, end]")
        else:
            if region is not None:
                raise ValueError("Specify either 'region' or 'anchors', not both")
            if not isinstance(anchors, dict):
                raise ValueError("anchors must be a mapping with 'left' and 'right'")
            for key in ("left", "right"):
                v = anchors.get(key, "")
                if not isinstance(v, str) or not re.fullmatch(r"[ACGTacgt]+", v or ""):
                    raise ValueError(f"anchors.{key} must be DNA (A/C/G/T)")

    # ------------------------------ Generation ------------------------------ #
    def generate(self, *, ref_entry: Dict, params: Dict, rng: np.random.Generator) -> Iterable[Dict]:
        original = str(ref_entry["sequence"]).upper()
        name = ref_entry.get("ref_name", "<unknown>")
        if not re.fullmatch(r"[ACGT]+", original or ""):
            raise ValueError(f"[{name}] invalid characters in sequence")

        # Region placement: from anchors OR explicit region
        anchors = params.get("anchors", None)
        if anchors is not None:
            left = str(anchors.get("left", "")).upper()
            right = str(anchors.get("right", "")).upper()

            # find unique left and right
            def _all_occurs(hay: str, needle: str) -> list[int]:
                return [m.start() for m in re.finditer(re.escape(needle), hay)]

            left_hits = _all_occurs(original, left)
            right_hits = _all_occurs(original, right)
            if not left_hits:
                raise ValueError(f"[{name}] anchors.left not found in reference")
            if not right_hits:
                raise ValueError(f"[{name}] anchors.right not found in reference")
            if len(left_hits) > 1:
                raise ValueError(f"[{name}] anchors.left is ambiguous (matches={len(left_hits)})")
            # choose the first right that occurs after left
            L0 = left_hits[0]
            R_candidates = [r for r in right_hits if r >= L0 + len(left)]
            if not R_candidates:
                raise ValueError(f"[{name}] anchors.right occurs before/overlaps anchors.left")
            if len(R_candidates) > 1:
                raise ValueError(f"[{name}] anchors.right is ambiguous after left (matches={len(R_candidates)})")
            R0 = R_candidates[0]
            s, e = L0 + len(left), R0
            is_insert = s == e
        else:
            s, e, is_insert = _normalize_region(params.get("region"), len(original))

        # --------- One-time run summary (INFO) ----------
        program = params.get("program") or {}
        mode = str(program["mode"])
        stem_len = program["stem_len"]
        L_start, L_stop, L_step = (
            int(stem_len["start"]),
            int(stem_len["stop"]),
            int(stem_len["step"]),
        )
        grow_from = str(program["grow_from"])
        default_spl = int(program["samples_per_length"])
        default_gc = float(program.get("gc_target", 0.5))
        rng_seed = int(program.get("rng_seed", 42))

        # One succinct “plan” banner (INFO)
        n_lengths = len(list(range(L_start, L_stop + 1, L_step)))
        placement = "anchors" if params.get("anchors") is not None else "region"
        _LOG.info(
            "[plan] ref=%s • %s=%d:%d (insert=%s) • mode=%s grow=%s • L=%d..%d step=%d (N=%d) • samples/L=%d • gc=%.2f • seed=%d",  # noqa
            name,
            placement,
            s,
            e,
            bool(is_insert),
            mode,
            grow_from,
            L_start,
            L_stop,
            L_step,
            n_lengths,
            default_spl,
            default_gc,
            rng_seed,
        )

        # Seed normalization
        seed = params.get("seed") or {}
        up_init = (seed.get("upstream_stem") or "").upper()
        cap = seed["cap"].upper()
        down_init = seed.get("downstream_stem")
        if down_init is None:
            down_init = reverse_complement(up_init)
        down_init = (down_init or "").upper()

        # Category tracks (optional)
        strata = program.get("strata") or [
            {
                "id": "default",
                "label": "default",
                "kind": "single",
                "overrides": {
                    "gc_target": default_gc,
                },
                "samples_per_length": default_spl,
            }
        ]

        # Dedupe/attempt controls
        retry_per_length = int((params.get("dedupe") or {}).get("retry_per_length", 32))

        # Seed paired length and mode guard
        paired0 = min(len(up_init), len(down_init))
        if mode == "extend" and L_start < paired0:
            raise ValueError("program.stem_len.start must be ≥ seed paired length (extend mode)")

        # Inclusive schedule
        L_list: List[int] = list(range(L_start, L_stop + 1, L_step))

        # Reproducible base seed derived from outer RNG and user seed
        base_seed = int(
            hashlib.blake2b(
                f"{params.get('_derived_seed')}-{rng_seed}".encode("utf-8"),
                digest_size=8,
            ).hexdigest(),
            16,
        )

        # Iterate category tracks
        _LOG.info(
            "[scan_stem_loop] tracks (%d): %s",
            len(strata),
            ", ".join(str(t.get("id")) for t in strata),
        )

        for track_order, track in enumerate(strata):
            track_id = str(track.get("id"))
            track_label = str(track.get("label", track_id))
            track_kind = str(track.get("kind", "mismatch_rate"))
            ovr = track.get("overrides", {}) or {}
            gc_target = float(ovr.get("gc_target", default_gc))
            mm_rate = float(ovr.get("mismatch_rate", 0.0))
            samples_per_length = int(track.get("samples_per_length", default_spl))

            _LOG.debug(
                "[track] id=%s label=%s kind=%s gc=%.2f mismatch=%.2f samples/L=%d",
                track_id,
                track_label,
                track_kind,
                gc_target,
                mm_rate,
                samples_per_length,
            )

            total_emitted = 0
            shortfalls: List[str] = []
            previews: List[str] = []
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

                        # Per-attempt deterministic RNG (track-aware)
                        seed_bytes = hashlib.blake2b(
                            f"{base_seed}|{track_id}|{L}|{r}|{attempts}".encode("utf-8"),
                            digest_size=8,
                        ).digest()
                        inner_seed = int.from_bytes(seed_bytes, "big", signed=False)
                        rgen = np.random.default_rng(inner_seed)

                        # Start from seed arms
                        up = up_init
                        down = down_init

                        # Adjust paired length if rebuilding to a shorter target
                        if L < paired0:
                            if mode == "rebuild":
                                trim = paired0 - L
                                if grow_from == "cap":
                                    up = up[: max(0, len(up) - trim)]
                                    down = down[trim:]
                                else:  # base
                                    up = up[trim:]
                                    down = down[: max(0, len(down) - trim)]
                            need = 0
                        else:
                            need = L - min(len(up), len(down))

                        # Grow to reach paired length L
                        for _ in range(need):
                            u = draw_gc_aware(rgen, gc_target)
                            d = RC[u]
                            if rgen.random() < mm_rate:
                                d = non_complement_of(u, rgen)
                            up, down = _attach_pair(up, down, u, d, grow_from)

                        # Compute metrics over paired columns only
                        paired_columns, mismatch_count, gc_count = _pair_counts(up, down)
                        gc_frac = gc_count / max(1, paired_columns)
                        mis_frac = mismatch_count / max(1, paired_columns)
                        longest_run = _longest_match_run(up, down)

                        hp_subseq = up + cap + down
                        hp_id = _hp_hash(hp_subseq)

                        # Hairpin-only dedupe per (track, length)
                        if hp_id in seen_hashes:
                            continue  # duplicate; retry another attempt
                        seen_hashes.add(hp_id)

                        # Assemble into full sequence
                        if is_insert:
                            full = original[:s] + hp_subseq + original[s:]
                        else:
                            full = original[:s] + hp_subseq + original[e:]

                        # Capture one representative "preview" per (track, L)
                        if found == 0:
                            # Include L so the preview clearly corresponds to a specific length.
                            previews.append(f"[L={L}] " + _preview_line(original, s, e, up, cap, down))

                        variant = {
                            "sequence": full,
                            "modifications": [
                                (
                                    "hp "
                                    f"L={paired_columns} grow_from={grow_from} mode={mode} "
                                    f"region={s}:{e} cap={len(cap)} "
                                    f"asym={abs(len(up) - len(down))} mm={mis_frac:.3f} gc={gc_frac:.3f} "
                                    f"rep={r} hash={hp_id} cat={track_id}"
                                )
                            ],
                            # hp_* namespace
                            "hp_region_start": s,
                            "hp_region_end": e,
                            "hp_region_is_insertion": bool(is_insert),
                            "hp_placed_by": ("anchors" if anchors is not None else "region"),
                            "hp_anchor_left_len": (len(anchors["left"]) if anchors else None),
                            "hp_anchor_right_len": (len(anchors["right"]) if anchors else None),
                            # (left/right positions only when anchors used)
                            **(
                                {}
                                if anchors is None
                                else {
                                    "hp_anchor_left_pos": int(s - len(anchors["left"])),
                                    "hp_anchor_right_pos": int(e),
                                }
                            ),
                            "hp_mode": mode,
                            "hp_grow_from": grow_from,
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
                            # Category metadata
                            "hp_category_id": track_id,
                            "hp_category_label": track_label,
                            "hp_category_kind": track_kind,
                            "hp_category_order": int(track_order),
                            # gen_* namespace
                            "gen_replicate_idx": r,
                            "gen_gc_target": gc_target,
                            "gen_mismatch_rate": mm_rate,
                        }

                        yield variant
                        found += 1
                        break  # next replicate

                total_emitted += found
                if found < quota:
                    shortfalls.append(f"L={L}:{found}/{quota}")

            # One digest line per track (INFO)
            if shortfalls:
                _LOG.info(
                    "[digest] cat=%s • L=%d..%d step=%d × %d = %d variants (short at %s)",
                    track_id,
                    L_start,
                    L_stop,
                    L_step,
                    samples_per_length,
                    total_emitted,
                    ", ".join(shortfalls),
                )
            else:
                _LOG.info(
                    "[digest] cat=%s • L=%d..%d step=%d × %d = %d variants",
                    track_id,
                    L_start,
                    L_stop,
                    L_step,
                    samples_per_length,
                    total_emitted,
                )
            # One representative preview per L for this track (DEBUG)
            for pv in previews:
                _LOG.debug("  %s", pv)
