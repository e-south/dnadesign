"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/convert_legacy_tfbs.py

TFBS parsing and sequence-scan helpers for legacy densegen conversion repair flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from .convert_legacy_inputs import profile_60bp_dual_promoter

_PROMOTERS = {
    "sigma70_high": {"upstream": "TTGACA", "downstream": "TATAAT"},
    "sigma70_mid": {"upstream": "ACCGCG", "downstream": "TATAAT"},
    "sigma70_low": {"upstream": "GCAGGT", "downstream": "TATAAT"},
}

_DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(sequence: str) -> str:
    return sequence.translate(_DNA_COMP)[::-1]


def _parse_tfbs_parts(parts: Sequence[str], *, min_len: int) -> list[tuple[str, str]]:
    """
    Parse ['tf:motif', ...] into [('tf', 'MOTIF'), ...] while dropping short motifs.
    """
    out: list[tuple[str, str]] = []
    for raw in parts or []:
        if not isinstance(raw, str) or ":" not in raw:
            continue
        tf, motif = raw.split(":", 1)
        tf = (tf or "").strip().lower()
        motif = (motif or "").strip().upper()
        if not tf or not motif or len(motif) < int(min_len):
            continue
        out.append((tf, motif))
    return out


def _scan_used_tfbs(seq: str, tfbs_parts: list[tuple[str, str]]) -> tuple[list[str], list[dict], dict]:
    """
    Scan one sequence against parsed TFBS motifs and return used motifs, details, and TF counts.
    """
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts = {"cpxr": 0, "lexa": 0}

    sequence_upper = (seq or "").upper()

    for tf, motif in tfbs_parts:
        if not motif:
            continue
        forward_index = sequence_upper.find(motif)
        reverse_motif = _revcomp(motif)
        reverse_index = sequence_upper.find(reverse_motif)

        if forward_index < 0 and reverse_index < 0:
            continue

        if forward_index >= 0 and (reverse_index < 0 or forward_index <= reverse_index):
            used_simple.append(f"{tf}:{motif}")
            used_detail.append({"offset": int(forward_index), "orientation": "fwd", "tf": tf, "tfbs": motif})
        else:
            used_simple.append(f"{tf}:{motif}")
            used_detail.append({"offset": int(reverse_index), "orientation": "rev", "tf": tf, "tfbs": motif})

        if tf in counts:
            counts[tf] += 1

    used_detail.sort(key=lambda detail: (detail["offset"], detail["tf"]))
    return used_simple, used_detail, counts


def _detect_promoter_forward(seq: str, plan_name: str) -> list[dict]:
    """
    Find forward promoter motifs for the provided plan.
    """
    fallback_plan = profile_60bp_dual_promoter().densegen_plan
    plan = (plan_name or "").strip() or fallback_plan
    promoter_parts = _PROMOTERS.get(plan, _PROMOTERS[fallback_plan])
    sequence_upper = (seq or "").upper()
    extras: list[dict] = []

    for key in ("upstream", "downstream"):
        motif = promoter_parts.get(key, "")
        if not motif:
            continue
        start = 0
        while True:
            index = sequence_upper.find(motif, start)
            if index < 0:
                break
            extras.append(
                {
                    "offset": int(index),
                    "orientation": "fwd",
                    "tf": f"{plan}_{key}",
                    "tfbs": motif,
                }
            )
            start = index + 1

    extras.sort(key=lambda detail: (detail["offset"], detail["tf"]))
    return extras
