"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/io/meme_export.py

Helpers for writing minimal MEME motif files from Cruncher PWMs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from dnadesign.cruncher.core.pwm import PWM

_SAFE_ID = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")


def sanitize_meme_id(text: str) -> str:
    cleaned = "".join(ch if ch in _SAFE_ID else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "motif"


def _normalize_background(background: Sequence[float]) -> tuple[float, float, float, float]:
    if len(background) != 4:
        raise ValueError("MEME background must have 4 values in A,C,G,T order.")
    values = tuple(float(item) for item in background)
    total = sum(values)
    if total <= 0:
        raise ValueError("MEME background must sum to > 0.")
    return tuple(value / total for value in values)


def build_minimal_meme_text(
    pwm: PWM,
    *,
    motif_id: str | None = None,
    background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> tuple[str, str]:
    alphabet = tuple(str(token).upper() for token in pwm.alphabet)
    if alphabet != ("A", "C", "G", "T"):
        raise ValueError(f"Minimal MEME export requires A,C,G,T alphabet order, got {alphabet!r}.")
    bg = _normalize_background(background)
    resolved_id = sanitize_meme_id(motif_id or pwm.name)
    rows = np.asarray(pwm.matrix, dtype=float)
    lines = [
        "MEME version 4",
        "",
        "ALPHABET= ACGT",
        "",
        "strands: + -",
        "",
        "Background letter frequencies:",
        f"A {bg[0]:.6g} C {bg[1]:.6g} G {bg[2]:.6g} T {bg[3]:.6g}",
        "",
        f"MOTIF {resolved_id}",
        f"letter-probability matrix: alength= 4 w= {int(pwm.length)}",
    ]
    for row in rows:
        lines.append(f"{float(row[0]):.6g} {float(row[1]):.6g} {float(row[2]):.6g} {float(row[3]):.6g}")
    return resolved_id, "\n".join(lines) + "\n"


def write_minimal_meme_motif(
    pwm: PWM,
    out_path: Path,
    *,
    motif_id: str | None = None,
    background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> str:
    resolved_id, text = build_minimal_meme_text(pwm, motif_id=motif_id, background=background)
    out_path.write_text(text, encoding="utf-8")
    return resolved_id
