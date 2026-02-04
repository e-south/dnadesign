"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/motif_labels.py

Helpers for mapping motif IDs to display-friendly labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.io.parsers.meme import parse_meme_file

from ..adapters.sources.pwm_jaspar import _parse_jaspar
from ..config import resolve_relative_path


def motif_display_name(motif_id: str, tf_name: str | None) -> str:
    if tf_name and str(tf_name).strip():
        return str(tf_name).strip()
    motif_id = str(motif_id)
    if "_" in motif_id:
        return motif_id.split("_", 1)[0]
    return motif_id


def _artifact_motif_metadata(path: Path) -> tuple[str, str | None]:
    raw = json.loads(path.read_text())
    motif_id = raw.get("motif_id")
    if not motif_id or not str(motif_id).strip():
        raise ValueError(f"PWM artifact missing motif_id: {path}")
    tf_name = raw.get("tf_name")
    tf_name = str(tf_name).strip() if tf_name and str(tf_name).strip() else None
    return str(motif_id).strip(), tf_name


def _filter_meme_motifs(motifs, motif_ids: list[str] | None) -> list:
    if not motif_ids:
        return list(motifs)
    keep = {m.strip().lower() for m in motif_ids if m}
    filtered = []
    for motif in motifs:
        cand = {
            getattr(motif, "motif_id", None),
            getattr(motif, "motif_name", None),
            getattr(motif, "motif_label", None),
        }
        cand = {str(x).strip().lower() for x in cand if x}
        if cand & keep:
            filtered.append(motif)
    return filtered


def _meme_motif_ids(path: Path, motif_ids: list[str] | None) -> list[str]:
    result = parse_meme_file(path)
    motifs = _filter_meme_motifs(result.motifs, motif_ids)
    labels: list[str] = []
    for motif in motifs:
        label = (
            getattr(motif, "motif_id", None)
            or getattr(motif, "motif_name", None)
            or getattr(motif, "motif_label", None)
        )
        if label:
            labels.append(str(label))
    return labels


def _jaspar_motif_labels(path: Path, motif_ids: list[str] | None) -> list[str]:
    motifs = _parse_jaspar(path)
    if motif_ids:
        keep = {m for m in motif_ids if m}
        motifs = [m for m in motifs if m.motif_id in keep]
    return [m.motif_id for m in motifs if m.motif_id]


def input_motifs(
    inp,
    cfg_path: Path,
) -> list[tuple[str, str]]:
    input_type = str(inp.type)
    motifs: list[tuple[str, str]] = []
    if input_type == "pwm_meme":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        for motif_id in _meme_motif_ids(path, getattr(inp, "motif_ids", None)):
            motifs.append((motif_id, motif_display_name(motif_id, None)))
    elif input_type == "pwm_meme_set":
        for raw in getattr(inp, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            for motif_id in _meme_motif_ids(path, getattr(inp, "motif_ids", None)):
                motifs.append((motif_id, motif_display_name(motif_id, None)))
    elif input_type == "pwm_jaspar":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        for motif in _jaspar_motif_labels(path, getattr(inp, "motif_ids", None)):
            motifs.append((motif, motif_display_name(motif, None)))
    elif input_type == "pwm_matrix_csv":
        motif_id = getattr(inp, "motif_id", None)
        if motif_id:
            motif_id = str(motif_id)
            motifs.append((motif_id, motif_display_name(motif_id, None)))
    elif input_type == "pwm_artifact":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        motif_id, tf_name = _artifact_motif_metadata(path)
        motifs.append((motif_id, motif_display_name(motif_id, tf_name)))
    elif input_type == "pwm_artifact_set":
        for raw in getattr(inp, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            motif_id, tf_name = _artifact_motif_metadata(path)
            motifs.append((motif_id, motif_display_name(motif_id, tf_name)))
    return motifs
