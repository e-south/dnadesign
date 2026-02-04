"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/usage_tracking.py

TFBS usage parsing and aggregation helpers for pipeline reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _compute_used_tf_info(
    sol,
    library_for_opt,
    regulator_labels,
    fixed_elements,
    site_id_by_index,
    source_by_index,
    tfbs_id_by_index,
    motif_id_by_index,
):
    promoter_motifs = set()
    if fixed_elements is not None:
        if hasattr(fixed_elements, "promoter_constraints"):
            pcs = getattr(fixed_elements, "promoter_constraints") or []
        else:
            pcs = (fixed_elements or {}).get("promoter_constraints") or []
        for pc in pcs:
            if hasattr(pc, "upstream") or hasattr(pc, "downstream"):
                up = getattr(pc, "upstream", None)
                dn = getattr(pc, "downstream", None)
                for v in (up, dn):
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())
            elif isinstance(pc, dict):
                for k in ("upstream", "downstream"):
                    v = pc.get(k)
                    if isinstance(v, str) and v.strip():
                        promoter_motifs.add(v.strip().upper())

    lib = getattr(sol, "library", [])
    orig_n = len(library_for_opt)
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts: dict[str, int] = {}
    used_tf_set: set[str] = set()

    for offset, idx in sol.offset_indices_in_order():
        base_idx = idx % len(lib)
        orientation = "fwd" if idx < len(lib) else "rev"
        motif = lib[base_idx]
        if motif in promoter_motifs or base_idx >= orig_n:
            continue
        tf_label = (
            regulator_labels[base_idx] if regulator_labels is not None and base_idx < len(regulator_labels) else ""
        )
        tfbs = motif
        used_simple.append(f"{tf_label}:{tfbs}" if tf_label else tfbs)
        entry = {
            "tf": tf_label,
            "tfbs": tfbs,
            "orientation": orientation,
            "offset": int(offset),
            "offset_raw": int(offset),
            "length": len(tfbs),
            "end": int(offset) + len(tfbs),
            "pad_left": 0,
        }
        if site_id_by_index is not None and base_idx < len(site_id_by_index):
            site_id = site_id_by_index[base_idx]
            if site_id is not None:
                entry["site_id"] = site_id
        if source_by_index is not None and base_idx < len(source_by_index):
            source = source_by_index[base_idx]
            if source is not None:
                entry["source"] = source
        if tfbs_id_by_index is not None and base_idx < len(tfbs_id_by_index):
            tfbs_id = tfbs_id_by_index[base_idx]
            if tfbs_id is not None:
                entry["tfbs_id"] = tfbs_id
        if motif_id_by_index is not None and base_idx < len(motif_id_by_index):
            motif_id = motif_id_by_index[base_idx]
            if motif_id is not None:
                entry["motif_id"] = motif_id
        used_detail.append(entry)
        if tf_label:
            counts[tf_label] = counts.get(tf_label, 0) + 1
            used_tf_set.add(tf_label)
    return used_simple, used_detail, counts, sorted(used_tf_set)


def _parse_used_tfbs_detail(val) -> list[dict]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                val = json.loads(s)
            except Exception as exc:
                raise ValueError(f"Failed to parse used_tfbs_detail JSON: {s[:120]}") from exc
    if isinstance(val, (list, tuple, np.ndarray)):
        out: list[dict] = []
        for item in list(val):
            if isinstance(item, dict):
                out.append(item)
        return out
    return []


def _update_usage_counts(
    usage_counts: dict[tuple[str, str], int],
    used_tfbs_detail: list[dict],
) -> None:
    for entry in used_tfbs_detail:
        tf = str(entry.get("tf") or "").strip()
        tfbs = str(entry.get("tfbs") or "").strip()
        if not tf or not tfbs:
            continue
        key = (tf, tfbs)
        usage_counts[key] = int(usage_counts.get(key, 0)) + 1


def _update_usage_summary(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    used_tfbs_detail: list[dict],
) -> None:
    _update_usage_counts(usage_counts, used_tfbs_detail)
    for entry in used_tfbs_detail:
        tf = str(entry.get("tf") or "").strip()
        if not tf:
            continue
        tf_usage_counts[tf] = int(tf_usage_counts.get(tf, 0)) + 1
