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
    stage_a_best_hit_score_by_index=None,
    stage_a_rank_within_regulator_by_index=None,
    stage_a_tier_by_index=None,
    stage_a_fimo_start_by_index=None,
    stage_a_fimo_stop_by_index=None,
    stage_a_fimo_strand_by_index=None,
    stage_a_selection_rank_by_index=None,
    stage_a_selection_score_norm_by_index=None,
    stage_a_tfbs_core_by_index=None,
    stage_a_score_theoretical_max_by_index=None,
    stage_a_selection_policy_by_index=None,
    stage_a_nearest_selected_similarity_by_index=None,
    stage_a_nearest_selected_distance_by_index=None,
    stage_a_nearest_selected_distance_norm_by_index=None,
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

    def _value(values, idx):
        if values is None or idx >= len(values):
            return None
        raw = values[idx]
        if raw is None:
            return None
        try:
            if pd.isna(raw):
                return None
        except Exception:
            pass
        return raw

    for offset, idx in sol.offset_indices_in_order():
        base_idx = idx % len(lib)
        orientation = "fwd" if idx < len(lib) else "rev"
        motif = lib[base_idx]
        if motif in promoter_motifs or base_idx >= orig_n:
            continue
        tf_label = (
            regulator_labels[base_idx] if regulator_labels is not None and base_idx < len(regulator_labels) else ""
        )
        sequence = motif
        used_simple.append(f"{tf_label}:{sequence}" if tf_label else sequence)
        entry = {
            "part_kind": "tfbs",
            "part_index": int(base_idx),
            "regulator": tf_label,
            "sequence": sequence,
            "core_sequence": sequence,
            "orientation": orientation,
            "offset": int(offset),
            "offset_raw": int(offset),
            "length": len(sequence),
            "end": int(offset) + len(sequence),
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
        best_hit_score = _value(stage_a_best_hit_score_by_index, base_idx)
        if best_hit_score is not None:
            entry["score_best_hit_raw"] = float(best_hit_score)
        score_theoretical_max = _value(stage_a_score_theoretical_max_by_index, base_idx)
        if score_theoretical_max is not None:
            entry["score_theoretical_max"] = float(score_theoretical_max)
        rank_within_regulator = _value(stage_a_rank_within_regulator_by_index, base_idx)
        if rank_within_regulator is not None:
            entry["rank_among_mined_positive"] = int(rank_within_regulator)
        tier = _value(stage_a_tier_by_index, base_idx)
        if tier is not None:
            entry["tier"] = int(tier)
        fimo_start = _value(stage_a_fimo_start_by_index, base_idx)
        if fimo_start is not None:
            entry["matched_start"] = int(fimo_start)
        fimo_stop = _value(stage_a_fimo_stop_by_index, base_idx)
        if fimo_stop is not None:
            entry["matched_stop"] = int(fimo_stop)
        fimo_strand = _value(stage_a_fimo_strand_by_index, base_idx)
        if fimo_strand is not None:
            entry["matched_strand"] = str(fimo_strand)
        selection_rank = _value(stage_a_selection_rank_by_index, base_idx)
        if selection_rank is not None:
            entry["rank_among_selected"] = int(selection_rank)
        selection_score_norm = _value(stage_a_selection_score_norm_by_index, base_idx)
        if selection_score_norm is not None:
            entry["score_relative_to_theoretical_max"] = float(selection_score_norm)
        tfbs_core = _value(stage_a_tfbs_core_by_index, base_idx)
        if tfbs_core is not None:
            entry["core_sequence"] = str(tfbs_core)
        selection_policy = _value(stage_a_selection_policy_by_index, base_idx)
        if selection_policy is not None:
            entry["selection_policy"] = str(selection_policy)
        nearest_similarity = _value(stage_a_nearest_selected_similarity_by_index, base_idx)
        if nearest_similarity is not None:
            entry["nearest_selected_similarity"] = float(nearest_similarity)
        nearest_distance = _value(stage_a_nearest_selected_distance_by_index, base_idx)
        if nearest_distance is not None:
            entry["nearest_selected_distance"] = float(nearest_distance)
        nearest_distance_norm = _value(stage_a_nearest_selected_distance_norm_by_index, base_idx)
        if nearest_distance_norm is not None:
            entry["nearest_selected_distance_norm"] = float(nearest_distance_norm)
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
    if not isinstance(val, (list, tuple, np.ndarray)):
        raise ValueError("used_tfbs_detail must be a JSON list string or list-like of dict entries.")
    out: list[dict] = []
    for item in list(val):
        if not isinstance(item, dict):
            raise ValueError("used_tfbs_detail must contain dict entries.")
        out.append(item)
    return out


def _update_usage_counts(
    usage_counts: dict[tuple[str, str], int],
    used_tfbs_detail: list[dict],
) -> None:
    for entry in used_tfbs_detail:
        regulator = str(entry.get("regulator") or "").strip()
        sequence = str(entry.get("sequence") or "").strip()
        if not regulator or not sequence:
            continue
        key = (regulator, sequence)
        usage_counts[key] = int(usage_counts.get(key, 0)) + 1


def _update_usage_summary(
    usage_counts: dict[tuple[str, str], int],
    tf_usage_counts: dict[str, int],
    used_tfbs_detail: list[dict],
) -> None:
    _update_usage_counts(usage_counts, used_tfbs_detail)
    for entry in used_tfbs_detail:
        regulator = str(entry.get("regulator") or "").strip()
        if not regulator:
            continue
        tf_usage_counts[regulator] = int(tf_usage_counts.get(regulator, 0)) + 1
