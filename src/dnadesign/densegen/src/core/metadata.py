"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/metadata.py

Centralized metadata derivation for DenseGen outputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from .metadata_schema import validate_metadata

PWM_ONLY_FIELDS = {"input_pwm_ids"}


def _apply_retention_policy(meta: dict) -> None:
    input_mode = str(meta.get("input_mode") or "").strip()
    allowed_modes = {"binding_sites", "sequence_library", "pwm_sampled", "plan_pool"}
    if input_mode not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Supported values: {allowed}.")
    if input_mode != "pwm_sampled":
        for field in PWM_ONLY_FIELDS:
            meta[field] = []


def _coerce_promoter_placements(promoter_detail: dict | None) -> list[dict]:
    detail = promoter_detail if isinstance(promoter_detail, dict) else {}
    placements = detail.get("placements", [])
    if hasattr(placements, "tolist"):
        placements = placements.tolist()
    if not isinstance(placements, (list, tuple)):
        return []
    out: list[dict] = []
    for item in placements:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _merge_parts_detail(used_tfbs_detail: List[dict], promoter_detail: dict | None, *, pad_meta: dict) -> list[dict]:
    parts: list[dict] = []
    pad_left = 0
    if bool(pad_meta.get("used")) and str(pad_meta.get("end") or "").strip().lower() == "5prime":
        pad_left = int(pad_meta.get("bases") or 0)

    for item in used_tfbs_detail or []:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        part_kind = str(entry.get("part_kind") or "tfbs").strip().lower() or "tfbs"
        entry["part_kind"] = part_kind
        if part_kind == "tfbs":
            sequence_literal = str(entry.get("sequence") or "").strip().upper()
            if sequence_literal:
                entry["sequence"] = sequence_literal
            if sequence_literal and not str(entry.get("core_sequence") or "").strip():
                entry["core_sequence"] = sequence_literal
            if str(entry.get("core_sequence") or "").strip():
                entry["core_sequence"] = str(entry.get("core_sequence")).strip().upper()
        parts.append(entry)

    for placement_index, placement in enumerate(_coerce_promoter_placements(promoter_detail)):
        name = str(placement.get("name") or "promoter").strip() or "promoter"
        spacer_length = placement.get("spacer_length")
        try:
            spacer_value = int(spacer_length) if spacer_length is not None else None
        except Exception:
            spacer_value = None
        variant_ids = placement.get("variant_ids")
        if hasattr(variant_ids, "as_py"):
            variant_ids = variant_ids.as_py()
        if not isinstance(variant_ids, dict):
            variant_ids = {}
        up_variant = str(variant_ids.get("up_id") or "").strip() or None
        down_variant = str(variant_ids.get("down_id") or "").strip() or None

        upstream_seq = str(placement.get("upstream_seq") or "").strip().upper()
        if upstream_seq:
            upstream_offset_raw = int(placement.get("upstream_start"))
            upstream_offset = upstream_offset_raw + pad_left
            parts.append(
                {
                    "part_kind": "fixed_element",
                    "role": "upstream",
                    "constraint_name": name,
                    "sequence": upstream_seq,
                    "offset_raw": upstream_offset_raw,
                    "pad_left": pad_left,
                    "offset": upstream_offset,
                    "length": len(upstream_seq),
                    "end": upstream_offset + len(upstream_seq),
                    "variant_id": up_variant,
                    "spacer_length": spacer_value,
                    "placement_index": int(placement_index),
                }
            )

        downstream_seq = str(placement.get("downstream_seq") or "").strip().upper()
        if downstream_seq:
            downstream_offset_raw = int(placement.get("downstream_start"))
            downstream_offset = downstream_offset_raw + pad_left
            parts.append(
                {
                    "part_kind": "fixed_element",
                    "role": "downstream",
                    "constraint_name": name,
                    "sequence": downstream_seq,
                    "offset_raw": downstream_offset_raw,
                    "pad_left": pad_left,
                    "offset": downstream_offset,
                    "length": len(downstream_seq),
                    "end": downstream_offset + len(downstream_seq),
                    "variant_id": down_variant,
                    "spacer_length": spacer_value,
                    "placement_index": int(placement_index),
                }
            )
    return parts


def _pad_literal(final_sequence: str, pad_meta: dict) -> str | None:
    if not bool(pad_meta.get("used", False)):
        return None
    seq = str(final_sequence or "").strip().upper()
    if not seq:
        return None
    bases = pad_meta.get("bases")
    try:
        n = int(bases)
    except Exception:
        return None
    if n <= 0:
        return None
    n = min(n, len(seq))
    end = str(pad_meta.get("end") or "").strip().lower()
    if end == "5prime":
        return seq[:n]
    if end == "3prime":
        return seq[-n:]
    return None


def build_metadata(
    *,
    sol,
    plan_name: str,
    tfbs_parts: List[str],
    regulator_labels: List[str],
    library_for_opt: List[str],
    fixed_elements,
    chosen_solver: str | None,
    solver_strategy: str,
    solver_attempt_timeout_seconds: float | None,
    solver_threads: int | None,
    solver_strands: str,
    seq_len: int,
    actual_length: int,
    pad_meta: dict,
    sampling_meta: dict,
    schema_version: str,
    created_at: str,
    run_id: str,
    run_root: str,
    run_config_path: str,
    run_config_sha256: str,
    random_seed: int,
    policy_pad: str,
    policy_sampling: str,
    policy_solver: str,
    input_meta: dict,
    fixed_elements_dump: dict,
    used_tfbs: List[str],
    used_tfbs_detail: List[dict],
    used_tf_counts: dict,
    used_tf_list: List[str],
    min_count_per_tf: int,
    covers_all_tfs_in_solution: bool,
    required_regulators: List[str],
    min_required_regulators: int | None,
    min_count_by_regulator: dict[str, int] | None,
    covers_required_regulators: bool,
    gc_total: float,
    gc_core: float,
    promoter_detail: dict | None = None,
    sequence_validation: dict | None = None,
    input_row_count: int,
    input_tf_count: int,
    input_tfbs_count: int,
    input_tf_tfbs_pair_count: int | None,
    sampling_fraction: float | None,
    sampling_fraction_pairs: float | None,
    sampling_library_index: int,
    sampling_library_hash: str,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    final_sequence: str = "",
) -> dict:
    library_unique_tfbs = len(set(library_for_opt)) if library_for_opt else 0
    library_unique_tf = len({part.split(":", 1)[0] for part in (library_for_opt or []) if ":" in str(part)})
    if promoter_detail is None:
        promoter_detail = {"placements": []}
    if sequence_validation is None:
        sequence_validation = {"validation_passed": True, "violations": []}
    used_tf_counts_list = [
        {"tf": str(tf), "count": int(n)} for tf, n in sorted(used_tf_counts.items(), key=lambda kv: kv[0])
    ]
    min_count_by_regulator_list = [
        {"tf": str(tf), "min_count": int(n)}
        for tf, n in sorted((min_count_by_regulator or {}).items(), key=lambda kv: kv[0])
    ]
    meta = {
        "schema_version": schema_version,
        "created_at": created_at,
        "run_id": run_id,
        "length": int(actual_length),
        "plan": plan_name,
        "input_name": input_meta.get("input_name"),
        "input_mode": input_meta.get("input_mode"),
        "input_pwm_ids": list(input_meta.get("input_pwm_ids") or []),
        "used_tfbs": used_tfbs,
        "used_tfbs_detail": _merge_parts_detail(used_tfbs_detail, promoter_detail, pad_meta=pad_meta),
        "used_tf_counts": used_tf_counts_list,
        "library_unique_tf_count": int(library_unique_tf),
        "library_unique_tfbs_count": int(library_unique_tfbs),
        "covers_all_tfs_in_solution": bool(covers_all_tfs_in_solution),
        "compression_ratio": getattr(sol, "compression_ratio", None),
        "sampling_library_hash": str(sampling_library_hash),
        "sampling_library_index": int(sampling_library_index),
        "required_regulators": required_regulators,
        "min_count_by_regulator": min_count_by_regulator_list,
        "pad_used": pad_meta.get("used", False),
        "pad_bases": pad_meta.get("bases"),
        "pad_end": pad_meta.get("end"),
        "pad_literal": _pad_literal(final_sequence, pad_meta),
        "gc_total": gc_total,
        "gc_core": gc_core,
        "sequence_validation": sequence_validation,
    }
    _apply_retention_policy(meta)
    validate_metadata(meta)
    return meta
