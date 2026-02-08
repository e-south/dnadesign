"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b.py

Stage-B library construction helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import logging
import random

import numpy as np
import pandas as pd

from ...config import ResolvedPlanItem, SamplingConfig
from ..artifacts.ids import hash_tfbs_id
from ..artifacts.pool import POOL_MODE_TFBS, PoolData
from ..sampler import TFSampler, compute_tf_weights
from .progress import _aggregate_failure_counts_for_sampling

log = logging.getLogger(__name__)


def _min_count_by_regulator(labels: list[str] | None, min_count_per_tf: int) -> dict[str, int] | None:
    if not labels or min_count_per_tf <= 0:
        return None
    return {tf: int(min_count_per_tf) for tf in sorted(set(labels))}


def _merge_min_counts(base: dict[str, int] | None, override: dict[str, int] | None) -> dict[str, int] | None:
    if not base and not override:
        return None
    merged = dict(base or {})
    for key, val in (override or {}).items():
        merged[key] = max(int(val), merged.get(key, 0))
    return merged


def _extract_side_biases(fixed_elements) -> tuple[list[str], list[str]]:
    if fixed_elements is None:
        return [], []
    sb = fixed_elements.side_biases
    if sb is None:
        return [], []
    return list(sb.left or []), list(sb.right or [])


def _fixed_elements_dump(fixed_elements) -> dict:
    if fixed_elements is None:
        return {
            "promoter_constraints": [],
            "side_biases": {"left": [], "right": []},
        }
    if hasattr(fixed_elements, "model_dump"):
        dump = fixed_elements.model_dump()
    else:
        dump = dict(fixed_elements or {})

    pcs_raw = dump.get("promoter_constraints") or []
    pcs = []
    keys = ("name", "upstream", "downstream", "spacer_length", "upstream_pos", "downstream_pos")
    for pc in pcs_raw:
        if hasattr(pc, "model_dump"):
            pc = pc.model_dump()
        if isinstance(pc, dict):
            pcs.append({k: pc.get(k) for k in keys})

    left, right = _extract_side_biases(fixed_elements)

    return {"promoter_constraints": pcs, "side_biases": {"left": left, "right": right}}


def _fixed_elements_label(fixed_elements) -> str | None:
    if fixed_elements is None:
        return None
    if hasattr(fixed_elements, "promoter_constraints"):
        pcs = getattr(fixed_elements, "promoter_constraints") or []
    else:
        pcs = (fixed_elements or {}).get("promoter_constraints") or []
    labels: list[str] = []
    for pc in pcs:
        name = None
        if hasattr(pc, "name"):
            name = pc.name
        elif isinstance(pc, dict):
            name = pc.get("name")
        if name is None:
            continue
        s = str(name).strip()
        if not s:
            continue
        if s not in labels:
            labels.append(s)
    if not labels:
        return None
    return "+".join(labels)


def _max_fixed_element_len(fixed_elements_dump: dict) -> int:
    max_len = 0
    pcs = fixed_elements_dump.get("promoter_constraints") or []
    for pc in pcs:
        if not isinstance(pc, dict):
            continue
        for key in ("upstream", "downstream"):
            seq = pc.get(key)
            if isinstance(seq, str):
                seq = seq.strip().upper()
                if seq:
                    max_len = max(max_len, len(seq))
    return max_len


def _min_fixed_elements_length(fixed_elements_dump: dict) -> int:
    total = 0
    pcs = fixed_elements_dump.get("promoter_constraints") or []
    for pc in pcs:
        if not isinstance(pc, dict):
            continue
        upstream = str(pc.get("upstream") or "").strip().upper()
        downstream = str(pc.get("downstream") or "").strip().upper()
        spacer = pc.get("spacer_length")
        if isinstance(spacer, (list, tuple)) and spacer:
            spacer_min = min(int(v) for v in spacer)
        elif isinstance(spacer, (int, float)):
            spacer_min = int(spacer)
        else:
            spacer_min = 0
        total += len(upstream) + len(downstream) + max(0, spacer_min)
    return total


def _require_library_bp(
    library: list[str],
    sequence_length: int,
    *,
    library_size: int | None = None,
    pool_strategy: str | None = None,
) -> int:
    return sum(len(str(s)) for s in library)


def select_group_members(
    *,
    groups: list,
    available_tfs: set[str],
    np_rng: np.random.Generator,
    sampling_strategy: str,
    weight_by_tf: dict[str, float] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    required_members: list[str] = []
    selection: dict[str, list[str]] = {}
    for group in groups:
        members = list(group.members or [])
        missing = [m for m in members if m not in available_tfs]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Regulator group '{group.name}' contains missing regulators: {preview}")
        if group.min_required > len(members):
            raise ValueError(
                f"Regulator group '{group.name}' min_required exceeds group size "
                f"({group.min_required} > {len(members)})."
            )
        if sampling_strategy == "coverage_weighted":
            weights = np.array([float(weight_by_tf.get(tf, 0.0)) if weight_by_tf else 0.0 for tf in members])
            total = float(weights.sum())
            if total <= 0:
                raise ValueError(f"Regulator group '{group.name}' has no sampling weight.")
            if group.min_required == len(members):
                selected = sorted(members)
            else:
                weights = weights / total
                chosen = np_rng.choice(len(members), size=int(group.min_required), replace=False, p=weights)
                selected = sorted([members[int(i)] for i in chosen])
        else:
            if group.min_required == len(members):
                selected = sorted(members)
            else:
                chosen = np_rng.choice(len(members), size=int(group.min_required), replace=False)
                selected = sorted([members[int(i)] for i in chosen])
        selection[group.name] = selected
        required_members.extend(selected)
    return required_members, selection


def _min_required_length_for_constraints(
    *,
    library_tfbs: list[str],
    library_tfs: list[str],
    fixed_elements_dump: dict,
    groups: list,
    min_count_by_regulator: dict[str, int] | None,
    min_count_per_tf: int,
) -> tuple[int, dict[str, int]]:
    lengths_by_tf: dict[str, list[int]] = {}
    for idx, tfbs in enumerate(library_tfbs):
        tf = library_tfs[idx] if idx < len(library_tfs) else ""
        if not tf:
            continue
        lengths_by_tf.setdefault(tf, []).append(len(str(tfbs)))
    for tf in lengths_by_tf:
        lengths_by_tf[tf].sort()

    per_tf_required: dict[str, int] = {}
    if min_count_by_regulator:
        for tf, count in min_count_by_regulator.items():
            per_tf_required[tf] = max(per_tf_required.get(tf, 0), int(count))
    if min_count_per_tf > 0:
        for tf in lengths_by_tf:
            per_tf_required[tf] = max(per_tf_required.get(tf, 0), int(min_count_per_tf))

    missing = []
    per_tf_total = 0
    for tf, count in per_tf_required.items():
        lengths = lengths_by_tf.get(tf, [])
        if len(lengths) < int(count):
            missing.append(f"{tf}({len(lengths)}/{count})")
            continue
        per_tf_total += sum(lengths[: int(count)])
    if missing:
        preview = ", ".join(missing[:6])
        raise ValueError(f"Not enough TFBS to satisfy per-regulator minimums: {preview}")

    group_required_extra = 0
    if groups:
        already_required = {tf for tf, count in per_tf_required.items() if count > 0}
        for group in groups:
            members = list(group.members or [])
            available = [tf for tf in members if tf in lengths_by_tf]
            if len(available) < int(group.min_required):
                raise ValueError(
                    f"Regulator group '{group.name}' exceeds available regulators in the Stage-B library "
                    f"({len(available)} < {int(group.min_required)})."
                )
            remaining = max(0, int(group.min_required) - len(set(available) & already_required))
            if remaining > 0:
                candidates = [lengths_by_tf[tf][0] for tf in available if tf not in already_required]
                if len(candidates) < remaining:
                    raise ValueError(
                        f"Regulator group '{group.name}' exceeds available regulators after fixed minimums "
                        f"({len(candidates)} < {remaining})."
                    )
                group_required_extra += sum(sorted(candidates)[:remaining])

    fixed_min = _min_fixed_elements_length(fixed_elements_dump)
    total = fixed_min + per_tf_total + group_required_extra
    return total, {
        "fixed_elements_min": fixed_min,
        "per_tf_min": per_tf_total,
        "min_required_extra": group_required_extra,
    }


def assess_library_feasibility(
    *,
    library_tfbs: list[str],
    library_tfs: list[str],
    fixed_elements,
    groups: list,
    min_count_by_regulator: dict[str, int] | None,
    min_count_per_tf: int,
    sequence_length: int,
) -> tuple[int, dict[str, int], dict[str, int | bool]]:
    fixed_elements_dump = _fixed_elements_dump(fixed_elements)
    min_required_len, min_breakdown = _min_required_length_for_constraints(
        library_tfbs=library_tfbs,
        library_tfs=library_tfs,
        fixed_elements_dump=fixed_elements_dump,
        groups=groups,
        min_count_by_regulator=min_count_by_regulator,
        min_count_per_tf=min_count_per_tf,
    )
    fixed_bp = int(min_breakdown["fixed_elements_min"])
    min_required_bp = int(min_required_len) - fixed_bp
    slack_bp = int(sequence_length) - int(min_required_len)
    infeasible = slack_bp < 0
    return (
        min_required_len,
        min_breakdown,
        {
            "fixed_bp": fixed_bp,
            "min_required_bp": min_required_bp,
            "slack_bp": slack_bp,
            "infeasible": infeasible,
            "sequence_length": int(sequence_length),
        },
    )


def _hash_library(
    library_for_opt: list[str],
    regulator_labels: list[str] | None,
    site_id_by_index: list[str | None] | None,
    source_by_index: list[str | None] | None,
) -> str:
    parts: list[str] = []
    for idx, motif in enumerate(library_for_opt):
        label = ""
        if regulator_labels is not None and idx < len(regulator_labels):
            label = str(regulator_labels[idx])
        site_id = None
        if site_id_by_index is not None and idx < len(site_id_by_index):
            site_id = site_id_by_index[idx]
        source = None
        if source_by_index is not None and idx < len(source_by_index):
            source = source_by_index[idx]
        payload = "\t".join(
            [
                str(motif),
                label,
                str(site_id) if site_id is not None else "None",
                str(source) if source is not None else "None",
            ]
        )
        parts.append(payload)
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def build_library_for_plan(
    *,
    source_label: str,
    plan_item: ResolvedPlanItem,
    pool: PoolData,
    sampling_cfg: SamplingConfig,
    seq_len: int,
    min_count_per_tf: int,
    usage_counts: dict[tuple[str, str], int],
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None,
    rng: random.Random,
    np_rng: np.random.Generator,
    library_index_start: int,
) -> tuple[list[str], list[str], list[str], dict]:
    pool_strategy = sampling_cfg.pool_strategy
    library_size = sampling_cfg.library_size
    library_sampling_strategy = sampling_cfg.library_sampling_strategy
    cover_all_tfs = sampling_cfg.cover_all_regulators
    unique_binding_sites = sampling_cfg.unique_binding_sites
    unique_binding_cores = sampling_cfg.unique_binding_cores
    max_sites_per_tf = sampling_cfg.max_sites_per_regulator
    relax_on_exhaustion = sampling_cfg.relax_on_exhaustion
    iterative_max_libraries = sampling_cfg.iterative_max_libraries
    iterative_min_new_solutions = sampling_cfg.iterative_min_new_solutions

    data_entries = list(pool.sequences or [])
    meta_df = pool.df if pool.pool_mode == POOL_MODE_TFBS else None

    fixed_elements = plan_item.fixed_elements
    constraints = plan_item.regulator_constraints
    groups = list(constraints.groups or [])
    plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})
    side_left, side_right = _extract_side_biases(fixed_elements)
    required_bias_motifs = list(dict.fromkeys([*side_left, *side_right]))

    libraries_built = int(library_index_start)

    def _finalize(
        library: list[str],
        parts: list[str],
        reg_labels: list[str],
        info: dict,
        *,
        site_id_by_index: list[str | None] | None,
        source_by_index: list[str | None] | None,
        tfbs_id_by_index: list[str | None] | None,
        motif_id_by_index: list[str | None] | None,
        stage_a_best_hit_score_by_index: list[float | None] | None,
        stage_a_rank_within_regulator_by_index: list[int | None] | None,
        stage_a_tier_by_index: list[int | None] | None,
        stage_a_fimo_start_by_index: list[int | None] | None,
        stage_a_fimo_stop_by_index: list[int | None] | None,
        stage_a_fimo_strand_by_index: list[str | None] | None,
        stage_a_selection_rank_by_index: list[int | None] | None,
        stage_a_selection_score_norm_by_index: list[float | None] | None,
        stage_a_tfbs_core_by_index: list[str | None] | None,
    ) -> tuple[list[str], list[str], list[str], dict]:
        nonlocal libraries_built
        libraries_built += 1
        info["library_index"] = libraries_built
        info["library_hash"] = _hash_library(library, reg_labels, site_id_by_index, source_by_index)
        info["site_id_by_index"] = site_id_by_index
        info["source_by_index"] = source_by_index
        info["tfbs_id_by_index"] = tfbs_id_by_index
        info["motif_id_by_index"] = motif_id_by_index
        info["stage_a_best_hit_score_by_index"] = stage_a_best_hit_score_by_index
        info["stage_a_rank_within_regulator_by_index"] = stage_a_rank_within_regulator_by_index
        info["stage_a_tier_by_index"] = stage_a_tier_by_index
        info["stage_a_fimo_start_by_index"] = stage_a_fimo_start_by_index
        info["stage_a_fimo_stop_by_index"] = stage_a_fimo_stop_by_index
        info["stage_a_fimo_strand_by_index"] = stage_a_fimo_strand_by_index
        info["stage_a_selection_rank_by_index"] = stage_a_selection_rank_by_index
        info["stage_a_selection_score_norm_by_index"] = stage_a_selection_score_norm_by_index
        info["stage_a_tfbs_core_by_index"] = stage_a_tfbs_core_by_index
        return library, parts, reg_labels, info

    if meta_df is not None and isinstance(meta_df, pd.DataFrame):
        if unique_binding_cores and "tfbs_core" not in meta_df.columns:
            raise ValueError(
                "generation.sampling.unique_binding_cores=true requires tfbs_core in the Stage-A pool. "
                "Rebuild pools with core-aware sampling or disable unique_binding_cores."
            )
        available_tfs = set(meta_df["tf"].tolist())
        all_group_members = list(dict.fromkeys([m for g in groups for m in g.members]))
        tfbs_counts = (
            meta_df.groupby("tf")["tfbs_core"].nunique()
            if unique_binding_cores
            else (
                meta_df.groupby("tf")["tfbs"].nunique()
                if unique_binding_sites
                else meta_df.groupby("tf")["tfbs"].size()
            )
        )
        missing = [t for t in all_group_members if t not in available_tfs]
        if missing:
            preview = ", ".join(missing[:10])
            available_preview = ", ".join(sorted(available_tfs)[:10]) if available_tfs else "n/a"
            raise ValueError(
                f"Regulator constraints reference missing regulators: {preview}. "
                f"Available regulators: {available_preview}."
            )
        if plan_min_count_by_regulator:
            missing_counts = [t for t in plan_min_count_by_regulator if t not in available_tfs]
            if missing_counts:
                preview = ", ".join(missing_counts[:10])
                raise ValueError(f"min_count_by_regulator TFs not found in input: {preview}")
            for tf, min_count in plan_min_count_by_regulator.items():
                max_allowed = int(tfbs_counts.get(tf, 0))
                if max_sites_per_tf is not None:
                    max_allowed = min(max_allowed, int(max_sites_per_tf))
                if library_size > 0:
                    max_allowed = min(max_allowed, int(library_size))
                if int(min_count) > max_allowed:
                    raise ValueError(
                        f"min_count_by_regulator[{tf}]={min_count} exceeds available sites ({max_allowed}). "
                        "Increase library_size, relax min_count_by_regulator, or allow non-unique binding sites/cores."
                    )
        if pool_strategy in {"subsample", "iterative_subsample"} and cover_all_tfs:
            if library_size > 0 and library_size < len(available_tfs):
                raise ValueError(
                    "library_size is too small to cover all regulators. "
                    f"library_size={library_size} but available_tfs={len(available_tfs)}. "
                    "Increase library_size or disable cover_all_regulators."
                )

        failure_counts_by_tfbs: dict[tuple[str, str], int] | None = None
        if library_sampling_strategy == "coverage_weighted" and sampling_cfg.avoid_failed_motifs:
            failure_counts_by_tfbs = _aggregate_failure_counts_for_sampling(
                failure_counts,
                input_name=source_label,
                plan_name=plan_item.name,
            )
        weight_df = meta_df
        if unique_binding_sites:
            weight_df = weight_df.drop_duplicates(["tf", "tfbs"]).reset_index(drop=True)
        weight_by_tf = None
        if library_sampling_strategy == "coverage_weighted":
            weight_by_tf, _, _ = compute_tf_weights(
                weight_df,
                usage_counts=usage_counts,
                coverage_boost_alpha=float(sampling_cfg.coverage_boost_alpha),
                coverage_boost_power=float(sampling_cfg.coverage_boost_power),
                failure_counts=failure_counts_by_tfbs,
                avoid_failed_motifs=bool(sampling_cfg.avoid_failed_motifs),
                failure_penalty_alpha=float(sampling_cfg.failure_penalty_alpha),
                failure_penalty_power=float(sampling_cfg.failure_penalty_power),
            )
        if groups:
            required_regulators_selected, _ = select_group_members(
                groups=groups,
                available_tfs=available_tfs,
                np_rng=np_rng,
                sampling_strategy=library_sampling_strategy,
                weight_by_tf=weight_by_tf,
            )
        else:
            required_regulators_selected = []

        if pool_strategy == "full":
            lib_df = meta_df.copy()
            if unique_binding_sites:
                lib_df = lib_df.drop_duplicates(["tf", "tfbs"])
            if unique_binding_cores:
                lib_df = lib_df.drop_duplicates(["tf", "tfbs_core"])
            if required_bias_motifs:
                missing_bias = [m for m in required_bias_motifs if m not in set(lib_df["tfbs"])]
                if missing_bias:
                    preview = ", ".join(missing_bias[:10])
                    raise ValueError(f"Required side-bias motifs not found in input: {preview}")
            lib_df = lib_df.reset_index(drop=True)
            library = lib_df["tfbs"].tolist()
            reg_labels = lib_df["tf"].tolist()
            parts = [f"{tf}:{tfbs}" for tf, tfbs in zip(reg_labels, lib_df["tfbs"].tolist())]
            site_id_by_index = lib_df["site_id"].tolist() if "site_id" in lib_df.columns else None
            source_by_index = lib_df["source"].tolist() if "source" in lib_df.columns else None
            tfbs_id_by_index = lib_df["tfbs_id"].tolist() if "tfbs_id" in lib_df.columns else None
            motif_id_by_index = lib_df["motif_id"].tolist() if "motif_id" in lib_df.columns else None
            stage_a_best_hit_score_by_index = (
                lib_df["best_hit_score"].tolist() if "best_hit_score" in lib_df.columns else None
            )
            stage_a_rank_within_regulator_by_index = (
                lib_df["rank_within_regulator"].tolist() if "rank_within_regulator" in lib_df.columns else None
            )
            stage_a_tier_by_index = lib_df["tier"].tolist() if "tier" in lib_df.columns else None
            stage_a_fimo_start_by_index = lib_df["fimo_start"].tolist() if "fimo_start" in lib_df.columns else None
            stage_a_fimo_stop_by_index = lib_df["fimo_stop"].tolist() if "fimo_stop" in lib_df.columns else None
            stage_a_fimo_strand_by_index = lib_df["fimo_strand"].tolist() if "fimo_strand" in lib_df.columns else None
            stage_a_selection_rank_by_index = (
                lib_df["selection_rank"].tolist() if "selection_rank" in lib_df.columns else None
            )
            stage_a_selection_score_norm_by_index = (
                lib_df["selection_score_norm"].tolist() if "selection_score_norm" in lib_df.columns else None
            )
            stage_a_tfbs_core_by_index = lib_df["tfbs_core"].tolist() if "tfbs_core" in lib_df.columns else None
            library_bp = _require_library_bp(
                library,
                seq_len,
                library_size=len(library),
                pool_strategy=pool_strategy,
            )
            info = {
                "achieved_length": library_bp,
                "relaxed_cap": False,
                "final_cap": None,
                "pool_strategy": pool_strategy,
                "library_size": len(library),
                "iterative_max_libraries": iterative_max_libraries,
                "iterative_min_new_solutions": iterative_min_new_solutions,
                "required_regulators_selected": required_regulators_selected,
            }
            return _finalize(
                library,
                parts,
                reg_labels,
                info,
                site_id_by_index=site_id_by_index,
                source_by_index=source_by_index,
                tfbs_id_by_index=tfbs_id_by_index,
                motif_id_by_index=motif_id_by_index,
                stage_a_best_hit_score_by_index=stage_a_best_hit_score_by_index,
                stage_a_rank_within_regulator_by_index=stage_a_rank_within_regulator_by_index,
                stage_a_tier_by_index=stage_a_tier_by_index,
                stage_a_fimo_start_by_index=stage_a_fimo_start_by_index,
                stage_a_fimo_stop_by_index=stage_a_fimo_stop_by_index,
                stage_a_fimo_strand_by_index=stage_a_fimo_strand_by_index,
                stage_a_selection_rank_by_index=stage_a_selection_rank_by_index,
                stage_a_selection_score_norm_by_index=stage_a_selection_score_norm_by_index,
                stage_a_tfbs_core_by_index=stage_a_tfbs_core_by_index,
            )

        sampler = TFSampler(meta_df, np_rng)
        required_tfs_for_library = list(
            dict.fromkeys([*required_regulators_selected, *plan_min_count_by_regulator.keys()])
        )
        if pool_strategy in {"subsample", "iterative_subsample"}:
            required_slots = len(required_bias_motifs) + len(required_tfs_for_library)
            if library_size < required_slots:
                raise ValueError(
                    "library_size is too small for required motifs. "
                    f"library_size={library_size} but required_tfbs={len(required_bias_motifs)} "
                    f"+ required_tfs={len(required_tfs_for_library)} "
                    "(regulator constraints). "
                    "Increase library_size or relax required constraints."
                )
        library, parts, reg_labels, info = sampler.generate_binding_site_library(
            library_size,
            required_tfbs=required_bias_motifs,
            required_tfs=required_tfs_for_library,
            cover_all_tfs=cover_all_tfs,
            unique_binding_sites=unique_binding_sites,
            unique_binding_cores=unique_binding_cores,
            max_sites_per_tf=max_sites_per_tf,
            relax_on_exhaustion=relax_on_exhaustion,
            sampling_strategy=library_sampling_strategy,
            usage_counts=usage_counts if library_sampling_strategy == "coverage_weighted" else None,
            coverage_boost_alpha=float(sampling_cfg.coverage_boost_alpha),
            coverage_boost_power=float(sampling_cfg.coverage_boost_power),
            failure_counts=failure_counts_by_tfbs,
            avoid_failed_motifs=bool(sampling_cfg.avoid_failed_motifs),
            failure_penalty_alpha=float(sampling_cfg.failure_penalty_alpha),
            failure_penalty_power=float(sampling_cfg.failure_penalty_power),
        )
        library_bp = _require_library_bp(
            library,
            seq_len,
            library_size=library_size,
            pool_strategy=pool_strategy,
        )
        info["achieved_length"] = library_bp
        info.update(
            {
                "pool_strategy": pool_strategy,
                "library_size": library_size,
                "library_sampling_strategy": library_sampling_strategy,
                "coverage_boost_alpha": float(sampling_cfg.coverage_boost_alpha),
                "coverage_boost_power": float(sampling_cfg.coverage_boost_power),
                "iterative_max_libraries": iterative_max_libraries,
                "iterative_min_new_solutions": iterative_min_new_solutions,
                "required_regulators_selected": required_regulators_selected,
            }
        )
        site_id_by_index = info.get("site_id_by_index")
        source_by_index = info.get("source_by_index")
        tfbs_id_by_index = info.get("tfbs_id_by_index")
        motif_id_by_index = info.get("motif_id_by_index")
        stage_a_best_hit_score_by_index = info.get("stage_a_best_hit_score_by_index")
        stage_a_rank_within_regulator_by_index = info.get("stage_a_rank_within_regulator_by_index")
        stage_a_tier_by_index = info.get("stage_a_tier_by_index")
        stage_a_fimo_start_by_index = info.get("stage_a_fimo_start_by_index")
        stage_a_fimo_stop_by_index = info.get("stage_a_fimo_stop_by_index")
        stage_a_fimo_strand_by_index = info.get("stage_a_fimo_strand_by_index")
        stage_a_selection_rank_by_index = info.get("stage_a_selection_rank_by_index")
        stage_a_selection_score_norm_by_index = info.get("stage_a_selection_score_norm_by_index")
        stage_a_tfbs_core_by_index = info.get("stage_a_tfbs_core_by_index")
        return _finalize(
            library,
            parts,
            reg_labels,
            info,
            site_id_by_index=site_id_by_index,
            source_by_index=source_by_index,
            tfbs_id_by_index=tfbs_id_by_index,
            motif_id_by_index=motif_id_by_index,
            stage_a_best_hit_score_by_index=stage_a_best_hit_score_by_index,
            stage_a_rank_within_regulator_by_index=stage_a_rank_within_regulator_by_index,
            stage_a_tier_by_index=stage_a_tier_by_index,
            stage_a_fimo_start_by_index=stage_a_fimo_start_by_index,
            stage_a_fimo_stop_by_index=stage_a_fimo_stop_by_index,
            stage_a_fimo_strand_by_index=stage_a_fimo_strand_by_index,
            stage_a_selection_rank_by_index=stage_a_selection_rank_by_index,
            stage_a_selection_score_norm_by_index=stage_a_selection_score_norm_by_index,
            stage_a_tfbs_core_by_index=stage_a_tfbs_core_by_index,
        )

    if groups or plan_min_count_by_regulator:
        members = [m for g in groups for m in g.members] if groups else []
        preview = ", ".join(members[:10]) if members else "n/a"
        raise ValueError(
            "Regulator constraints are set (groups/min_count) "
            "but the input does not provide regulators. "
            f"group_members={preview}."
        )
    all_sequences = [s for s in data_entries]
    if not all_sequences:
        raise ValueError(f"No sequences found for source {source_label}")
    pool = list(dict.fromkeys(all_sequences)) if unique_binding_sites else list(all_sequences)
    if pool_strategy == "full":
        if required_bias_motifs:
            missing = [m for m in required_bias_motifs if m not in pool]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
        library = pool
    else:
        if library_size > len(pool):
            raise ValueError(f"library_size={library_size} exceeds available unique sequences ({len(pool)}).")
        take = min(max(1, int(library_size)), len(pool))
        if required_bias_motifs:
            missing = [m for m in required_bias_motifs if m not in pool]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Required side-bias motifs not found in sequences input: {preview}")
            if take < len(required_bias_motifs):
                raise ValueError(
                    f"library_size={take} is smaller than required side_biases ({len(required_bias_motifs)})."
                )
            required_set = set(required_bias_motifs)
            remaining = [s for s in pool if s not in required_set]
            library = list(required_bias_motifs) + rng.sample(remaining, take - len(required_bias_motifs))
        else:
            library = rng.sample(pool, take)
    tf_parts: list[str] = []
    reg_labels: list[str] = []
    library_bp = _require_library_bp(library, seq_len)
    info = {
        "achieved_length": library_bp,
        "relaxed_cap": False,
        "final_cap": None,
        "pool_strategy": pool_strategy,
        "library_size": len(library) if pool_strategy == "full" else library_size,
        "iterative_max_libraries": iterative_max_libraries,
        "iterative_min_new_solutions": iterative_min_new_solutions,
    }
    tfbs_id_by_index = [
        hash_tfbs_id(motif_id=None, sequence=seq, scoring_backend="sequence_library") for seq in library
    ]
    return _finalize(
        library,
        tf_parts,
        reg_labels,
        info,
        site_id_by_index=None,
        source_by_index=None,
        tfbs_id_by_index=tfbs_id_by_index,
        motif_id_by_index=None,
        stage_a_best_hit_score_by_index=None,
        stage_a_rank_within_regulator_by_index=None,
        stage_a_tier_by_index=None,
        stage_a_fimo_start_by_index=None,
        stage_a_fimo_stop_by_index=None,
        stage_a_fimo_strand_by_index=None,
        stage_a_selection_rank_by_index=None,
        stage_a_selection_score_norm_by_index=None,
        stage_a_tfbs_core_by_index=None,
    )


def _compute_sampling_fraction(
    library: list[str],
    *,
    input_tfbs_count: int,
    pool_strategy: str,
) -> float | None:
    if pool_strategy == "full":
        return 1.0
    if input_tfbs_count > 0:
        return len(set(library)) / float(input_tfbs_count)
    return None


def _compute_sampling_fraction_pairs(
    library: list[str],
    regulator_labels: list[str] | None,
    *,
    input_pair_count: int | None,
    pool_strategy: str,
) -> float | None:
    if input_pair_count is None or input_pair_count <= 0:
        return None
    if not regulator_labels:
        return None
    if pool_strategy == "full":
        return 1.0
    pairs = set(zip(regulator_labels[: len(library)], library))
    return len(pairs) / float(input_pair_count)
