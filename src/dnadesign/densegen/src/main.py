"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/main.py

CLI entry point for DenseGen.

- Loads configuration YAML (default: <densegen-root>/config.yaml)
- Ingests sources (csv_tfbs, csv_sequences, usr_sequences)
- Splits work via a "quota plan" for fixed constraints (per-plan quotas)
- Samples TFBS libraries and runs dense-arrays
- Streams outputs to USR, JSONL, or both
- Logs structured INFO with stall guards and resumable progress

Run (recommended):
  python -m dnadesign.densegen.src.main --config ../config.yaml

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# Import shim so this file works BOTH as a package module and as a direct script
# --------------------------------------------------------------------------------------
try:
    from .data_ingestor import data_source_factory
    from .logging_utils import setup_logging
    from .optimizer_wrapper import DenseArrayOptimizer, random_fill
    from .outputs import build_sinks
    from .sampler import TFSampler
except Exception as _rel_import_err:
    import sys as _sys

    _THIS = Path(__file__).resolve()
    _PKG_SEARCH_ROOT = _THIS.parents[3]
    if str(_PKG_SEARCH_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PKG_SEARCH_ROOT))
    try:
        from dnadesign.densegen.src.data_ingestor import data_source_factory  # type: ignore
        from dnadesign.densegen.src.logging_utils import setup_logging  # type: ignore
        from dnadesign.densegen.src.optimizer_wrapper import (  # type: ignore
            DenseArrayOptimizer,
            random_fill,
        )
        from dnadesign.densegen.src.outputs import build_sinks  # type: ignore
        from dnadesign.densegen.src.sampler import TFSampler  # type: ignore
    except Exception:
        raise _rel_import_err

log = logging.getLogger(__name__)


# ----------------- local path helpers -----------------
def _densegen_root_from(file_path: Path) -> Path:
    """
    Walk up until we find the DenseGen root (folder that has 'src').
    Works whether run as module or direct script.
    """
    p = file_path.resolve()
    for parent in p.parents:
        if (parent / "src").is_dir():
            return parent
    return p.parents[1]


DENSEGEN_ROOT = _densegen_root_from(Path(__file__))


@dataclass
class ConfigLoader:
    path: Path

    @property
    def config(self) -> dict:
        with Path(self.path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        return raw.get("densegen", raw)


def _default_config_path() -> Path:
    return DENSEGEN_ROOT / "config.yaml"


# ----------------- helpers -----------------
def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "densegen"


def _infer_input_name(inputs_cfg: list[dict]) -> str:
    if not inputs_cfg:
        return "densegen"
    first = inputs_cfg[0] or {}
    name = (
        first.get("name")
        or Path(first.get("path", first.get("dataset", "densegen"))).stem
    )
    return _sanitize_filename(str(name))


def select_solver(
    preferred: str, fallback: str, test_length: int = 10
) -> Tuple[str, List[str]]:
    """
    Pick a solver once at startup by probing with a tiny dummy instance.
    Returns (solver_name, extra_solver_options_to_append).
    """
    import dense_arrays as da

    try:
        dummy = da.Optimizer(library=["AT"], sequence_length=test_length)
        _ = dummy.optimal(solver=preferred)
        log.info("Solver selected: %s", preferred)
        return preferred, []
    except Exception as e:
        log.warning(
            "Preferred solver %s failed during probe (%s); falling back to %s.",
            preferred,
            e,
            fallback,
        )
        return fallback, []


def _build_quota_plan(dcfg: dict) -> List[dict]:
    """
    Accepts either:
      generation.plan: [{name, quota|fraction, fixed_elements:{promoter_constraints:{...}}}, ...]
    Or legacy:
      fixed_elements.promoter_constraints: [ {name, ...}, ... ] + quota
      (even split across constraints)
    """
    plan = []
    gen = dcfg.get("generation", {})
    if gen.get("plan"):
        for item in gen["plan"]:
            assert "name" in item and (
                "quota" in item or "fraction" in item
            ), "Each plan item needs name and quota|fraction"
            fe = item.get("fixed_elements", {}) or {}
            pc = fe.get("promoter_constraints")
            if isinstance(pc, dict):
                fe["promoter_constraints"] = [pc]
            plan.append(
                {
                    "name": item["name"],
                    "quota": item.get("quota"),
                    "fraction": item.get("fraction"),
                    "fixed_elements": fe,
                }
            )
        # Resolve fractions if present
        total_quota = int(gen.get("quota", 0))
        frac_items = [
            p for p in plan if p.get("fraction") is not None and p.get("quota") is None
        ]
        if frac_items and total_quota <= 0:
            raise ValueError("Plan uses fractions but generation.quota is not set.")
        if frac_items:
            used = sum(int(p["quota"]) for p in plan if p.get("quota") is not None)
            remain = max(0, total_quota - used)
            left = remain
            for i, p in enumerate(frac_items):
                q = int(round(remain * float(p["fraction"])))
                if i == len(frac_items) - 1:
                    q = left
                p["quota"] = q
                left -= q
        return plan

    # Legacy/simple
    fe_legacy = dcfg.get("fixed_elements", {}) or {}
    pcs = fe_legacy.get("promoter_constraints")
    total_quota = int(gen.get("quota", dcfg.get("quota", 0)))
    if isinstance(pcs, list) and pcs:
        per = max(1, total_quota // len(pcs))
        for c in pcs:
            name = c.get("name", "constraint")
            plan.append(
                {
                    "name": name,
                    "quota": per,
                    "fixed_elements": {"promoter_constraints": [c]},
                }
            )
        remainder = total_quota - per * len(pcs)
        if remainder > 0:
            plan[0]["quota"] += remainder
    else:
        plan.append({"name": "default", "quota": total_quota, "fixed_elements": {}})
    return plan


def _summarize_tf_counts(tfbs_parts: List[str], max_items: int = 6) -> str:
    if not tfbs_parts:
        return ""
    counts = Counter([p.split(":", 1)[0] for p in tfbs_parts])
    items = [f"{tf}×{n}" for tf, n in counts.most_common(max_items)]
    extra = len(counts) - min(len(counts), max_items)
    return ", ".join(items) + (f" (+{extra} TFs)" if extra > 0 else "")


def _compute_used_tf_info(sol, library_for_opt, tfbs_parts, fixed_elements):
    """
    From a DenseArray solution, recover which library motifs made it into the
    final sequence, mapped back to TFs.
    Returns: (used_tfbs_simple, used_tfbs_detail, used_counts, used_tf_list)
    - used_tfbs_simple: ["TF:TFBS", ...] (one per placement; duplicates allowed)
    - used_tfbs_detail: [{"tf","tfbs","orientation","offset"}, ...]
    - used_counts: {"TF": count, ...}
    - used_tf_list: sorted list of TFs used in the solution
    """
    # Promoter motifs (if present) should not count toward TF coverage
    promoter_motifs = set()
    pc = (fixed_elements or {}).get("promoter_constraints")
    if isinstance(pc, list) and pc:
        pc = pc[0]
    if isinstance(pc, dict):
        for k in ("upstream", "downstream"):
            v = pc.get(k)
            if isinstance(v, str) and v.strip() and v.strip().lower() != "none":
                promoter_motifs.add(v.strip().upper())

    lib = getattr(sol, "library", [])
    orig_n = len(library_for_opt)
    used_simple: list[str] = []
    used_detail: list[dict] = []
    counts: dict[str, int] = {}
    used_tf_set: set[str] = set()

    # DenseArray exposes .offset_indices_in_order()
    for offset, idx in sol.offset_indices_in_order():
        base_idx = idx % len(lib)
        orientation = "fwd" if idx < len(lib) else "rev"
        motif = lib[base_idx]
        # Skip promoter motifs that were appended to lib for constraints
        if motif in promoter_motifs or base_idx >= orig_n:
            continue
        # Map back to TF + TFBS via tfbs_parts (aligned to original library order)
        meta = tfbs_parts[base_idx] if base_idx < len(tfbs_parts) else ""
        if ":" in meta:
            tf, tfbs = meta.split(":", 1)
        else:
            tf, tfbs = "", meta
        used_simple.append(meta)
        used_detail.append(
            {"tf": tf, "tfbs": tfbs, "orientation": orientation, "offset": int(offset)}
        )
        counts[tf] = counts.get(tf, 0) + 1
        used_tf_set.add(tf)

    return used_simple, used_detail, counts, sorted(used_tf_set)


def _derive_meta(
    *,
    sol,
    plan_name: str,
    tfbs_parts: List[str],
    library_for_opt: List[str],
    fixed_elements: dict,
    chosen_solver: str,
    seq_len: int,
    use_diverse: bool,
    gap_meta: dict,
    used_tfbs: List[str],
    used_tfbs_detail: List[dict],
    used_tf_counts: dict,
    used_tf_list: List[str],
    min_count_per_tf_required: int,
    covers_all_tfs_in_solution: bool,
) -> dict:
    return {
        "plan": plan_name,
        "tf_list": (
            sorted({p.split(":", 1)[0] for p in tfbs_parts}) if tfbs_parts else []
        ),
        "tfbs_parts": tfbs_parts or [],
        "used_tfbs": used_tfbs,
        "used_tfbs_detail": used_tfbs_detail,
        "used_tf_counts": used_tf_counts,
        "used_tf_list": used_tf_list,
        "covers_all_tfs_in_solution": bool(covers_all_tfs_in_solution),
        "min_count_per_tf_required": int(min_count_per_tf_required),
        "visual": str(sol),
        "compression_ratio": getattr(sol, "compression_ratio", None),
        "solver": chosen_solver,
        "diverse": bool(use_diverse),
        "library_size": len(library_for_opt),
        "sequence_length": seq_len,
        "promoter_constraint": (
            (fixed_elements or {}).get("promoter_constraints") or [{}]
        )[0].get("name"),
        # Gap-fill meta (expanded & tolerant)
        "gap_fill_used": gap_meta.get("used", False),
        "gap_fill_bases": gap_meta.get("bases"),
        "gap_fill_end": gap_meta.get("end"),
        "gap_fill_gc_min": gap_meta.get("gc_min"),
        "gap_fill_gc_max": gap_meta.get("gc_max"),
        "gap_fill_gc_actual": gap_meta.get("gc_actual"),
        "gap_fill_relaxed": gap_meta.get("relaxed"),
        "gap_fill_attempts": gap_meta.get("attempts"),
    }


# ----------------- core processor -----------------
def _process_plan_for_source(
    source_cfg: dict,
    plan_item: dict,
    global_cfg: dict,
    sinks,
    *,
    chosen_solver: str,
    one_subsample_only: bool = False,
    already_generated: int = 0,
) -> int:
    """
    Generate sequences for a single source under one plan item (quota + constraints).

    If one_subsample_only=True, we generate up to
    runtime.arrays_generated_before_resample solutions from the *current* motif
    library (or fewer if stall/dup-guard triggers), then return *before* resampling.

    Returns the number of sequences produced in this call.
    """
    # Labels and config
    source_label = (
        source_cfg.get("name")
        or Path(source_cfg.get("path", source_cfg.get("dataset", "input"))).stem
    )
    plan_name = plan_item["name"]
    quota = int(plan_item["quota"])

    gen = global_cfg.get("generation", {})
    seq_len = int(gen.get("sequence_length", global_cfg.get("sequence_length", 100)))
    sampling_cfg = gen.get("sampling", {}) or {}

    subsample_over = int(
        sampling_cfg.get(
            "subsample_over_length_budget_by",
            global_cfg.get("subsample_over_length_budget_by", 30),
        )
    )

    # Legacy mapping
    if sampling_cfg.get("unique_tf_only", False):
        sampling_cfg.setdefault("cover_all_tfs", True)
        sampling_cfg.setdefault("unique_binding_sites", True)
        sampling_cfg.setdefault("max_sites_per_tf", None)

    cover_all_tfs = bool(sampling_cfg.get("cover_all_tfs", False))
    unique_binding_sites = bool(sampling_cfg.get("unique_binding_sites", True))
    max_sites_per_tf = sampling_cfg.get("max_sites_per_tf", None)
    relax_on_exhaustion = bool(sampling_cfg.get("relax_on_exhaustion", True))

    runtime_cfg = global_cfg.get("runtime", {}) or {}
    max_per_subsample = int(
        runtime_cfg.get(
            "arrays_generated_before_resample",
            global_cfg.get("arrays_generated_before_resample", 1),
        )
    )
    # Enforce >=1 site per TF in the final *solution* (post-solve check)
    min_count_per_tf_required = int(
        runtime_cfg.get(
            "min_count_per_tf",
            1 if bool(runtime_cfg.get("require_min_count_per_tf", False)) else 0,
        )
    )

    # Support both snake_case and hyphenated/typo aliases
    max_dupes = int(
        runtime_cfg.get(
            "max_duplicate_solutions",
            runtime_cfg.get(
                "max_deduplicate_solutions",
                runtime_cfg.get("max-deduplicate-solutions", 3),
            ),
        )
    )
    max_resample_attempts = int(runtime_cfg.get("max_resample_attempts", 50))
    stall_seconds = int(runtime_cfg.get("stall_seconds_before_resample", 30))
    stall_warn_every = int(
        runtime_cfg.get("stall_warning_every_seconds", max(15, stall_seconds // 2))
    )

    post = global_cfg.get("postprocess", {}) or {}
    fill_gap = bool(post.get("fill_gap", global_cfg.get("fill_gap", False)))
    fill_end = str(post.get("fill_gap_end", global_cfg.get("fill_gap_end", "5prime")))
    fill_gc_min = float(post.get("fill_gc_min", global_cfg.get("fill_gc_min", 0.40)))
    fill_gc_max = float(post.get("fill_gc_max", global_cfg.get("fill_gc_max", 0.60)))

    solver_cfg = global_cfg.get("solver", {}) or {}
    solver_opts = list(
        solver_cfg.get("options", global_cfg.get("solver_options", ["Threads=16"]))
    )
    use_diverse = bool(
        solver_cfg.get("diverse_solution", global_cfg.get("diverse_solution", False))
    )

    log_cfg = global_cfg.get("logging", {}) or {}
    print_visual = bool(log_cfg.get("print_visual", True))

    # Load source
    src_obj = data_source_factory(source_cfg)
    data_entries, meta_df = src_obj.load_data()

    # Build initial library
    library_for_opt: List[str]
    tfbs_parts: List[str]
    if meta_df is not None and isinstance(meta_df, pd.DataFrame):
        sampler = TFSampler(meta_df)
        library_for_opt, tfbs_parts = sampler.generate_binding_site_subsample(
            seq_len,
            subsample_over,
            cover_all_tfs=cover_all_tfs,
            unique_binding_sites=unique_binding_sites,
            max_sites_per_tf=max_sites_per_tf,
            relax_on_exhaustion=relax_on_exhaustion,
        )
    else:
        all_sequences = [s for s in data_entries]
        if not all_sequences:
            raise ValueError(f"No sequences found for source {source_label}")
        take = min(
            max(1, int(sampling_cfg.get("subsample_size", 16))), len(all_sequences)
        )
        library_for_opt = random.sample(all_sequences, take)
        tfbs_parts = []

    # Library summary (succinct)
    tf_summary = _summarize_tf_counts(tfbs_parts)
    if tf_summary:
        log.info(
            "Library for %s/%s: %d motifs | TF counts: %s",
            source_label,
            plan_name,
            len(library_for_opt),
            tf_summary,
        )
    else:
        log.info(
            "Library for %s/%s: %d motifs",
            source_label,
            plan_name,
            len(library_for_opt),
        )

    # Helper to (re)build solver/generator for the current library
    fixed_elements = plan_item.get("fixed_elements", {})

    def _make_generator(_library_for_opt: List[str]):
        _optimizer_wrapper = DenseArrayOptimizer(
            library=_library_for_opt,
            sequence_length=seq_len,
            solver=chosen_solver,
            solver_options=solver_opts,
            fixed_elements=fixed_elements,
        )
        _opt = _optimizer_wrapper.get_optimizer_instance()
        _gen = (
            _opt.solutions_diverse(solver=chosen_solver, solver_options=solver_opts)
            if use_diverse and hasattr(_opt, "solutions_diverse")
            else _opt.solutions(solver=chosen_solver, solver_options=solver_opts)
        )
        return _opt, _gen

    # Initial generator
    opt, generator = _make_generator(library_for_opt)

    # generation — outer loop accumulates toward the plan quota
    global_generated = already_generated
    produced_total_this_call = 0

    while global_generated < quota:
        # "Subsample try": attempt to produce up to max_per_subsample sequences,
        # allowing up to max_resample_attempts RESAMPLES if we stall/duplicate out.
        local_generated = 0
        resamples_in_try = 0

        while local_generated < max_per_subsample and global_generated < quota:
            # Per-library state
            fingerprints = set()
            consecutive_dup = 0
            subsample_started = time.monotonic()
            last_log_warn = subsample_started
            produced_this_library = 0

            for sol in generator:
                now = time.monotonic()

                # Stall guard (only before producing anything from this library)
                if (
                    now - subsample_started >= stall_seconds
                    and produced_this_library == 0
                ):
                    log.info(
                        "[%s/%s] Stall (> %ds) with no solutions; will resample.",
                        source_label,
                        plan_name,
                        stall_seconds,
                    )
                    break
                if (
                    now - last_log_warn >= stall_warn_every
                    and produced_this_library == 0
                ):
                    log.info(
                        "[%s/%s] Still working... %.1fs on current library.",
                        source_label,
                        plan_name,
                        now - subsample_started,
                    )
                    last_log_warn = now

                # Forbid duplicates within this solver instance
                opt.forbid(sol)
                seq = sol.sequence
                if seq in fingerprints:
                    consecutive_dup += 1
                    if consecutive_dup >= max_dupes:
                        log.info(
                            "[%s/%s] Duplicate guard (>= %d in a row); will resample.",
                            source_label,
                            plan_name,
                            max_dupes,
                        )
                        break
                    continue
                consecutive_dup = 0
                fingerprints.add(seq)

                # Which TFBS actually made it in?
                used_tfbs, used_tfbs_detail, used_tf_counts, used_tf_list = (
                    _compute_used_tf_info(
                        sol, library_for_opt, tfbs_parts, fixed_elements
                    )
                )
                tf_list_from_library = (
                    sorted({p.split(":", 1)[0] for p in tfbs_parts})
                    if tfbs_parts
                    else []
                )

                # Optional per-TF min count check on the FINAL sequence
                covers_all = True
                if min_count_per_tf_required > 0 and tf_list_from_library:
                    missing = [
                        tf
                        for tf in tf_list_from_library
                        if used_tf_counts.get(tf, 0) < min_count_per_tf_required
                    ]
                    if missing:
                        covers_all = False
                        # Skip/forbid this solution; advance generator
                        continue

                # Gap fill (best-effort GC). Support both legacy and new random_fill signatures.
                gap_meta = {"used": False}
                final_seq = seq
                if fill_gap and len(final_seq) < seq_len:
                    gap = seq_len - len(final_seq)
                    rf = random_fill(gap, fill_gc_min, fill_gc_max)
                    if isinstance(rf, tuple) and len(rf) == 2:
                        pad, pad_info = rf
                        pad_info = pad_info or {}
                    else:
                        pad, pad_info = rf, {}
                    final_seq = (
                        (pad + final_seq)
                        if fill_end.lower() == "5prime"
                        else (final_seq + pad)
                    )
                    gap_meta = {
                        "used": True,
                        "bases": gap,
                        "end": fill_end,
                        "gc_min": fill_gc_min,
                        "gc_max": fill_gc_max,
                        "gc_actual": pad_info.get("gc_actual"),
                        "relaxed": pad_info.get("relaxed"),
                        "attempts": pad_info.get("attempts"),
                    }

                # Derived metadata
                derived = _derive_meta(
                    sol=sol,
                    plan_name=plan_name,
                    tfbs_parts=tfbs_parts,
                    library_for_opt=library_for_opt,
                    fixed_elements=fixed_elements,
                    chosen_solver=chosen_solver,
                    seq_len=seq_len,
                    use_diverse=use_diverse,
                    gap_meta=gap_meta,
                    used_tfbs=used_tfbs,
                    used_tfbs_detail=used_tfbs_detail,
                    used_tf_counts=used_tf_counts,
                    used_tf_list=used_tf_list,
                    min_count_per_tf_required=min_count_per_tf_required,
                    covers_all_tfs_in_solution=covers_all,
                )

                # Write
                src_label = f"densegen:{source_label}:{plan_name}"
                for sink in sinks:
                    sink.add(final_seq, derived, source_label=src_label)

                # Progress/log
                global_generated += 1
                local_generated += 1
                produced_this_library += 1
                produced_total_this_call += 1

                pct = 100.0 * (global_generated / max(1, quota))
                cr = getattr(sol, "compression_ratio", float("nan"))
                if print_visual:
                    log.info(
                        "╭─ %s/%s  %d/%d (%.2f%%) — local %d/%d — CR=%.3f\n"
                        "%s\n"
                        "sequence %s\n"
                        "╰────────────────────────────────────────────────────────",
                        source_label,
                        plan_name,
                        global_generated,
                        quota,
                        pct,
                        local_generated,
                        max_per_subsample,
                        cr,
                        derived["visual"],
                        final_seq,
                    )
                else:
                    log.info(
                        "[%s/%s] %d/%d (%.2f%%) (local %d/%d) CR=%.3f | seq %s",
                        source_label,
                        plan_name,
                        global_generated,
                        quota,
                        pct,
                        local_generated,
                        max_per_subsample,
                        cr,
                        final_seq,
                    )

                # Done for this subsample try?
                if local_generated >= max_per_subsample or global_generated >= quota:
                    break

            # If we met the local goal, break out of the per-try loop.
            if local_generated >= max_per_subsample or global_generated >= quota:
                break

            # Otherwise, we need to RESAMPLE to keep trying to reach local target.
            resamples_in_try += 1
            if resamples_in_try > max_resample_attempts:
                log.info(
                    "[%s/%s] Reached max_resample_attempts (%d) for this subsample try "
                    "(produced %d/%d here). Moving on.",
                    source_label,
                    plan_name,
                    max_resample_attempts,
                    local_generated,
                    max_per_subsample,
                )
                break  # finish this subsample try early

            # Resample a new library and continue filling the remainder of this try
            if meta_df is not None and isinstance(meta_df, pd.DataFrame):
                sampler = TFSampler(meta_df)
                library_for_opt, tfbs_parts = sampler.generate_binding_site_subsample(
                    seq_len,
                    subsample_over,
                    cover_all_tfs=cover_all_tfs,
                    unique_binding_sites=unique_binding_sites,
                    max_sites_per_tf=max_sites_per_tf,
                    relax_on_exhaustion=relax_on_exhaustion,
                )
            else:
                take = min(
                    max(1, int(sampling_cfg.get("subsample_size", 16))),
                    len(all_sequences),
                )
                library_for_opt = random.sample(all_sequences, take)
                tfbs_parts = []

            tf_summary = _summarize_tf_counts(tfbs_parts)
            if tf_summary:
                log.info(
                    "Resampled library for %s/%s: %d motifs | TF counts: %s",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                    tf_summary,
                )
            else:
                log.info(
                    "Resampled library for %s/%s: %d motifs",
                    source_label,
                    plan_name,
                    len(library_for_opt),
                )

            opt, generator = _make_generator(library_for_opt)

        # flush sinks after each subsample try
        for sink in sinks:
            sink.flush()

        # In round-robin, return to scheduler after ONE subsample try (success or not).
        if one_subsample_only:
            return produced_total_this_call

        # Otherwise continue toward quota with a *new* subsample try on next loop.

    log.info("Completed %s/%s: %d/%d", source_label, plan_name, global_generated, quota)
    return produced_total_this_call


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DenseGen — Dense Array Generator")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_default_config_path()),
        help="Path to densegen.yaml",
    )
    args = parser.parse_args(argv)

    # Load config
    cfg_path = Path(args.config)
    loader = ConfigLoader(cfg_path)
    cfg = loader.config

    # Inputs (used for default logfile naming)
    inputs = cfg.get("inputs", cfg.get("input_sources", []))
    assert (
        inputs
    ), "No inputs defined. Add at least one 'csv_tfbs', 'csv_sequences', or 'usr_sequences'."

    # Logging (auto logfile name based on first input if not provided)
    log_cfg = cfg.get("logging", {}) or {}
    default_logfile = DENSEGEN_ROOT / "logs" / f"{_infer_input_name(inputs)}.log"
    logfile = log_cfg.get("file", str(default_logfile))
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        logfile=logfile,
        suppress_solver_stderr=bool(log_cfg.get("suppress_solver_stderr", True)),
    )

    # Seed
    seed = int(
        cfg.get("runtime", {}).get(
            "random_seed", cfg.get("generation", {}).get("random_seed", 1337)
        )
    )
    random.seed(seed)

    # Build the plan (per-constraint quotas)
    plan = _build_quota_plan(cfg)
    log.info("Quota plan: %s", ", ".join(f"{p['name']}={p['quota']}" for p in plan))

    # Decide solver ONCE up-front
    solver_cfg = cfg.get("solver", {}) or {}
    preferred = str(solver_cfg.get("backend", cfg.get("solver", "CBC")))
    chosen_solver, _ = select_solver(preferred, "CBC")

    # Build sinks (USR / JSONL / both)
    sinks = list(build_sinks(cfg))

    # Process sources × plan
    round_robin = bool(
        cfg.get("runtime", {}).get("round_robin", cfg.get("round_robin", False))
    )

    if not round_robin:
        for s in inputs:
            source_cfg = copy.deepcopy(s)
            source_cfg["name"] = (
                source_cfg.get("name")
                or Path(source_cfg.get("path", source_cfg.get("dataset", "input"))).stem
            )
            for item in plan:
                _process_plan_for_source(
                    source_cfg,
                    copy.deepcopy(item),
                    cfg,
                    sinks,
                    chosen_solver=chosen_solver,
                    one_subsample_only=False,
                    already_generated=0,
                )
    else:
        # Round-robin *by subsample*: one subsample try per (source, plan) per pass.
        produced_counts: dict[tuple[str, str], int] = {}
        done = False
        while not done:
            done = True
            for s in inputs:
                source_cfg = copy.deepcopy(s)
                source_cfg["name"] = (
                    source_cfg.get("name")
                    or Path(
                        source_cfg.get("path", source_cfg.get("dataset", "input"))
                    ).stem
                )
                for item in plan:
                    key = (source_cfg["name"], item["name"])
                    current = produced_counts.get(key, 0)
                    quota = int(item["quota"])
                    if current >= quota:
                        continue  # already satisfied
                    done = False
                    produced = _process_plan_for_source(
                        source_cfg,
                        copy.deepcopy(item),
                        cfg,
                        sinks,
                        chosen_solver=chosen_solver,
                        one_subsample_only=True,  # honors arrays_generated_before_resample (+ per-try resample cap)
                        already_generated=current,
                    )
                    produced_counts[key] = current + produced

    # final flush
    for sink in sinks:
        sink.flush()
    log.info("All inputs processed. Dense arrays saved to configured sinks.")


if __name__ == "__main__":
    main()
