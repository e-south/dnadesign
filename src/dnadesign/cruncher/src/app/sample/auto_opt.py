"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/auto_opt.py

Auto-optimize sampling configurations and select pilot candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.elites import find_elites_parquet
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.sample.artifacts import (
    _auto_opt_best_marker_path,
    _format_run_path,
    _prune_auto_opt_runs,
    _selected_config_payload,
    _write_auto_opt_best_marker,
    _write_length_ladder_table,
)
from dnadesign.cruncher.app.sample.diagnostics import (
    _bootstrap_seed,
    _bootstrap_seed_payload,
    _bootstrap_top_k_ci,
    _draw_scores_from_sequences,
    _pooled_bootstrap_seed,
    _top_k_median_from_scores,
)
from dnadesign.cruncher.app.sample.optimizer_config import (
    _boost_beta_ladder,
    _boost_cooling,
    _format_auto_opt_config_summary,
    _format_move_probs,
    _resize_beta_ladder,
    _resolve_beta_ladder,
)
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set
from dnadesign.cruncher.app.sample.run_set import _run_sample_for_set
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    run_group_label,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig, CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pwm import PWM

logger = logging.getLogger(__name__)


@dataclass
class AutoOptCandidate:
    kind: str
    length: int | None
    budget: int | None
    cooling_boost: float
    move_profile: str
    swap_prob: float | None
    ladder_size: int | None
    move_probs: dict[str, float] | None
    move_probs_label: str | None
    run_dir: Path
    run_dirs: list[Path]
    best_score: float | None
    top_k_median_final: float | None
    best_score_final: float | None
    top_k_ci_low: float | None
    top_k_ci_high: float | None
    rhat: float | None
    ess: float | None
    unique_fraction: float | None
    balance_median: float | None
    diversity: float | None
    improvement: float | None
    acceptance_b: float | None
    acceptance_m: float | None
    acceptance_mh: float | None
    swap_rate: float | None
    status: str
    quality: str
    warnings: list[str]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class AutoOptSpec:
    kind: str
    length: int
    move_profile: str
    cooling_boost: float
    swap_prob: float | None = None
    ladder_size: int | None = None
    move_probs: dict[str, float] | None = None
    move_probs_label: str | None = None


def _stable_pilot_seed(
    *,
    pilot_cfg: SampleConfig,
    tfs: list[str],
    lockmap: dict[str, object],
    label: str,
    kind: str,
) -> int:
    moves = resolve_move_config(pilot_cfg.moves)
    payload = {
        "label": label,
        "kind": kind,
        "seed": pilot_cfg.rng.seed,
        "mode": pilot_cfg.mode,
        "objective": pilot_cfg.objective.model_dump(),
        "init": pilot_cfg.init.model_dump(),
        "budget": pilot_cfg.budget.model_dump(),
        "elites": pilot_cfg.elites.model_dump(),
        "moves": moves.model_dump(),
        "optimizer": pilot_cfg.optimizer.model_dump(),
        "optimizers": pilot_cfg.optimizers.model_dump(),
        "tfs": sorted(tfs),
        "locks": {tf: lockmap[tf].sha256 for tf in sorted(tfs)},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return int(digest[:8], 16)


def _candidate_lengths(sample_cfg: SampleConfig, auto_cfg: AutoOptConfig, pwms: dict[str, PWM]) -> list[int]:
    max_w = max(pwm.length for pwm in pwms.values()) if pwms else sample_cfg.init.length
    sum_w = sum(pwm.length for pwm in pwms.values()) if pwms else sample_cfg.init.length
    length_cfg = auto_cfg.length
    if not length_cfg.enabled:
        return [sample_cfg.init.length]
    if length_cfg.mode == "ladder":
        min_len = length_cfg.min_length if length_cfg.min_length is not None else max_w
        max_len = length_cfg.max_length if length_cfg.max_length is not None else sum_w
        min_len = max(min_len, max_w)
        max_len = max(max_len, min_len)
        lengths = list(range(min_len, max_len + 1, 1))
    else:
        min_len = length_cfg.min_length if length_cfg.min_length is not None else max_w
        max_len = length_cfg.max_length if length_cfg.max_length is not None else sample_cfg.init.length
        min_len = max(min_len, max_w)
        max_len = max(max_len, min_len)
        step = max(1, length_cfg.step)
        lengths = list(range(min_len, max_len + 1, step))
        if sample_cfg.init.length not in lengths and min_len <= sample_cfg.init.length <= max_len:
            lengths.append(sample_cfg.init.length)
        lengths = sorted(set(lengths))
        if length_cfg.max_candidates and len(lengths) > length_cfg.max_candidates:
            lengths = lengths[: length_cfg.max_candidates]
    if not lengths:
        raise ValueError("Auto-opt length selection produced no valid candidates.")
    return lengths


_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_sequence_string(seq: str) -> np.ndarray:
    clean = seq.strip().upper()
    try:
        return np.array([_BASE_TO_INT[ch] for ch in clean], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Invalid base in sequence '{seq}'") from exc


def _sample_insert_base(pad_with: str | None, rng: np.random.Generator) -> np.int8:
    if pad_with is None or pad_with == "background":
        return np.int8(rng.integers(0, 4))
    base = pad_with.upper()
    if base not in _BASE_TO_INT:
        raise ValueError(f"Invalid pad_with='{pad_with}' for warm-start insertion")
    return np.int8(_BASE_TO_INT[base])


def _extend_sequence_by_one(seq_arr: np.ndarray, *, rng: np.random.Generator, pad_with: str | None) -> np.ndarray:
    insert_pos = int(rng.integers(0, seq_arr.size + 1))
    insert_base = _sample_insert_base(pad_with, rng)
    return np.concatenate([seq_arr[:insert_pos], np.array([insert_base], dtype=np.int8), seq_arr[insert_pos:]])


def _warm_start_seeds_from_elites(
    run_dir: Path,
    *,
    target_length: int,
    rng: np.random.Generator,
    pad_with: str | None,
    max_seeds: int | None = None,
) -> list[np.ndarray]:
    try:
        elite_path = find_elites_parquet(run_dir)
        elites_df = read_parquet(elite_path)
    except Exception as exc:
        logger.warning("Warm-start: unable to load elites from %s (%s).", run_dir, exc)
        return []
    if "sequence" not in elites_df.columns:
        return []
    seeds: list[np.ndarray] = []
    for seq_str in elites_df["sequence"].astype(str).tolist():
        seq_arr = _encode_sequence_string(seq_str)
        if seq_arr.size == target_length:
            seeds.append(seq_arr.copy())
        elif seq_arr.size == target_length - 1:
            seeds.append(_extend_sequence_by_one(seq_arr, rng=rng, pad_with=pad_with))
        else:
            continue
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    return seeds


def _warm_start_seeds_from_sequences(
    run_dir: Path,
    *,
    target_length: int,
    rng: np.random.Generator,
    pad_with: str | None,
    max_seeds: int | None = None,
) -> list[np.ndarray]:
    seq_path = sequences_path(run_dir)
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Warm-start requires artifacts/sequences.parquet in {run_dir}. "
            "Re-run `cruncher sample` with sample.output.save_sequences=true."
        )
    try:
        seq_df = read_parquet(seq_path)
    except Exception as exc:
        raise ValueError(f"Warm-start failed to read {seq_path}.") from exc
    if "sequence" not in seq_df.columns:
        raise ValueError(f"Warm-start requires a 'sequence' column in {seq_path}.")
    if "phase" in seq_df.columns:
        seq_df = seq_df[seq_df["phase"] == "draw"].copy()

    score_cols = [col for col in seq_df.columns if col.startswith("score_")]
    use_combined = "combined_score_final" in seq_df.columns

    candidates: list[tuple[float, np.ndarray]] = []
    for _, row in seq_df.iterrows():
        seq_str = str(row["sequence"])
        seq_arr = _encode_sequence_string(seq_str)
        score = 0.0
        if use_combined:
            raw = row.get("combined_score_final")
            score = float(raw) if raw is not None else 0.0
        elif score_cols:
            vals = [row.get(col) for col in score_cols]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                score = float(np.mean(vals))
        if seq_arr.size == target_length:
            candidates.append((score, seq_arr.copy()))
        elif seq_arr.size == target_length - 1:
            candidates.append((score, _extend_sequence_by_one(seq_arr, rng=rng, pad_with=pad_with)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    seeds: list[np.ndarray] = []
    for _, seq_arr in candidates:
        seeds.append(seq_arr)
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    if not seeds:
        raise ValueError(
            "Warm-start found no usable sequences. "
            f"Expected length {target_length} or {target_length - 1} in {seq_path}."
        )
    return seeds


def _pilot_budget_levels(base_cfg: SampleConfig, auto_cfg: AutoOptConfig) -> list[int]:
    base_total = base_cfg.budget.tune + base_cfg.budget.draws
    if base_total < 4:
        raise ValueError("auto_opt requires sample.budget.tune+draws >= 4")
    budgets = sorted({min(level, base_total) for level in auto_cfg.budget_levels})
    budgets = [b for b in budgets if b >= 4]
    if not budgets:
        raise ValueError("auto_opt.budget_levels produced no valid pilot budgets")
    return budgets


def _budget_to_tune_draws(base_cfg: SampleConfig, total_sweeps: int) -> tuple[int, int]:
    base_total = base_cfg.budget.tune + base_cfg.budget.draws
    ratio = base_cfg.budget.tune / base_total if base_total > 0 else 0.0
    tune = int(round(total_sweeps * ratio))
    draws = total_sweeps - tune
    if draws < 4:
        draws = 4
        tune = max(0, total_sweeps - draws)
    return tune, draws


def _build_pilot_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    auto_cfg: AutoOptConfig,
    total_sweeps: int,
    cooling_boost: float = 1.0,
    length_override: int | None = None,
    move_profile: str | None = None,
    swap_prob: float | None = None,
    ladder_size: int | None = None,
    move_probs: dict[str, float] | None = None,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    pilot = base_cfg.model_copy(deep=True)
    tune, draws = _budget_to_tune_draws(base_cfg, total_sweeps)
    pilot.budget.tune = tune
    pilot.budget.draws = draws
    if base_cfg.budget.tune != tune or base_cfg.budget.draws != draws:
        notes.append("overrode tune/draws for pilot budget")

    pilot.output.trace.save = True
    if not base_cfg.output.trace.save:
        notes.append("enabled output.trace.save for pilot diagnostics")
    pilot.output.save_sequences = True
    if not base_cfg.output.save_sequences:
        notes.append("enabled output.save_sequences for pilot diagnostics")
    pilot.output.trace.include_tune = False
    if base_cfg.output.trace.include_tune:
        notes.append("disabled output.trace.include_tune for pilots")
    pilot.output.live_metrics = False
    if base_cfg.output.live_metrics:
        notes.append("disabled output.live_metrics for pilots")
    pilot.ui.progress_bar = False
    pilot.ui.progress_every = 0
    if not auto_cfg.allow_trim_polish_in_pilots:
        if pilot.output.trim.enabled:
            pilot.output.trim.enabled = False
            notes.append("disabled output.trim.enabled for pilots")
        if pilot.output.polish.enabled:
            pilot.output.polish.enabled = False
            notes.append("disabled output.polish.enabled for pilots")

    if length_override is not None and pilot.init.length != length_override:
        pilot.init.length = int(length_override)
        notes.append(f"overrode init.length={pilot.init.length} for auto-opt length")

    pilot.optimizer.name = kind
    if kind == "pt" and pilot.budget.restarts != 1:
        pilot.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT pilots")

    if kind == "gibbs":
        adaptive = pilot.optimizers.gibbs.adaptive_beta
        if not adaptive.enabled:
            adaptive.enabled = True
            notes.append("enabled optimizers.gibbs.adaptive_beta for pilots")
        if adaptive.target_acceptance != 0.35:
            adaptive.target_acceptance = 0.35
            notes.append("set adaptive_beta.target_acceptance=0.35 for pilots")
        if adaptive.window != 100:
            adaptive.window = 100
            notes.append("set adaptive_beta.window=100 for pilots")
        if adaptive.k != 0.5:
            adaptive.k = 0.5
            notes.append("set adaptive_beta.k=0.5 for pilots")
        if adaptive.max_beta != 100.0:
            adaptive.max_beta = 100.0
            notes.append("set adaptive_beta.max_beta=100.0 for pilots")
        if tuple(adaptive.moves) != ("B", "M"):
            adaptive.moves = ["B", "M"]
            notes.append("set adaptive_beta.moves=[B,M] for pilots")

    if move_profile is not None and pilot.moves.profile != move_profile:
        pilot.moves.profile = move_profile
        notes.append(f"set moves.profile={move_profile} for pilots")
    if move_probs is not None:
        pilot.moves.overrides.move_probs = dict(move_probs)
        notes.append("set moves.overrides.move_probs for pilots")

    if kind == "pt":
        ladder = pilot.optimizers.pt.ladder_adapt
        if not ladder.enabled:
            ladder.enabled = True
            notes.append("enabled optimizers.pt.ladder_adapt for pilots")
        if ladder.target_swap != 0.25:
            ladder.target_swap = 0.25
            notes.append("set ladder_adapt.target_swap=0.25 for pilots")
        if ladder.window != 50:
            ladder.window = 50
            notes.append("set ladder_adapt.window=50 for pilots")
        if ladder.k != 0.5:
            ladder.k = 0.5
            notes.append("set ladder_adapt.k=0.5 for pilots")
        if ladder.max_scale != 50.0:
            ladder.max_scale = 50.0
            notes.append("set ladder_adapt.max_scale=50.0 for pilots")
        if swap_prob is not None and pilot.optimizers.pt.swap_prob != float(swap_prob):
            pilot.optimizers.pt.swap_prob = float(swap_prob)
            notes.append(f"set optimizers.pt.swap_prob={pilot.optimizers.pt.swap_prob:g} for pilots")
        if ladder_size is not None:
            current_ladder, _ = _resolve_beta_ladder(pilot.optimizers.pt)
            if int(ladder_size) != len(current_ladder):
                resized, resize_notes = _resize_beta_ladder(pilot.optimizers.pt.beta_ladder, int(ladder_size))
                pilot.optimizers.pt.beta_ladder = resized
                notes.extend(resize_notes)

    if kind == "gibbs" and cooling_boost != 1:
        schedule = pilot.optimizers.gibbs.beta_schedule
        boosted, boost_notes = _boost_cooling(schedule, cooling_boost)
        pilot.optimizers.gibbs.beta_schedule = boosted
        notes.extend(boost_notes)
    if kind == "pt" and cooling_boost != 1:
        ladder_cfg = pilot.optimizers.pt.beta_ladder
        boosted, boost_notes = _boost_beta_ladder(ladder_cfg, cooling_boost)
        pilot.optimizers.pt.beta_ladder = boosted
        notes.extend(boost_notes)

    return pilot, notes


def _build_final_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    length_override: int | None = None,
    cooling_boost: float = 1.0,
    move_profile: str | None = None,
    swap_prob: float | None = None,
    ladder_size: int | None = None,
    move_probs: dict[str, float] | None = None,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    final_cfg = base_cfg.model_copy(deep=True)
    if length_override is not None and final_cfg.init.length != length_override:
        final_cfg.init.length = int(length_override)
        notes.append(f"set init.length={final_cfg.init.length} from auto-opt selection")
    final_cfg.optimizer.name = kind
    if kind == "pt" and final_cfg.budget.restarts != 1:
        final_cfg.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT final run")
    if kind == "gibbs" and cooling_boost != 1:
        schedule = final_cfg.optimizers.gibbs.beta_schedule
        boosted, boost_notes = _boost_cooling(schedule, cooling_boost)
        final_cfg.optimizers.gibbs.beta_schedule = boosted
        notes.extend(boost_notes)
    if kind == "pt" and cooling_boost != 1:
        ladder_cfg = final_cfg.optimizers.pt.beta_ladder
        boosted, boost_notes = _boost_beta_ladder(ladder_cfg, cooling_boost)
        final_cfg.optimizers.pt.beta_ladder = boosted
        notes.extend(boost_notes)
    if move_profile is not None and final_cfg.moves.profile != move_profile:
        final_cfg.moves.profile = move_profile
        notes.append(f"set moves.profile={move_profile} from auto-opt selection")
    if move_probs is not None:
        final_cfg.moves.overrides.move_probs = dict(move_probs)
        notes.append("set moves.overrides.move_probs from auto-opt selection")
    if kind == "pt":
        if swap_prob is not None and final_cfg.optimizers.pt.swap_prob != float(swap_prob):
            final_cfg.optimizers.pt.swap_prob = float(swap_prob)
            notes.append(f"set optimizers.pt.swap_prob={final_cfg.optimizers.pt.swap_prob:g} from auto-opt selection")
        if ladder_size is not None:
            current_ladder, _ = _resolve_beta_ladder(final_cfg.optimizers.pt)
            if int(ladder_size) != len(current_ladder):
                resized, resize_notes = _resize_beta_ladder(final_cfg.optimizers.pt.beta_ladder, int(ladder_size))
                final_cfg.optimizers.pt.beta_ladder = resized
                notes.extend(resize_notes)
    return final_cfg, notes


def _assess_candidate_quality(
    candidate: AutoOptCandidate,
    auto_cfg: AutoOptConfig,
    *,
    mode: str,
) -> list[str]:
    notes: list[str] = []
    if candidate.status == "fail":
        candidate.quality = "fail"
        return notes
    if mode != "auto_opt":
        candidate.quality = "ok"
        return notes
    rhat_ok = 1.15
    rhat_fail = 1.30
    ess_ok = 50
    ess_fail = 10
    unique_ok = 0.10
    unique_fail = 0.0

    pilot_draws = None
    pilot_short = False
    if isinstance(candidate.diagnostics, dict):
        metrics = candidate.diagnostics.get("metrics")
        if isinstance(metrics, dict):
            trace = metrics.get("trace")
            if isinstance(trace, dict):
                pilot_draws = trace.get("draws")
    try:
        pilot_draws = int(pilot_draws) if pilot_draws is not None else None
    except (TypeError, ValueError):
        pilot_draws = None
    pilot_min_draws = 200
    if pilot_draws is not None and pilot_draws < pilot_min_draws:
        pilot_short = True
        notes.append(f"pilot draws={pilot_draws} < {pilot_min_draws}; diagnostics are directional at short budgets")

    quality = "ok"
    if not pilot_short:
        if candidate.rhat is not None:
            if candidate.rhat >= rhat_fail:
                quality = "fail"
                notes.append(f"rhat={candidate.rhat:.3f} >= {rhat_fail}")
            elif candidate.rhat > rhat_ok:
                quality = "warn"
                notes.append(f"rhat={candidate.rhat:.3f} > {rhat_ok}")
        if candidate.ess is not None:
            if candidate.ess < ess_fail:
                quality = "fail"
                notes.append(f"ess={candidate.ess:.1f} < {ess_fail}")
            elif candidate.ess < ess_ok and quality != "fail":
                quality = "warn"
                notes.append(f"ess={candidate.ess:.1f} < {ess_ok}")
        if candidate.unique_fraction is not None:
            if candidate.unique_fraction <= unique_fail:
                quality = "fail"
                notes.append(f"unique_fraction={candidate.unique_fraction:.2f} <= {unique_fail:.2f}")
            elif candidate.unique_fraction < unique_ok and quality != "fail":
                quality = "warn"
                notes.append(f"unique_fraction={candidate.unique_fraction:.2f} < {unique_ok:.2f}")

    candidate.quality = quality
    if quality == "warn" and not pilot_short:
        notes.append("pilot diagnostics are weak; consider increasing budgets or beta schedules")
    return notes


def _run_auto_optimize_for_set(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    set_index: int,
    set_count: int,
    include_set_index: bool,
    tfs: list[str],
    lockmap: dict[str, object],
    sample_cfg: SampleConfig,
    auto_cfg: AutoOptConfig,
) -> Path:
    logger.info("Auto-optimize enabled: running pilot sweeps (gibbs + pt).")
    run_group = run_group_label(tfs, set_index, include_set_index=include_set_index)
    pwms = _load_pwms_for_set(cfg=cfg, config_path=config_path, tfs=tfs, lockmap=lockmap)
    lengths = _candidate_lengths(sample_cfg, auto_cfg, pwms)
    budgets = _pilot_budget_levels(sample_cfg, auto_cfg)
    move_profiles = list(dict.fromkeys(auto_cfg.move_profiles or [sample_cfg.moves.profile]))
    cooling_boosts = list(dict.fromkeys(auto_cfg.cooling_boosts or [1.0]))
    beta_schedule_scales = list(dict.fromkeys(auto_cfg.beta_schedule_scales or []))
    beta_ladder_scales = list(dict.fromkeys(auto_cfg.beta_ladder_scales or []))
    pt_swap_probs = list(dict.fromkeys(auto_cfg.pt_swap_probs or [sample_cfg.optimizers.pt.swap_prob]))
    pt_ladder_sizes = list(dict.fromkeys(auto_cfg.pt_ladder_sizes or []))
    if not pt_ladder_sizes:
        ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
        pt_ladder_sizes = [len(ladder)]
    gibbs_move_probs = list(auto_cfg.gibbs_move_probs or [])
    gibbs_move_variants: list[tuple[dict[str, float] | None, str | None]] = [(None, None)]
    if gibbs_move_probs:
        gibbs_move_variants = [(dict(mp), f"mp{idx + 1}") for idx, mp in enumerate(gibbs_move_probs)]
    gibbs_scales = sorted(set(cooling_boosts + (beta_schedule_scales or [])))
    pt_scales = sorted(set(cooling_boosts + (beta_ladder_scales or [])))
    logger.info("Auto-opt length candidates: %s", ", ".join(str(L) for L in lengths))
    logger.info("Auto-opt budget levels: %s", ", ".join(str(b) for b in budgets))
    logger.info("Auto-opt move profiles: %s", ", ".join(move_profiles))
    logger.info("Auto-opt gibbs beta scales: %s", ", ".join(f"{b:g}" for b in gibbs_scales))
    logger.info("Auto-opt pt beta scales: %s", ", ".join(f"{b:g}" for b in pt_scales))
    logger.info("Auto-opt PT swap_probs: %s", ", ".join(f"{p:g}" for p in pt_swap_probs))
    logger.info("Auto-opt PT ladder sizes: %s", ", ".join(str(n) for n in pt_ladder_sizes))
    if gibbs_move_probs:
        logger.info(
            "Auto-opt gibbs move_probs variants: %s",
            ", ".join(label or _format_move_probs(mp) for mp, label in gibbs_move_variants if mp),
        )
    logger.info("Auto-opt keep_pilots: %s", auto_cfg.keep_pilots)

    length_cfg = auto_cfg.length
    candidate_specs: list[AutoOptSpec] = []
    for length in lengths:
        for kind in ("gibbs", "pt"):
            scales = gibbs_scales if kind == "gibbs" else pt_scales
            for move_profile in move_profiles:
                if kind == "gibbs":
                    for move_probs, move_label in gibbs_move_variants:
                        for scale in scales:
                            candidate_specs.append(
                                AutoOptSpec(
                                    kind=kind,
                                    length=length,
                                    move_profile=move_profile,
                                    cooling_boost=scale,
                                    move_probs=move_probs,
                                    move_probs_label=move_label,
                                )
                            )
                else:
                    for swap_prob in pt_swap_probs:
                        for ladder_size in pt_ladder_sizes:
                            for scale in scales:
                                candidate_specs.append(
                                    AutoOptSpec(
                                        kind=kind,
                                        length=length,
                                        move_profile=move_profile,
                                        cooling_boost=scale,
                                        swap_prob=swap_prob,
                                        ladder_size=ladder_size,
                                    )
                                )
    seed_rng = np.random.default_rng(sample_cfg.rng.seed)

    def _run_candidate(
        *,
        kind: str,
        length: int,
        budget: int,
        label: str,
        cooling_boost: float,
        move_profile: str,
        swap_prob: float | None = None,
        ladder_size: int | None = None,
        move_probs: dict[str, float] | None = None,
        move_probs_label: str | None = None,
        init_seeds: list[np.ndarray] | None = None,
    ) -> AutoOptCandidate:
        runs: list[AutoOptCandidate] = []
        for rep in range(auto_cfg.replicates):
            pilot_cfg, notes = _build_pilot_sample_cfg(
                sample_cfg,
                kind=kind,
                auto_cfg=auto_cfg,
                total_sweeps=budget,
                cooling_boost=cooling_boost,
                length_override=length,
                move_profile=move_profile,
                swap_prob=swap_prob,
                ladder_size=ladder_size,
                move_probs=move_probs,
            )
            seed_label = f"{label}_R{rep + 1}"
            if sample_cfg.rng.deterministic:
                seed = _stable_pilot_seed(pilot_cfg=pilot_cfg, tfs=tfs, lockmap=lockmap, label=seed_label, kind=kind)
            else:
                seed = int(seed_rng.integers(0, 2**31 - 1))
            pilot_cfg.rng.seed = seed
            notes.append(f"set pilot seed={seed}")
            if notes:
                logger.debug(
                    "Auto-opt %s %s (L=%d B=%d R=%d) overrides: %s",
                    label,
                    kind,
                    pilot_cfg.init.length,
                    budget,
                    rep + 1,
                    "; ".join(notes),
                )
            pilot_meta = {
                "mode": "pilot",
                "attempt": label,
                "candidate": kind,
                "length": pilot_cfg.init.length,
                "seed": pilot_cfg.rng.seed,
                "draws": pilot_cfg.budget.draws,
                "tune": pilot_cfg.budget.tune,
                "restarts": pilot_cfg.budget.restarts,
                "budget": budget,
                "replicate": rep + 1,
                "cooling_boost": cooling_boost,
                "beta_scale": cooling_boost,
                "move_profile": move_profile,
                "swap_prob": swap_prob,
                "ladder_size": ladder_size,
                "move_probs_label": move_probs_label,
            }
            pilot_run_dir = _run_sample_for_set(
                cfg,
                config_path,
                set_index=set_index,
                set_count=set_count,
                include_set_index=include_set_index,
                tfs=tfs,
                lockmap=lockmap,
                sample_cfg=pilot_cfg,
                stage="auto_opt",
                run_kind=f"auto_opt_{label}_{kind}_L{pilot_cfg.init.length}_B{budget}_R{rep + 1}",
                auto_opt_meta=pilot_meta,
                init_seeds=init_seeds,
            )
            try:
                candidate = _evaluate_pilot_run(
                    pilot_run_dir,
                    kind,
                    budget=budget,
                    mode=sample_cfg.mode,
                    scorecard_top_k=auto_cfg.policy.scorecard.top_k,
                    cooling_boost=cooling_boost,
                    move_profile=move_profile,
                    swap_prob=swap_prob,
                    ladder_size=ladder_size,
                    move_probs=move_probs,
                    move_probs_label=move_probs_label,
                )
                if candidate.length is None:
                    candidate.length = pilot_cfg.init.length
            except Exception as exc:
                logger.warning("Auto-opt %s %s failed: %s", label, kind, exc)
                candidate = AutoOptCandidate(
                    kind=kind,
                    length=pilot_cfg.init.length,
                    budget=budget,
                    cooling_boost=cooling_boost,
                    move_profile=move_profile,
                    swap_prob=swap_prob,
                    ladder_size=ladder_size,
                    move_probs=move_probs,
                    move_probs_label=move_probs_label,
                    run_dir=pilot_run_dir,
                    run_dirs=[pilot_run_dir],
                    best_score=None,
                    top_k_median_final=None,
                    best_score_final=None,
                    top_k_ci_low=None,
                    top_k_ci_high=None,
                    rhat=None,
                    ess=None,
                    unique_fraction=None,
                    balance_median=None,
                    diversity=None,
                    improvement=None,
                    acceptance_b=None,
                    acceptance_m=None,
                    acceptance_mh=None,
                    swap_rate=None,
                    status="fail",
                    quality="fail",
                    warnings=[str(exc)],
                    diagnostics={},
                )
            runs.append(candidate)
        return _aggregate_candidate_runs(runs, budget=budget, scorecard_top_k=auto_cfg.policy.scorecard.top_k)

    def _log_candidate(candidate: AutoOptCandidate, *, label: str) -> None:
        diag_status = "n/a"
        if isinstance(candidate.diagnostics, dict):
            diag_status = candidate.diagnostics.get("status") or "n/a"
        move_probs_label = "-"
        if candidate.move_probs is not None:
            move_probs_label = _format_move_probs(candidate.move_probs)
        logger.debug(
            (
                "Auto-opt %s %s L=%s B=%s beta=%s move=%s swap=%s ladder=%s moveset=%s: "
                "scorecard=%s diagnostics=%s top_k_median=%s best=%s balance=%s diversity=%s rhat=%s ess=%s "
                "unique_fraction=%s"
            ),
            label,
            candidate.kind,
            candidate.length if candidate.length is not None else "n/a",
            candidate.budget if candidate.budget is not None else "n/a",
            f"{candidate.cooling_boost:g}",
            candidate.move_profile,
            f"{candidate.swap_prob:g}" if candidate.swap_prob is not None else "n/a",
            str(candidate.ladder_size) if candidate.ladder_size is not None else "n/a",
            move_probs_label,
            candidate.quality,
            diag_status,
            f"{candidate.top_k_median_final:.3f}" if candidate.top_k_median_final is not None else "n/a",
            f"{candidate.best_score_final:.3f}" if candidate.best_score_final is not None else "n/a",
            f"{candidate.balance_median:.3f}" if candidate.balance_median is not None else "n/a",
            f"{candidate.diversity:.3f}" if candidate.diversity is not None else "n/a",
            f"{candidate.rhat:.3f}" if candidate.rhat is not None else "n/a",
            f"{candidate.ess:.1f}" if candidate.ess is not None else "n/a",
            f"{candidate.unique_fraction:.2f}" if candidate.unique_fraction is not None else "n/a",
        )
        if candidate.warnings:
            is_pilot = label.startswith("pilot_")
            if is_pilot:
                log_fn = logger.info
            else:
                log_fn = logger.warning if candidate.quality == "fail" else logger.info
            log_fn("Auto-opt %s %s warnings: %s", label, candidate.kind, "; ".join(candidate.warnings))

    all_candidates: list[AutoOptCandidate] = []
    final_level_candidates: list[AutoOptCandidate] = []
    length_summary_rows: list[dict[str, object]] = []

    def _scaled_budgets(base_budgets: list[int], *, scale: float) -> list[int]:
        base_total = sample_cfg.budget.tune + sample_cfg.budget.draws
        return [min(base_total, max(4, int(round(b * scale)))) for b in base_budgets]

    def _spec_label(spec: AutoOptSpec, *, idx: int, budget: int) -> str:
        label = f"pilot_{idx}_L{spec.length}_B{budget}_M{spec.move_profile}_C{spec.cooling_boost:g}"
        if spec.kind == "pt":
            swap_label = f"{spec.swap_prob:g}" if spec.swap_prob is not None else "n/a"
            ladder_label = str(spec.ladder_size) if spec.ladder_size is not None else "n/a"
            label = f"{label}_S{swap_label}_T{ladder_label}"
        if spec.move_probs_label:
            label = f"{label}_{spec.move_probs_label}"
        return label

    if length_cfg.mode == "ladder":
        previous_best_run: Path | None = None
        final_level_candidates_all: list[AutoOptCandidate] = []
        for length_idx, length in enumerate(lengths):
            length_budgets = budgets
            if length_idx > 0:
                length_budgets = _scaled_budgets(budgets, scale=length_cfg.ladder_budget_scale)
            init_seeds: list[np.ndarray] | None = None
            if length_cfg.warm_start and previous_best_run is not None:
                init_seeds = _warm_start_seeds_from_sequences(
                    previous_best_run,
                    target_length=length,
                    rng=seed_rng,
                    pad_with=sample_cfg.init.pad_with,
                    max_seeds=sample_cfg.elites.k,
                )
            current_specs: list[AutoOptSpec] = []
            for spec in candidate_specs:
                if spec.length == length:
                    current_specs.append(spec)
            level_candidates: list[AutoOptCandidate] = []
            length_final_candidates: list[AutoOptCandidate] = []
            start_time = time.perf_counter()
            for idx, budget in enumerate(length_budgets):
                level_candidates = []
                for spec in current_specs:
                    label = _spec_label(spec, idx=idx + 1, budget=budget)
                    candidate = _run_candidate(
                        kind=spec.kind,
                        length=spec.length,
                        budget=budget,
                        label=label,
                        cooling_boost=spec.cooling_boost,
                        move_profile=spec.move_profile,
                        swap_prob=spec.swap_prob,
                        ladder_size=spec.ladder_size,
                        move_probs=spec.move_probs,
                        move_probs_label=spec.move_probs_label,
                        init_seeds=init_seeds,
                    )
                    candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode="auto_opt")
                    if candidate_notes:
                        candidate.warnings.extend(candidate_notes)
                    _log_candidate(candidate, label=label)
                    level_candidates.append(candidate)
                    all_candidates.append(candidate)
                confidence_level, _, _ = _confidence_from_candidates(level_candidates, auto_cfg)
                if confidence_level == "high":
                    length_final_candidates = level_candidates
                    if idx < len(length_budgets) - 1:
                        logger.info(
                            "Auto-opt length %s: confident at budget %s; skipping remaining budget levels.",
                            length,
                            budget,
                        )
                    break
                if idx < len(length_budgets) - 1:
                    ranked = _rank_auto_opt_candidates(level_candidates, auto_cfg)
                    current_specs = [
                        AutoOptSpec(
                            kind=c.kind,
                            length=int(c.length) if c.length is not None else length,
                            move_profile=c.move_profile,
                            cooling_boost=c.cooling_boost,
                            swap_prob=c.swap_prob,
                            ladder_size=c.ladder_size,
                            move_probs=c.move_probs,
                            move_probs_label=c.move_probs_label,
                        )
                        for c in ranked
                    ]
                else:
                    length_final_candidates = level_candidates
            runtime_sec = time.perf_counter() - start_time
            if not length_final_candidates:
                raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")
            ranked_length = _rank_auto_opt_candidates(length_final_candidates, auto_cfg)
            best_for_length = ranked_length[0]
            previous_best_run = best_for_length.run_dir
            length_summary_rows.append(
                {
                    "length": length,
                    "best_score": best_for_length.top_k_median_final,
                    "balance": best_for_length.balance_median,
                    "diversity": best_for_length.diversity,
                    "unique_fraction": best_for_length.unique_fraction,
                    "runtime_sec": round(runtime_sec, 3),
                }
            )
            final_level_candidates_all.extend(length_final_candidates)
        final_level_candidates = final_level_candidates_all
    else:
        current_specs = list(candidate_specs)
        for idx, budget in enumerate(budgets):
            level_candidates = []
            for spec in current_specs:
                label = _spec_label(spec, idx=idx + 1, budget=budget)
                candidate = _run_candidate(
                    kind=spec.kind,
                    length=spec.length,
                    budget=budget,
                    label=label,
                    cooling_boost=spec.cooling_boost,
                    move_profile=spec.move_profile,
                    swap_prob=spec.swap_prob,
                    ladder_size=spec.ladder_size,
                    move_probs=spec.move_probs,
                    move_probs_label=spec.move_probs_label,
                )
                candidate_notes = _assess_candidate_quality(candidate, auto_cfg, mode="auto_opt")
                if candidate_notes:
                    candidate.warnings.extend(candidate_notes)
                _log_candidate(candidate, label=label)
                level_candidates.append(candidate)
                all_candidates.append(candidate)

            confidence_level, _, _ = _confidence_from_candidates(level_candidates, auto_cfg)
            if confidence_level == "high":
                final_level_candidates = level_candidates
                if idx < len(budgets) - 1:
                    logger.info(
                        "Auto-opt: confident at budget %s; skipping remaining budget levels.",
                        budget,
                    )
                break
            if idx < len(budgets) - 1:
                ranked = _rank_auto_opt_candidates(level_candidates, auto_cfg)
                current_specs = [
                    AutoOptSpec(
                        kind=c.kind,
                        length=int(c.length) if c.length is not None else lengths[0],
                        move_profile=c.move_profile,
                        cooling_boost=c.cooling_boost,
                        swap_prob=c.swap_prob,
                        ladder_size=c.ladder_size,
                        move_probs=c.move_probs,
                        move_probs_label=c.move_probs_label,
                    )
                    for c in ranked
                ]
            else:
                final_level_candidates = level_candidates

    if not final_level_candidates:
        raise ValueError("Auto-optimize failed: no final-level candidates were evaluated.")

    decision_notes: list[str] = []
    viable, _ = _validate_auto_opt_candidates(
        final_level_candidates,
        allow_warn=auto_cfg.policy.allow_warn,
    )

    confidence_level, _best_candidate, _second_candidate = _confidence_from_candidates(viable, auto_cfg)
    selection_confident = confidence_level == "high"
    decision_notes.append(f"confidence={confidence_level}")
    if not selection_confident and auto_cfg.policy.allow_warn:
        logger.warning(
            "Auto-optimize: could not confidently separate top candidates at max budget; "
            "selecting best available candidate."
        )
    elif not selection_confident:
        raise ValueError(
            "Auto-optimize failed: could not confidently separate top candidates at the maximum "
            "configured budgets/replicates. Increase auto_opt.budget_levels and/or auto_opt.replicates."
        )

    winner = _select_auto_opt_candidate(viable, auto_cfg, allow_fail=False)

    if winner.quality == "ok":
        logger.info(
            (
                "Auto-optimize selected %s L=%s move=%s beta=%s swap=%s ladder=%s (top_k_median=%s, best=%s, "
                "balance=%s, rhat=%s, ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.move_profile,
            f"{winner.cooling_boost:g}",
            f"{winner.swap_prob:g}" if winner.swap_prob is not None else "n/a",
            str(winner.ladder_size) if winner.ladder_size is not None else "n/a",
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    else:
        logger.warning(
            (
                "Auto-optimize selected %s L=%s move=%s beta=%s swap=%s ladder=%s "
                "(quality=%s top_k_median=%s, best=%s, balance=%s, rhat=%s, "
                "ess=%s, unique_fraction=%s)."
            ),
            winner.kind,
            winner.length if winner.length is not None else "n/a",
            winner.move_profile,
            f"{winner.cooling_boost:g}",
            f"{winner.swap_prob:g}" if winner.swap_prob is not None else "n/a",
            str(winner.ladder_size) if winner.ladder_size is not None else "n/a",
            winner.quality,
            f"{winner.top_k_median_final:.3f}" if winner.top_k_median_final is not None else "n/a",
            f"{winner.best_score_final:.3f}" if winner.best_score_final is not None else "n/a",
            f"{winner.balance_median:.3f}" if winner.balance_median is not None else "n/a",
            f"{winner.rhat:.3f}" if winner.rhat is not None else "n/a",
            f"{winner.ess:.1f}" if winner.ess is not None else "n/a",
            f"{winner.unique_fraction:.2f}" if winner.unique_fraction is not None else "n/a",
        )
    if winner.cooling_boost > 1:
        decision_notes.append(f"cooling_boost={winner.cooling_boost:g}")
    if winner.move_profile != sample_cfg.moves.profile:
        decision_notes.append(f"move_profile={winner.move_profile}")
    if winner.move_probs is not None:
        decision_notes.append(f"move_probs={_format_move_probs(winner.move_probs)}")
    if winner.kind == "pt" and winner.swap_prob is not None:
        if winner.swap_prob != sample_cfg.optimizers.pt.swap_prob:
            decision_notes.append(f"swap_prob={winner.swap_prob:g}")
        if winner.ladder_size is not None:
            ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
            if winner.ladder_size != len(ladder):
                decision_notes.append(f"ladder_size={winner.ladder_size}")
    if decision_notes:
        logger.warning("Auto-opt notes: %s", ", ".join(decision_notes))

    final_cfg, notes = _build_final_sample_cfg(
        sample_cfg,
        kind=winner.kind,
        length_override=winner.length,
        cooling_boost=winner.cooling_boost,
        move_profile=winner.move_profile,
        swap_prob=winner.swap_prob,
        ladder_size=winner.ladder_size,
        move_probs=winner.move_probs,
    )
    if notes:
        logger.info("Auto-opt final overrides: %s", "; ".join(notes))
    length_aware = auto_cfg.length.enabled and len({c.length for c in all_candidates if c.length is not None}) > 1
    ranking_label = (
        "top_k_median -> best_score_final -> balance -> diversity -> improvement"
        if length_aware
        else "top_k_median -> best_score_final -> balance -> diversity -> improvement"
    )
    logger.info("Auto-opt selection ranking: %s.", ranking_label)
    logger.info("Auto-opt final config: %s", _format_auto_opt_config_summary(final_cfg))
    pilot_root = all_candidates[0].run_dir.parent if all_candidates else None
    best_marker_path = _auto_opt_best_marker_path(pilot_root, run_group=run_group) if pilot_root is not None else None
    length_ladder_path: Path | None = None
    if length_cfg.mode == "ladder" and length_summary_rows and pilot_root is not None:
        length_ladder_path = _write_length_ladder_table(length_summary_rows, pilot_root=pilot_root)
        logger.info(
            "Length ladder summary -> %s",
            _format_run_path(length_ladder_path, base=config_path.parent),
        )
    if pilot_root is not None:
        logger.info(
            "Auto-opt details: pilot manifests under %s; best marker -> %s; final config in config_used.yaml.",
            _format_run_path(pilot_root, base=config_path.parent),
            _format_run_path(best_marker_path, base=config_path.parent) if best_marker_path else "n/a",
        )

    decision_payload = {
        "mode": "final",
        "selected": winner.kind,
        "selected_length": winner.length,
        "selected_move_profile": winner.move_profile,
        "selected_cooling_boost": winner.cooling_boost,
        "selected_beta_scale": winner.cooling_boost,
        "selected_swap_prob": winner.swap_prob,
        "selected_ladder_size": winner.ladder_size,
        "selected_move_probs": winner.move_probs,
        "selected_move_probs_label": winner.move_probs_label,
        "selection_quality": winner.quality,
        "selection_confident": selection_confident,
        "selection_confidence": confidence_level,
        "notes": decision_notes,
        "config_summary": _format_auto_opt_config_summary(final_cfg),
        "selected_config": _selected_config_payload(final_cfg),
        "selection_metrics": {
            "top_k_median_final": winner.top_k_median_final,
            "best_score_final": winner.best_score_final,
            "top_k_ci_low": winner.top_k_ci_low,
            "top_k_ci_high": winner.top_k_ci_high,
        },
        "length_config": auto_cfg.length.model_dump(),
        "length_ladder_table": str(length_ladder_path) if length_ladder_path is not None else None,
        "auto_opt_config": {
            "budget_levels": auto_cfg.budget_levels,
            "replicates": auto_cfg.replicates,
            "keep_pilots": auto_cfg.keep_pilots,
            "prefer_simpler_if_close": auto_cfg.prefer_simpler_if_close,
            "tolerance": auto_cfg.tolerance.model_dump(),
            "scorecard_top_k": auto_cfg.policy.scorecard.top_k,
            "move_profiles": auto_cfg.move_profiles,
            "pt_swap_probs": auto_cfg.pt_swap_probs,
            "pt_ladder_sizes": auto_cfg.pt_ladder_sizes,
            "gibbs_move_probs": auto_cfg.gibbs_move_probs,
        },
        "pilot_root": str(pilot_root) if pilot_root is not None else None,
        "best_marker": str(best_marker_path) if best_marker_path is not None else None,
        "candidates": [
            {
                "kind": candidate.kind,
                "length": candidate.length,
                "budget": candidate.budget,
                "cooling_boost": candidate.cooling_boost,
                "move_profile": candidate.move_profile,
                "swap_prob": candidate.swap_prob,
                "ladder_size": candidate.ladder_size,
                "move_probs_label": candidate.move_probs_label,
                "runs": [path.name for path in candidate.run_dirs],
                "status": candidate.status,
                "quality": candidate.quality,
                "best_score": candidate.best_score,
                "top_k_median_final": candidate.top_k_median_final,
                "best_score_final": candidate.best_score_final,
                "top_k_ci_low": candidate.top_k_ci_low,
                "top_k_ci_high": candidate.top_k_ci_high,
                "balance_median": candidate.balance_median,
                "diversity": candidate.diversity,
                "improvement": candidate.improvement,
                "acceptance_b": candidate.acceptance_b,
                "acceptance_m": candidate.acceptance_m,
                "acceptance_mh": candidate.acceptance_mh,
                "swap_rate": candidate.swap_rate,
                "rhat": candidate.rhat,
                "ess": candidate.ess,
                "unique_fraction": candidate.unique_fraction,
            }
            for candidate in all_candidates
        ],
    }
    final_run_dir = _run_sample_for_set(
        cfg,
        config_path,
        set_index=set_index,
        set_count=set_count,
        include_set_index=include_set_index,
        tfs=tfs,
        lockmap=lockmap,
        sample_cfg=final_cfg,
        stage="sample",
        run_kind="auto_opt_final",
        auto_opt_meta=decision_payload,
    )
    if pilot_root is not None:
        marker_payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "selected_candidate": {
                "kind": winner.kind,
                "length": winner.length,
                "cooling_boost": winner.cooling_boost,
                "move_profile": winner.move_profile,
                "swap_prob": winner.swap_prob,
                "ladder_size": winner.ladder_size,
                "move_probs_label": winner.move_probs_label,
                "pilot_run": winner.run_dir.name,
            },
            "final_sample_run": final_run_dir.name,
            "config_summary": _format_auto_opt_config_summary(final_cfg),
        }
        marker_path = _write_auto_opt_best_marker(pilot_root, marker_payload, run_group=run_group)
        logger.info(
            "Auto-opt best marker updated -> %s",
            _format_run_path(marker_path, base=config_path.parent),
        )
        removed = _prune_auto_opt_runs(
            config_path=config_path,
            pilot_root=pilot_root,
            candidates=all_candidates,
            winner=winner,
            keep_mode=auto_cfg.keep_pilots,
            catalog_root=cfg.motif_store.catalog_root,
        )
        if removed:
            logger.info(
                "Auto-opt pruned %d pilot run(s) (keep_pilots=%s).",
                len(removed),
                auto_cfg.keep_pilots,
            )
    logger.info(
        "Auto-opt final config saved -> %s",
        _format_run_path(config_used_path(final_run_dir), base=config_path.parent),
    )
    return final_run_dir


def _evaluate_pilot_run(
    run_dir: Path,
    kind: str,
    *,
    scorecard_top_k: int | None = None,
    budget: int | None = None,
    mode: str | None = None,
    cooling_boost: float = 1.0,
    move_profile: str = "balanced",
    swap_prob: float | None = None,
    ladder_size: int | None = None,
    move_probs: dict[str, float] | None = None,
    move_probs_label: str | None = None,
) -> AutoOptCandidate:
    manifest = load_manifest(run_dir)
    length = manifest.get("sequence_length")
    trace_file = trace_path(run_dir)
    seq_path = sequences_path(run_dir)
    if not trace_file.exists():
        raise FileNotFoundError(f"Missing trace.nc in {run_dir}")
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences.parquet in {run_dir}")
    elite_path = find_elites_parquet(run_dir)
    seq_df = read_parquet(seq_path)
    elites_df = read_parquet(elite_path)
    tf_names = [m["tf_name"] for m in manifest.get("motifs", [])]
    top_k = scorecard_top_k
    if top_k is None:
        top_k = int(manifest.get("top_k") or 10)
    draw_scores = _draw_scores_from_sequences(seq_df)
    top_k_median_final = _top_k_median_from_scores(draw_scores, top_k)
    best_score_final = float(np.max(draw_scores)) if draw_scores.size else None
    best_score = top_k_median_final
    bootstrap_ci: tuple[float, float] | None = None
    if draw_scores.size:
        seed = _bootstrap_seed(manifest=manifest, run_dir=run_dir, kind=kind)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(draw_scores, k=top_k, rng=rng)

    import arviz as az

    trace_idata = az.from_netcdf(trace_file)
    sample_meta = None
    if scorecard_top_k is not None:
        sample_meta = {"top_k": scorecard_top_k}
    elif "top_k" in manifest:
        sample_meta = {"top_k": manifest.get("top_k")}
    if sample_meta is None:
        sample_meta = {}
    if mode is not None:
        sample_meta["mode"] = mode
    sample_meta["optimizer_kind"] = kind
    elites_cfg = manifest.get("elites")
    if isinstance(elites_cfg, dict):
        sample_meta["dsdna_canonicalize"] = elites_cfg.get("dsDNA_canonicalize", False)
        sample_meta["dsdna_hamming"] = elites_cfg.get("dsDNA_hamming", False)
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=seq_df,
        elites_df=elites_df,
        tf_names=tf_names,
        optimizer=manifest.get("optimizer", {}),
        optimizer_stats=manifest.get("optimizer_stats", {}),
        sample_meta=sample_meta,
    )
    metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, dict) else {}
    trace_metrics = metrics.get("trace", {})
    seq_metrics = metrics.get("sequences", {})
    elites_metrics = metrics.get("elites", {})
    optimizer_metrics = metrics.get("optimizer", {})
    rhat = trace_metrics.get("rhat")
    ess = trace_metrics.get("ess")
    unique_fraction = seq_metrics.get("unique_fraction")
    balance_median = elites_metrics.get("normalized_min_median") or elites_metrics.get("balance_index_median")
    diversity = elites_metrics.get("diversity_hamming")
    improvement = trace_metrics.get("best_so_far_slope") or trace_metrics.get("score_delta")
    acceptance_b = None
    acceptance_m = None
    acceptance_mh = None
    delta_frac_zero_mh = None
    if isinstance(optimizer_metrics, dict):
        acc = optimizer_metrics.get("acceptance_rate") or {}
        if isinstance(acc, dict):
            acceptance_b = acc.get("B")
            acceptance_m = acc.get("M")
        acceptance_mh = optimizer_metrics.get("acceptance_rate_mh")
        delta_frac_zero_mh = optimizer_metrics.get("delta_frac_zero_mh")
    swap_rate = optimizer_metrics.get("swap_acceptance_rate") if isinstance(optimizer_metrics, dict) else None
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []
    status = "ok"
    if isinstance(diagnostics, dict):
        diag_status = diagnostics.get("status", "ok")
        if diag_status and diag_status != "ok":
            warnings = list(warnings)
            warnings.append(f"Diagnostics status={diag_status}; see diagnostics.json for details.")

    if draw_scores.size == 0:
        warnings = list(warnings)
        warnings.append("Missing pilot metric: combined_score_final draw-phase values.")
        status = "fail"

    unique_draws: int | None = None
    if "sequence" in seq_df.columns:
        df_unique = seq_df
        if "phase" in df_unique.columns:
            df_unique = df_unique[df_unique["phase"] == "draw"]
        try:
            unique_draws = int(df_unique["sequence"].nunique())
        except Exception:
            unique_draws = None

    if kind == "gibbs" and status != "fail":
        moved = unique_draws is not None and unique_draws > 1
        accepted = acceptance_mh is not None and acceptance_mh > 0
        if not accepted and not moved:
            warnings = list(warnings)
            warnings.append("Gibbs pilot shows no accepted MH moves and no unique sequences.")
            status = "fail"

    if kind == "pt" and isinstance(optimizer_metrics, dict):
        swap_attempts = optimizer_metrics.get("swap_attempts")
        swap_prob = None
        optimizer_cfg = manifest.get("optimizer", {})
        if isinstance(optimizer_cfg, dict):
            pt_cfg = optimizer_cfg.get("pt", {})
            if isinstance(pt_cfg, dict):
                swap_prob = pt_cfg.get("swap_prob")
        if swap_prob and swap_attempts == 0:
            warnings = list(warnings)
            warnings.append("PT pilot recorded swap_prob>0 but swap_attempts=0.")

    if acceptance_mh is not None and acceptance_mh > 0.95 and delta_frac_zero_mh is not None:
        warnings = list(warnings)
        if delta_frac_zero_mh >= 0.8:
            warnings.append(
                "High MH acceptance likely due to hard-min plateaus (Δscore≈0). "
                "Enable objective.softmin and/or increase target_worst_tf_prob."
            )
        elif delta_frac_zero_mh < 0.5:
            warnings.append(
                "High MH acceptance with nontrivial Δscore suggests beta too low or moves too conservative. "
                "Increase beta schedule or move sizes."
            )

    return AutoOptCandidate(
        kind=kind,
        length=int(length) if isinstance(length, (int, float)) else None,
        budget=budget,
        cooling_boost=cooling_boost,
        move_profile=move_profile,
        swap_prob=swap_prob,
        ladder_size=ladder_size,
        move_probs=move_probs,
        move_probs_label=move_probs_label,
        run_dir=run_dir,
        run_dirs=[run_dir],
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=rhat,
        ess=ess,
        unique_fraction=unique_fraction,
        balance_median=balance_median,
        diversity=diversity,
        improvement=improvement,
        acceptance_b=acceptance_b,
        acceptance_m=acceptance_m,
        acceptance_mh=acceptance_mh,
        swap_rate=swap_rate,
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
    )


def _aggregate_candidate_runs(
    runs: list[AutoOptCandidate],
    *,
    budget: int,
    scorecard_top_k: int,
) -> AutoOptCandidate:
    if not runs:
        raise ValueError("Auto-opt aggregation requires at least one pilot run")

    def _median(values: list[float | None]) -> float | None:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return float(np.median(filtered))

    kind = runs[0].kind
    length = runs[0].length
    move_profile = runs[0].move_profile
    swap_prob = runs[0].swap_prob
    ladder_size = runs[0].ladder_size
    move_probs = runs[0].move_probs
    move_probs_label = runs[0].move_probs_label
    run_dirs = [run.run_dir for run in runs]
    status = "ok"
    if all(run.status == "fail" for run in runs):
        status = "fail"

    warnings: list[str] = []
    for run in runs:
        warnings.extend(run.warnings)
    if status != "ok":
        warnings.append("One or more pilot replicates failed")

    diagnostics = {
        "replicates": [run.diagnostics for run in runs],
        "run_dirs": [str(path) for path in run_dirs],
    }
    best_run = max(
        runs,
        key=lambda run: (
            float(run.top_k_median_final) if run.top_k_median_final is not None else float("-inf"),
            float(run.best_score_final) if run.best_score_final is not None else float("-inf"),
            float(run.balance_median) if run.balance_median is not None else float("-inf"),
        ),
    )

    pooled_scores: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for run_dir in run_dirs:
        seq_path = sequences_path(run_dir)
        if not seq_path.exists():
            warnings.append(f"Missing sequences.parquet for replicate {run_dir.name}")
            continue
        try:
            seq_df = read_parquet(seq_path)
        except Exception as exc:
            warnings.append(f"Failed to read sequences.parquet for {run_dir.name}: {exc}")
            continue
        scores = _draw_scores_from_sequences(seq_df)
        if scores.size:
            pooled_scores.append(scores)
        try:
            manifest = load_manifest(run_dir)
        except Exception as exc:
            warnings.append(f"Failed to read manifest for {run_dir.name}: {exc}")
        else:
            manifests.append(manifest)

    combined_scores = np.concatenate(pooled_scores) if pooled_scores else np.array([], dtype=float)
    if combined_scores.size:
        top_k_median_final = _top_k_median_from_scores(combined_scores, scorecard_top_k)
        best_score_final = float(np.max(combined_scores))
        best_score = top_k_median_final
    else:
        top_k_median_final = _median([run.top_k_median_final for run in runs])
        best_score_final = max(
            [float(run.best_score_final) for run in runs if run.best_score_final is not None],
            default=None,
        )
        best_score = top_k_median_final
    bootstrap_ci: tuple[float, float] | None = None
    if combined_scores.size:
        seed = _pooled_bootstrap_seed(manifests=manifests, kind=kind, length=length, budget=budget)
        if seed is None:
            payload = {"kind": kind, "length": length, "budget": budget, "replicate_count": len(run_dirs)}
            seed = _bootstrap_seed_payload(payload)
        rng = np.random.default_rng(seed)
        bootstrap_ci = _bootstrap_top_k_ci(combined_scores, k=scorecard_top_k, rng=rng)
    else:
        lows = [run.top_k_ci_low for run in runs if run.top_k_ci_low is not None]
        highs = [run.top_k_ci_high for run in runs if run.top_k_ci_high is not None]
        if lows and highs:
            bootstrap_ci = (float(min(lows)), float(max(highs)))

    return AutoOptCandidate(
        kind=kind,
        length=length,
        budget=budget,
        cooling_boost=float(np.median([run.cooling_boost for run in runs])),
        move_profile=move_profile,
        swap_prob=swap_prob,
        ladder_size=ladder_size,
        move_probs=move_probs,
        move_probs_label=move_probs_label,
        run_dir=best_run.run_dir,
        run_dirs=run_dirs,
        best_score=best_score,
        top_k_median_final=top_k_median_final,
        best_score_final=best_score_final,
        top_k_ci_low=bootstrap_ci[0] if bootstrap_ci else None,
        top_k_ci_high=bootstrap_ci[1] if bootstrap_ci else None,
        rhat=_median([run.rhat for run in runs]),
        ess=_median([run.ess for run in runs]),
        unique_fraction=_median([run.unique_fraction for run in runs]),
        balance_median=_median([run.balance_median for run in runs]),
        diversity=_median([run.diversity for run in runs]),
        improvement=_median([run.improvement for run in runs]),
        acceptance_b=_median([run.acceptance_b for run in runs]),
        acceptance_m=_median([run.acceptance_m for run in runs]),
        acceptance_mh=_median([run.acceptance_mh for run in runs]),
        swap_rate=_median([run.swap_rate for run in runs]),
        status=status,
        quality=status,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _rank_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
) -> list[AutoOptCandidate]:
    ranked: list[tuple[tuple[float, float, float, float, float, float, float], AutoOptCandidate]] = []
    status_rank = {"ok": 2, "warn": 1, "fail": 0}
    for candidate in candidates:
        score = candidate.top_k_median_final if candidate.top_k_median_final is not None else candidate.best_score
        if score is None:
            score = float("-inf")
        secondary = (
            candidate.best_score_final
            if candidate.best_score_final is not None
            else (candidate.best_score if candidate.best_score is not None else float("-inf"))
        )
        balance = candidate.balance_median if candidate.balance_median is not None else float("-inf")
        diversity = candidate.diversity if candidate.diversity is not None else float("-inf")
        improvement = candidate.improvement if candidate.improvement is not None else float("-inf")
        unique_fraction = candidate.unique_fraction if candidate.unique_fraction is not None else float("-inf")
        rank = (
            float(score),
            float(secondary),
            float(balance),
            float(diversity),
            float(improvement),
            float(unique_fraction),
            float(status_rank.get(candidate.quality, status_rank.get(candidate.status, 0))),
        )
        ranked.append((rank, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _confidence_from_candidates(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
) -> tuple[str, AutoOptCandidate | None, AutoOptCandidate | None]:
    if not candidates:
        return "low", None, None
    ranked = _rank_auto_opt_candidates(candidates, auto_cfg)
    best = ranked[0]
    if len(ranked) == 1:
        return "high" if best.quality == "ok" else "medium", best, None
    second = ranked[1]
    if best.top_k_ci_low is None or second.top_k_ci_high is None:
        ci_separated = False
    else:
        ci_separated = best.top_k_ci_low > second.top_k_ci_high
    confidence_level = "high" if ci_separated else "low"
    if best.quality == "warn" and confidence_level == "high":
        confidence_level = "medium"
    if best.quality != "ok" and confidence_level == "high":
        confidence_level = "medium"
    return confidence_level, best, second


def _validate_auto_opt_candidates(
    candidates: list[AutoOptCandidate],
    *,
    allow_warn: bool,
) -> tuple[list[AutoOptCandidate], bool]:
    _ = allow_warn
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates.")
    viable = [c for c in candidates if c.status != "fail"]
    if not viable:
        raise ValueError(
            "Auto-optimize failed: all pilot candidates failed catastrophic checks "
            "(missing scores or no movement). Re-run with larger budgets or fix the model inputs."
        )
    return viable, False


def _select_auto_opt_candidate(
    candidates: list[AutoOptCandidate],
    auto_cfg: AutoOptConfig,
    *,
    allow_fail: bool = False,
) -> AutoOptCandidate:
    if not candidates:
        raise ValueError("Auto-optimize did not produce any pilot candidates")

    ok_candidates = [c for c in candidates if c.status != "fail"]
    if not ok_candidates and allow_fail:
        ok_candidates = list(candidates)
    if auto_cfg.length.enabled and auto_cfg.length.prefer_shortest:
        lengths = [c.length for c in ok_candidates if c.length is not None]
        if lengths:
            min_len = min(lengths)
            ok_candidates = [c for c in ok_candidates if c.length == min_len]

    ranked = _rank_auto_opt_candidates(ok_candidates, auto_cfg)
    winner = ranked[0]
    if winner.status == "fail" and not allow_fail:
        raise ValueError("Auto-optimize failed: all pilot candidates reported missing diagnostics.")

    if auto_cfg.prefer_simpler_if_close and winner.kind == "pt":
        gibbs = [c for c in ranked if c.kind == "gibbs"]
        if gibbs and winner.top_k_median_final is not None and gibbs[0].top_k_median_final is not None:
            if gibbs[0].top_k_median_final >= winner.top_k_median_final - auto_cfg.tolerance.score:
                return gibbs[0]
    return winner
