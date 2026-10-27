"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/sample/preflight.py

Validate sample-run preconditions and derive scheduling metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np

from dnadesign.cruncher.config.schema_v3 import SampleConfig
from dnadesign.cruncher.core.objectives.capabilities import resolve_score_scale_capabilities
from dnadesign.cruncher.core.objectives.compiler import ObjectivePlanCompilation, compile_objective_plan
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.utils.hashing import sha256_bytes


class ConfigError(ValueError):
    """Raised when run configuration is internally contradictory."""


class RunError(RuntimeError):
    """Raised when run-time state violates strict sampling contracts."""


def _assert_sequence_buffers_aligned(optimizer: object) -> tuple[list[object], list[object], list[object]]:
    required_buffers = ("all_meta", "all_samples", "all_scores")
    resolved: dict[str, list[object]] = {}
    for name in required_buffers:
        value = getattr(optimizer, name, None)
        if not isinstance(value, list):
            raise RunError(f"Optimizer missing required sequence buffer '{name}' while saving sequences.")
        resolved[name] = value
    lengths = {name: len(values) for name, values in resolved.items()}
    if len(set(lengths.values())) != 1:
        raise RunError(
            "Optimizer sequence buffers have inconsistent lengths: "
            + ", ".join(f"{name}={lengths[name]}" for name in required_buffers)
        )
    return resolved["all_meta"], resolved["all_samples"], resolved["all_scores"]


def _assert_trace_meta_aligned(optimizer: object, *, expected_rows: int) -> list[dict[str, object]]:
    value = getattr(optimizer, "all_trace_meta", None)
    if not isinstance(value, list):
        raise RunError("Optimizer missing required sequence buffer 'all_trace_meta' while saving sequences.")
    if len(value) != expected_rows:
        raise RunError(
            f"Optimizer sequence trace metadata length mismatch: all_trace_meta={len(value)} expected={expected_rows}"
        )
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise RunError(f"Optimizer trace metadata row {idx} must be a dictionary.")
        rows.append(item)
    return rows


def _validate_objective_preflight(sample_cfg: SampleConfig, *, n_tfs: int) -> None:
    scale = sample_cfg.objective.score_scale
    combine_cfg = sample_cfg.objective.combine
    softmin_enabled = bool(sample_cfg.objective.softmin.enabled)
    multiplicity_enabled = bool(sample_cfg.objective.multiplicity.enabled)

    if n_tfs >= 2 and scale == "llr":
        raise ConfigError(
            "cruncher.sample.objective.score_scale='llr' is not allowed for multi-TF runs. "
            "Multi-TF runs require comparable scales. Use normalized-llr, logp, z, or consensus-neglop-sum."
        )

    if combine_cfg == "sum" and softmin_enabled:
        raise ConfigError(
            "cruncher.sample.objective.softmin.enabled=true is incompatible with "
            "cruncher.sample.objective.combine='sum'. Disable softmin or switch combine to "
            "min/softmin-compatible mode."
        )

    if scale == "consensus-neglop-sum" and softmin_enabled:
        raise ConfigError(
            "cruncher.sample.objective.softmin.enabled=true is incompatible with "
            "cruncher.sample.objective.score_scale='consensus-neglop-sum'. Disable softmin or switch combine to "
            "min/softmin-compatible mode."
        )

    if multiplicity_enabled and n_tfs != 1:
        raise ConfigError(
            "cruncher.sample.objective.multiplicity.enabled=true is only supported for runs with exactly one TF."
        )


def prepare_objective_plan(
    *,
    sample_cfg: SampleConfig,
    tfs: list[str],
    pwms: dict[str, PWM],
) -> ObjectivePlanCompilation:
    _validate_objective_preflight(sample_cfg, n_tfs=len(tfs))
    multiplicity_cfg = sample_cfg.objective.multiplicity
    if multiplicity_cfg.enabled:
        scale_caps = resolve_score_scale_capabilities(sample_cfg.objective.score_scale)
        if not scale_caps.occurrence_aggregation_safe:
            raise ConfigError(
                "cruncher.sample.objective.multiplicity requires an occurrence-safe score scale. "
                "Use llr, normalized-llr, or z."
            )
        if float(sample_cfg.elites.select.diversity) > 0.0:
            raise ConfigError(
                "cruncher.sample.elites.select.diversity is unsupported when multiplicity scoring is enabled."
            )
        if bool(sample_cfg.elites.postprocess.trim_uncovered_internal):
            raise ConfigError(
                "cruncher.sample.elites.postprocess.trim_uncovered_internal is unsupported when multiplicity "
                "scoring is enabled."
            )
        if not tfs:
            raise ConfigError("Multiplicity scoring requires exactly one active TF.")
        tf = tfs[0]
        pwm = pwms.get(tf)
        if pwm is None:
            raise ConfigError(f"Missing PWM for multiplicity objective TF '{tf}'.")
        copies = int(multiplicity_cfg.copies)
        min_gap = int(multiplicity_cfg.distinctness.min_gap)
        distinctness_mode = str(multiplicity_cfg.distinctness.mode)
        if distinctness_mode == "interval":
            minimum_span = int(pwm.length) * copies + max(0, copies - 1) * min_gap
        elif distinctness_mode == "offset":
            minimum_span = int(pwm.length) + max(0, copies - 1) * (min_gap + 1)
        else:
            raise ConfigError(
                f"Unsupported cruncher.sample.objective.multiplicity.distinctness.mode value '{distinctness_mode}'."
            )
        if minimum_span > int(sample_cfg.sequence_length):
            raise ConfigError(
                "Requested multiplicity objective is infeasible for the configured sequence length. "
                f"Need at least {minimum_span} bp for copies={copies}, motif_width={pwm.length}, "
                f"min_gap={min_gap}, distinctness_mode={distinctness_mode}."
            )
    try:
        return compile_objective_plan(sample_cfg=sample_cfg, tfs=tfs, pwms=pwms)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _softmin_schedule_payload(sample_cfg: SampleConfig) -> dict[str, object]:
    softmin_cfg = sample_cfg.objective.softmin
    if softmin_cfg.schedule == "fixed":
        return {"kind": "fixed", "beta": float(softmin_cfg.beta_end)}
    if softmin_cfg.schedule == "linear":
        return {"kind": "linear", "beta": (float(softmin_cfg.beta_start), float(softmin_cfg.beta_end))}
    raise ConfigError(
        "Unsupported cruncher.sample.objective.softmin.schedule value "
        f"'{softmin_cfg.schedule}'. Expected 'fixed' or 'linear'."
    )


def _mcmc_cooling_payload(sample_cfg: SampleConfig) -> dict[str, object]:
    cooling_cfg = sample_cfg.optimizer.cooling
    if cooling_cfg.kind == "fixed":
        if cooling_cfg.beta is None:
            raise ConfigError("sample.optimizer.cooling.beta is required when kind='fixed'.")
        return {"kind": "fixed", "beta": float(cooling_cfg.beta)}
    if cooling_cfg.kind == "linear":
        if cooling_cfg.beta_start is None or cooling_cfg.beta_end is None:
            raise ConfigError("sample.optimizer.cooling.beta_start and beta_end are required when kind='linear'.")
        return {
            "kind": "linear",
            "beta_start": float(cooling_cfg.beta_start),
            "beta_end": float(cooling_cfg.beta_end),
        }
    if cooling_cfg.kind == "piecewise":
        stages = [{"sweeps": int(stage.sweeps), "beta": float(stage.beta)} for stage in cooling_cfg.stages]
        return {"kind": "piecewise", "stages": stages}
    raise ConfigError(f"Unsupported sample.optimizer.cooling.kind: {cooling_cfg.kind!r}.")


def _final_executed_sweep_index(optimizer: object) -> int:
    all_meta = getattr(optimizer, "all_meta", None)
    if not isinstance(all_meta, list) or not all_meta:
        raise RunError(
            "Unable to resolve final executed sweep for softmin schedule: optimizer.all_meta is missing or empty."
        )
    sweep_indices: list[int] = []
    for item in all_meta:
        if not isinstance(item, tuple) or len(item) < 2 or not isinstance(item[1], (int, np.integer)):
            raise RunError(
                "Unable to resolve final executed sweep for softmin schedule: invalid optimizer.all_meta row."
            )
        sweep_indices.append(int(item[1]))
    final_idx = max(sweep_indices)
    if final_idx < 0:
        raise RunError("Unable to resolve final executed sweep for softmin schedule: final sweep index is negative.")
    return final_idx


def _resolve_final_softmin_beta(optimizer: object, sample_cfg: SampleConfig) -> float | None:
    softmin_cfg = sample_cfg.objective.softmin
    if not softmin_cfg.enabled:
        return None

    total = int(sample_cfg.budget.tune + sample_cfg.budget.draws)
    if total < 1:
        raise ConfigError(
            "cruncher.sample.budget is invalid for softmin scheduling; tune + draws must be >= 1 when "
            "cruncher.sample.objective.softmin.enabled=true."
        )

    final_sweep_idx = _final_executed_sweep_index(optimizer)
    if final_sweep_idx >= total:
        raise RunError(
            "Unable to resolve final softmin beta: final executed sweep index exceeds configured "
            "cruncher.sample.budget.tune + cruncher.sample.budget.draws."
        )

    schedule = _softmin_schedule_payload(sample_cfg)
    try:
        beta_of = make_beta_scheduler(schedule, total)
        return float(beta_of(final_sweep_idx))
    except Exception as exc:
        raise ConfigError(
            "Failed to evaluate cruncher.sample.objective.softmin schedule for the final executed sweep."
        ) from exc


def _assert_init_length_fits_pwms(sample_cfg: SampleConfig, pwms: dict[str, PWM]) -> None:
    if not pwms:
        raise ValueError("No PWMs available for sampling.")
    max_w = max(pwm.length for pwm in pwms.values())
    if sample_cfg.sequence_length < max_w:
        names = ", ".join(f"{tf}:{pwms[tf].length}" for tf in sorted(pwms))
        raise ValueError(
            f"sequence_length={sample_cfg.sequence_length} is shorter than the widest PWM (max={max_w}). "
            f"Per-TF lengths: {names}. Increase sample.sequence_length or trim PWMs via "
            "cruncher.sample.motif_width.maxw (with strategy=max_info)."
        )


def _resolve_elite_pool_size(
    *,
    pool_size_cfg: str | int,
    elite_k: int,
    candidate_count: int,
) -> int:
    if candidate_count < 0:
        raise ValueError("candidate_count must be >= 0")
    if pool_size_cfg == "all":
        return int(candidate_count)
    if pool_size_cfg == "auto":
        auto_target = max(4000, 500 * max(int(elite_k), 1))
        auto_target = min(auto_target, 20_000)
        return min(int(candidate_count), int(auto_target))
    value = int(pool_size_cfg)
    if value < 1:
        raise ValueError("pool_size must be >= 1")
    return min(value, int(candidate_count))


def _core_def_hash(pwm: PWM, pwm_hash: str | None) -> str:
    payload = {
        "pwm_hash": pwm_hash,
        "pwm_width": pwm.length,
        "source_length": pwm.source_length,
        "window_start": pwm.window_start,
        "window_strategy": pwm.window_strategy,
        "window_score": pwm.window_score,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


__all__ = [
    "ConfigError",
    "RunError",
    "_assert_init_length_fits_pwms",
    "_assert_sequence_buffers_aligned",
    "_assert_trace_meta_aligned",
    "_core_def_hash",
    "_mcmc_cooling_payload",
    "prepare_objective_plan",
    "_resolve_elite_pool_size",
    "_resolve_final_softmin_beta",
    "_softmin_schedule_payload",
    "_validate_objective_preflight",
]
