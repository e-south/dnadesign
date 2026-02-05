"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/metadata.py

Resolve analysis metadata and configuration-derived inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.artifacts.layout import config_used_path
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import (
    AnalysisConfig,
    CruncherConfig,
    SampleMovesConfig,
)
from dnadesign.cruncher.core.pwm import PWM


@dataclass(frozen=True)
class SampleMeta:
    optimizer_kind: str
    chains: int
    draws: int
    tune: int
    move_probs: dict[str, float]
    cooling_kind: str
    pwm_sum_threshold: float
    bidirectional: bool
    top_k: int
    mode: str
    dsdna_canonicalize: bool


def _load_pwms_from_config(run_dir: Path) -> tuple[dict[str, PWM], dict]:
    config_path = config_used_path(run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing meta/config_used.yaml in {run_dir}")
    payload = yaml.safe_load(config_path.read_text()) or {}
    cruncher_cfg = payload.get("cruncher")
    if not isinstance(cruncher_cfg, dict):
        raise ValueError("config_used.yaml missing top-level 'cruncher' section.")
    pwms_info = cruncher_cfg.get("pwms_info")
    if not isinstance(pwms_info, dict) or not pwms_info:
        raise ValueError("config_used.yaml missing pwms_info; re-run `cruncher sample`.")
    pwms: dict[str, PWM] = {}
    for tf_name, info in pwms_info.items():
        matrix = info.get("pwm_matrix")
        if not matrix:
            raise ValueError(f"config_used.yaml missing pwm_matrix for TF '{tf_name}'.")
        pwms[tf_name] = PWM(name=tf_name, matrix=np.array(matrix, dtype=float))
    return pwms, cruncher_cfg


def _resolve_sample_meta(cfg: CruncherConfig, used_cfg: dict) -> SampleMeta:
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")
    used_sample = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
    if not isinstance(used_sample, dict):
        raise ValueError("config_used.yaml missing sample section; re-run `cruncher sample`.")

    def _require(path: list[str], label: str) -> object:
        cursor: object = used_sample
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                raise ValueError(f"config_used.yaml missing sample.{label}; re-run `cruncher sample`.")
            cursor = cursor[key]
        return cursor

    def _coerce_int(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config_used.yaml sample.{label} must be an integer.")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"config_used.yaml sample.{label} must be an integer.")
        return int(value)

    def _coerce_float(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config_used.yaml sample.{label} must be a number.")
        return float(value)

    optimizer_kind = str(_require(["optimizer", "name"], "optimizer.name"))
    draws = _coerce_int(_require(["budget", "draws"], "budget.draws"), "budget.draws")
    tune = _coerce_int(_require(["budget", "tune"], "budget.tune"), "budget.tune")
    mode_val = _require(["mode"], "mode")
    if not isinstance(mode_val, str):
        raise ValueError("config_used.yaml sample.mode must be a string.")
    bidirectional_val = _require(["objective", "bidirectional"], "objective.bidirectional")
    if not isinstance(bidirectional_val, bool):
        raise ValueError("config_used.yaml sample.objective.bidirectional must be a boolean.")
    bidirectional = bidirectional_val
    top_k = _coerce_int(_require(["elites", "k"], "elites.k"), "elites.k")
    pwm_sum_threshold = _coerce_float(
        _require(["elites", "filters", "pwm_sum_min"], "elites.filters.pwm_sum_min"),
        "elites.filters.pwm_sum_min",
    )
    elites_cfg = used_sample.get("elites") if isinstance(used_sample, dict) else None
    if not isinstance(elites_cfg, dict):
        elites_cfg = {}
    selection_cfg = elites_cfg.get("selection") if isinstance(elites_cfg, dict) else None
    policy = "top_score"
    if isinstance(selection_cfg, dict):
        policy_val = selection_cfg.get("policy")
        if policy_val is not None and not isinstance(policy_val, str):
            raise ValueError("config_used.yaml sample.elites.selection.policy must be a string.")
        if policy_val:
            policy = policy_val

    if policy != "mmr":
        raise ValueError("config_used.yaml sample.elites.selection.policy must be 'mmr'.")
    dsdna_canonicalize_val = bidirectional

    moves_payload = _require(["moves"], "moves")
    moves_cfg = SampleMovesConfig.model_validate(moves_payload)
    move_probs = resolve_move_config(moves_cfg).move_probs

    if optimizer_kind != "pt":
        raise ValueError("config_used.yaml sample.optimizer.name must be 'pt'.")
    ladder = _require(["optimizers", "pt", "beta_ladder"], "optimizers.pt.beta_ladder")
    if not isinstance(ladder, dict):
        raise ValueError("config_used.yaml sample.optimizers.pt.beta_ladder must be a mapping.")
    cooling_kind = str(ladder.get("kind") or "")
    if cooling_kind == "fixed":
        chains = 1
    else:
        betas = ladder.get("betas")
        n_temps = ladder.get("n_temps")
        if isinstance(betas, list) and betas:
            chains = len(betas)
        elif isinstance(n_temps, int):
            chains = int(n_temps)
        else:
            raise ValueError("config_used.yaml sample.optimizers.pt.beta_ladder must define betas or n_temps.")

    return SampleMeta(
        optimizer_kind=optimizer_kind,
        chains=chains,
        draws=draws,
        tune=tune,
        move_probs=move_probs,
        cooling_kind=cooling_kind,
        pwm_sum_threshold=pwm_sum_threshold,
        bidirectional=bidirectional,
        top_k=top_k,
        mode=mode_val,
        dsdna_canonicalize=dsdna_canonicalize_val,
    )


def _resolve_scoring_params(used_cfg: dict) -> tuple[float, float | None]:
    if not isinstance(used_cfg, dict):
        raise ValueError("config_used.yaml is missing sample config; re-run `cruncher sample`.")
    sample = used_cfg.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("config_used.yaml missing sample section; re-run `cruncher sample`.")
    objective = sample.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("config_used.yaml missing sample.objective; re-run `cruncher sample`.")
    scoring = objective.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("config_used.yaml missing sample.objective.scoring; re-run `cruncher sample`.")
    pseudocounts = scoring.get("pwm_pseudocounts")
    if not isinstance(pseudocounts, (int, float)):
        raise ValueError("config_used.yaml sample.objective.scoring.pwm_pseudocounts must be a number.")
    log_odds_clip = scoring.get("log_odds_clip")
    if log_odds_clip is not None and not isinstance(log_odds_clip, (int, float)):
        raise ValueError("config_used.yaml sample.objective.scoring.log_odds_clip must be a number or null.")
    return float(pseudocounts), float(log_odds_clip) if log_odds_clip is not None else None


def _analysis_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}_{suffix}"


def _get_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("dnadesign")
    except Exception:
        return None


def _resolve_git_dir(path: Path) -> Path | None:
    git_path = path / ".git"
    if not git_path.exists():
        return None
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        try:
            payload = git_path.read_text().strip()
        except OSError:
            return None
        if payload.startswith("gitdir:"):
            git_dir = payload.split(":", 1)[1].strip()
            resolved = Path(git_dir)
            if not resolved.is_absolute():
                resolved = (git_path.parent / resolved).resolve()
            if resolved.exists():
                return resolved
    return None


def _get_git_commit(path: Path) -> str | None:
    probe = path.resolve()
    for _ in range(6):
        git_dir = _resolve_git_dir(probe)
        if git_dir is not None:
            try:
                head = (git_dir / "HEAD").read_text().strip()
            except OSError:
                return None
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_path = git_dir / ref
                if ref_path.exists():
                    try:
                        return ref_path.read_text().strip()
                    except OSError:
                        return None
            return head or None
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def _resolve_tf_pair(
    analysis_cfg: AnalysisConfig,
    tf_names: list[str],
    tf_pair_override: tuple[str, str] | None = None,
) -> tuple[str, str] | None:
    pair = tf_pair_override
    if pair is None:
        pair = analysis_cfg.tf_pair
    if pair is None:
        return None
    if len(pair) != 2:
        raise ValueError("analysis.tf_pair must contain exactly two TF names.")
    x_tf, y_tf = pair
    if x_tf not in tf_names or y_tf not in tf_names:
        raise ValueError(f"analysis.tf_pair must reference TFs in {tf_names}.")
    return x_tf, y_tf


def _auto_select_tf_pair(
    score_df: pd.DataFrame,
    tf_names: list[str],
) -> tuple[tuple[str, str] | None, str | None]:
    if len(tf_names) < 2 or score_df.empty:
        return None, "fewer than two TFs or empty score table"
    cols = [f"score_{tf}" for tf in tf_names if f"score_{tf}" in score_df.columns]
    if len(cols) < 2:
        return None, "missing per-TF score columns"
    medians = {tf: float(score_df[f"score_{tf}"].median()) for tf in tf_names if f"score_{tf}" in score_df.columns}
    if len(medians) < 2:
        return None, "insufficient score medians"
    worst_tf = sorted(medians.items(), key=lambda item: (item[1], item[0]))[0][0]
    corr = score_df[cols].corr()
    corr_col = corr.get(f"score_{worst_tf}") if hasattr(corr, "get") else None
    if corr_col is not None:
        candidates = {}
        for tf in tf_names:
            if tf == worst_tf:
                continue
            key = f"score_{tf}"
            if key not in corr_col.index:
                continue
            value = corr_col.get(key)
            if value is None or not np.isfinite(value):
                continue
            candidates[tf] = float(value)
        if candidates:
            tradeoff_tf = sorted(candidates.items(), key=lambda item: (item[1], item[0]))[0][0]
            return (worst_tf, tradeoff_tf), "worst-median TF paired with lowest correlation partner"
    sorted_tfs = [tf for tf, _ in sorted(medians.items(), key=lambda item: (item[1], item[0]))]
    if len(sorted_tfs) >= 2:
        return (sorted_tfs[0], sorted_tfs[1]), "fallback to two lowest medians"
    return None, "unable to select pair"
