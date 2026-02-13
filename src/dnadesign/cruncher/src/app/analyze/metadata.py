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
import yaml

from dnadesign.cruncher.artifacts.layout import config_used_path
from dnadesign.cruncher.core.optimizers.kinds import resolve_optimizer_kind
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.pwm_window import select_pwm_window


@dataclass(frozen=True)
class SampleMeta:
    optimizer_kind: str
    chains: int
    draws: int
    tune: int
    bidirectional: bool
    top_k: int
    mode: str


def _resolve_sampling_window(used_cfg: dict) -> tuple[int, int | None, str] | None:
    sample_payload = used_cfg.get("sample")
    if not isinstance(sample_payload, dict):
        return None
    motif_width = sample_payload.get("motif_width")
    if motif_width is None:
        return None
    if not isinstance(motif_width, dict):
        raise ValueError("config_used.yaml sample.motif_width must be an object when provided.")
    seq_len = sample_payload.get("sequence_length")
    if not isinstance(seq_len, int) or seq_len < 1:
        raise ValueError("config_used.yaml sample.sequence_length must be an integer >= 1 when motif_width is set.")
    maxw = motif_width.get("maxw")
    if maxw is not None and (not isinstance(maxw, int) or maxw < 1):
        raise ValueError("config_used.yaml sample.motif_width.maxw must be null or an integer >= 1.")
    strategy = str(motif_width.get("strategy") or "max_info")
    if strategy != "max_info":
        raise ValueError("config_used.yaml sample.motif_width.strategy must be 'max_info'.")
    return seq_len, maxw, strategy


def _effective_sampling_pwm(
    pwm: PWM,
    *,
    seq_len: int,
    maxw: int | None,
    strategy: str,
) -> PWM:
    max_allowed = seq_len if maxw is None else min(maxw, seq_len)
    if pwm.length <= max_allowed:
        return pwm
    return select_pwm_window(pwm, length=max_allowed, strategy=strategy)


def _load_pwms_from_config(run_dir: Path) -> tuple[dict[str, PWM], dict]:
    config_path = config_used_path(run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {run_dir}")
    payload = yaml.safe_load(config_path.read_text()) or {}
    cruncher_cfg = payload.get("cruncher")
    if not isinstance(cruncher_cfg, dict):
        raise ValueError("config_used.yaml missing top-level 'cruncher' section.")
    pwms_info = cruncher_cfg.get("pwms_info")
    if not isinstance(pwms_info, dict) or not pwms_info:
        raise ValueError("config_used.yaml missing pwms_info; re-run `cruncher sample`.")
    sampling_window = _resolve_sampling_window(cruncher_cfg)
    pwms: dict[str, PWM] = {}
    for tf_name, info in pwms_info.items():
        matrix = info.get("pwm_matrix")
        if not matrix:
            raise ValueError(f"config_used.yaml missing pwm_matrix for TF '{tf_name}'.")
        full_pwm = PWM(name=tf_name, matrix=np.array(matrix, dtype=float))
        if sampling_window is None:
            pwms[tf_name] = full_pwm
        else:
            seq_len, maxw, strategy = sampling_window
            pwms[tf_name] = _effective_sampling_pwm(full_pwm, seq_len=seq_len, maxw=maxw, strategy=strategy)
    return pwms, cruncher_cfg


def _resolve_sample_meta(used_cfg: dict, manifest: dict) -> SampleMeta:
    if not isinstance(used_cfg, dict):
        raise ValueError("config_used.yaml missing cruncher config; re-run `cruncher sample`.")

    optimizer_payload = manifest.get("optimizer") if isinstance(manifest, dict) else None
    if optimizer_payload is not None and not isinstance(optimizer_payload, dict):
        raise ValueError("Run manifest field 'optimizer' must be an object when provided.")
    optimizer_kind = resolve_optimizer_kind(
        optimizer_payload.get("kind") if isinstance(optimizer_payload, dict) else None,
        context="Run manifest field 'optimizer.kind'",
    )

    draws = int(manifest.get("draws") or 0)
    tune = int(manifest.get("adapt_sweeps") or 0)
    top_k = int(manifest.get("top_k") or 0)
    objective_payload = manifest.get("objective") if isinstance(manifest, dict) else None
    bidirectional = False
    if isinstance(objective_payload, dict):
        bidirectional = bool(objective_payload.get("bidirectional"))

    optimizer_stats = manifest.get("optimizer_stats") if isinstance(manifest, dict) else None
    beta_ladder = []
    if isinstance(optimizer_stats, dict):
        ladder_payload = optimizer_stats.get("beta_ladder_final")
        if isinstance(ladder_payload, list):
            beta_ladder = ladder_payload
    chains = len(beta_ladder) if beta_ladder else 1

    return SampleMeta(
        optimizer_kind=optimizer_kind,
        chains=chains,
        draws=draws,
        tune=tune,
        bidirectional=bidirectional,
        top_k=top_k,
        mode=optimizer_kind,
    )


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
