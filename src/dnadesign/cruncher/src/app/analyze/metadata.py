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
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.pwm import PWM


@dataclass(frozen=True)
class SampleMeta:
    optimizer_kind: str
    chains: int
    draws: int
    tune: int
    bidirectional: bool
    top_k: int
    mode: str


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
    pwms: dict[str, PWM] = {}
    for tf_name, info in pwms_info.items():
        matrix = info.get("pwm_matrix")
        if not matrix:
            raise ValueError(f"config_used.yaml missing pwm_matrix for TF '{tf_name}'.")
        pwms[tf_name] = PWM(name=tf_name, matrix=np.array(matrix, dtype=float))
    return pwms, cruncher_cfg


def _resolve_sample_meta(cfg: CruncherConfig, used_cfg: dict, manifest: dict) -> SampleMeta:
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")
    if not isinstance(used_cfg, dict):
        raise ValueError("config_used.yaml missing cruncher config; re-run `cruncher sample`.")

    optimizer_payload = manifest.get("optimizer") if isinstance(manifest, dict) else None
    optimizer_kind = "pt"
    if isinstance(optimizer_payload, dict):
        optimizer_kind = str(optimizer_payload.get("kind") or "pt")

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
        mode="pt",
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
