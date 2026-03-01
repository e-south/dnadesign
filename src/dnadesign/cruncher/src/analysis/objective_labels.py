"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/objective_labels.py

Shared objective scale and aggregate semantics labels for analysis visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping


def objective_scale_label(
    objective_config: Mapping[str, object] | None,
    *,
    unknown_fallback: str | None = None,
) -> str:
    cfg = objective_config if isinstance(objective_config, Mapping) else {}
    score_scale = str(cfg.get("score_scale") or "normalized-llr").strip().lower()
    if score_scale in {"llr", "raw-llr", "raw_llr"}:
        return "raw-LLR"
    if score_scale in {"normalized-llr", "norm-llr", "norm_llr"}:
        return "norm-LLR"
    if score_scale == "logp":
        return "logp"
    if score_scale == "z":
        return "z"
    if unknown_fallback is not None:
        return str(unknown_fallback)
    return score_scale


def objective_scalar_semantics(
    objective_config: Mapping[str, object] | None,
    *,
    unknown_scale_fallback: str | None = None,
) -> str:
    cfg = objective_config if isinstance(objective_config, Mapping) else {}
    combine = str(cfg.get("combine") or "min").strip().lower()
    scale_label = objective_scale_label(cfg, unknown_fallback=unknown_scale_fallback)
    softmin_cfg = cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, Mapping) and bool(softmin_cfg.get("enabled"))
    if combine == "sum":
        return f"sum TF best-window {scale_label}"
    if combine == "min" and softmin_enabled:
        return f"soft-min TF best-window {scale_label}"
    return f"min TF best-window {scale_label}"


__all__ = [
    "objective_scale_label",
    "objective_scalar_semantics",
]
