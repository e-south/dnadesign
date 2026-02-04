"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli_sampling.py

CLI helpers for describing Stage-A sampling configuration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .core.motif_labels import input_motifs


def format_selection_label(
    *,
    policy: str,
    alpha: float | None = None,
    relevance_norm: str | None = None,
    max_candidates: int | None = None,
) -> str:
    selection_policy = str(policy or "top_score")
    if selection_policy != "mmr":
        return selection_policy
    if alpha is None:
        raise ValueError("MMR selection alpha is required.")
    parts: list[str] = []
    if alpha is not None:
        parts.append(f"a={float(alpha):.2f}")
    if relevance_norm and relevance_norm != "minmax_raw_score":
        parts.append(f"rel={relevance_norm}")
    if max_candidates is not None:
        parts.append(f"cap={int(max_candidates)}")
    if not parts:
        return "mmr"
    return f"mmr({','.join(parts)})"


def stage_a_plan_rows(
    cfg,
    cfg_path: Path,
    selected_inputs: set[str] | None,
    *,
    show_motif_ids: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for inp in cfg.inputs:
        if selected_inputs and inp.name not in selected_inputs:
            continue
        input_type = str(inp.type)
        if not input_type.startswith("pwm_"):
            continue
        sampling = getattr(inp, "sampling", None)
        base_n_sites = getattr(sampling, "n_sites", None) if sampling else None

        motifs = input_motifs(inp, cfg_path)
        overrides = getattr(inp, "overrides_by_motif_id", None) if input_type == "pwm_artifact_set" else None
        for motif_id, display_name in motifs:
            effective_sampling = sampling
            if overrides and motif_id in overrides:
                override = overrides.get(motif_id)
                if override is not None:
                    effective_sampling = override
            reg_n_sites = getattr(effective_sampling, "n_sites", base_n_sites) if effective_sampling else base_n_sites
            length_cfg = getattr(effective_sampling, "length", None) if effective_sampling else None
            length_policy = str(getattr(length_cfg, "policy", "-")) if length_cfg else "-"
            length_range = getattr(length_cfg, "range", None) if length_cfg else None
            length_label = length_policy
            if length_policy == "range" and length_range:
                length_label = f"range({length_range[0]}..{length_range[1]})"
            mining_cfg = getattr(effective_sampling, "mining", None) if effective_sampling else None
            budget = getattr(mining_cfg, "budget", None) if mining_cfg else None
            budget_label = "-"
            if budget is not None:
                mode = getattr(budget, "mode", None)
                if mode == "fixed_candidates":
                    budget_label = f"fixed={getattr(budget, 'candidates', '-')}"
                elif mode == "tier_target":
                    target_frac = getattr(budget, "target_tier_fraction", None)
                    tier_label = "-" if target_frac is None else f"{float(target_frac) * 100:.3f}%"
                    budget_label = (
                        f"tier={tier_label}"
                        f" max_candidates={getattr(budget, 'max_candidates', '-')}"
                        f" max_seconds={getattr(budget, 'max_seconds', '-')}"
                    )
            uniqueness_cfg = getattr(effective_sampling, "uniqueness", None) if effective_sampling else None
            uniqueness_label = str(getattr(uniqueness_cfg, "key", "-"))
            selection_cfg = getattr(effective_sampling, "selection", None) if effective_sampling else None
            selection_policy = str(getattr(selection_cfg, "policy", "top_score"))
            selection_alpha = getattr(selection_cfg, "alpha", None)
            pool_cfg = getattr(selection_cfg, "pool", None) if selection_cfg is not None else None
            selection_label = format_selection_label(
                policy=selection_policy,
                alpha=selection_alpha,
                relevance_norm=getattr(pool_cfg, "relevance_norm", None) if pool_cfg is not None else None,
                max_candidates=getattr(pool_cfg, "max_candidates", None) if pool_cfg is not None else None,
            )
            label = motif_id if show_motif_ids else display_name
            rows.append(
                {
                    "input": str(inp.name),
                    "tf": str(label),
                    "retain": str(reg_n_sites) if reg_n_sites is not None else "-",
                    "budget": budget_label,
                    "eligibility": "best_hit_score>0",
                    "selection": selection_label,
                    "uniqueness": uniqueness_label,
                    "length": length_label,
                }
            )
    rows.sort(key=lambda row: (row["input"], row["tf"]))
    return rows
