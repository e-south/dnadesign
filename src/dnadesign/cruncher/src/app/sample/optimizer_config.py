"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/optimizer_config.py

Resolve optimizer configuration and summarize sampling settings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import (
    BetaLadderFixed,
    BetaLadderGeometric,
    CoolingFixed,
    CoolingLinear,
    CoolingPiecewise,
    SampleConfig,
)


def _default_beta_ladder(chains: int, beta_min: float, beta_max: float) -> list[float]:
    if chains < 2:
        raise ValueError("PT beta ladder requires at least 2 chains")
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("PT geometric ladder requires beta_min>0 and beta_max>0")
    ladder = np.geomspace(beta_min, beta_max, chains, dtype=float)
    return [float(beta) for beta in ladder]


def _resolve_beta_ladder(pt_cfg) -> tuple[list[float], list[str]]:
    notes: list[str] = []
    ladder = pt_cfg.beta_ladder
    if isinstance(ladder, BetaLadderFixed):
        return [float(ladder.beta)], notes
    if isinstance(ladder, BetaLadderGeometric):
        if ladder.betas is not None:
            return [float(b) for b in ladder.betas], notes
        notes.append("derived PT beta ladder from beta_min/beta_max/n_temps")
        return _default_beta_ladder(int(ladder.n_temps), float(ladder.beta_min), float(ladder.beta_max)), notes
    raise ValueError("Unsupported beta_ladder configuration")


def _resolve_optimizer_kind(sample_cfg: SampleConfig) -> str:
    kind = sample_cfg.optimizer.name
    if kind == "auto":
        raise ValueError("sample.optimizer.name must be 'gibbs' or 'pt' for a concrete run.")
    return kind


def _effective_chain_count(sample_cfg: SampleConfig, *, kind: str) -> int:
    if kind == "gibbs":
        return sample_cfg.budget.restarts
    if kind == "pt":
        ladder, _ = _resolve_beta_ladder(sample_cfg.optimizers.pt)
        return len(ladder)
    raise ValueError(f"Unknown optimizer kind '{kind}'")


def _boost_cooling(cooling: Any, factor: float) -> tuple[Any, list[str]]:
    if factor <= 1:
        return cooling, []
    notes = [f"boosted cooling by x{factor:g} for stabilization"]
    if isinstance(cooling, CoolingLinear):
        beta = [float(cooling.beta[0]) * factor, float(cooling.beta[1]) * factor]
        return CoolingLinear(beta=beta), notes
    if isinstance(cooling, CoolingFixed):
        return CoolingFixed(beta=float(cooling.beta) * factor), notes
    if isinstance(cooling, CoolingPiecewise):
        stages = [{"sweeps": stage.sweeps, "beta": float(stage.beta) * factor} for stage in cooling.stages]
        return CoolingPiecewise(stages=stages), notes
    return cooling, notes


def _boost_beta_ladder(ladder: Any, factor: float) -> tuple[Any, list[str]]:
    if factor <= 1:
        return ladder, []
    notes = [f"boosted PT beta ladder by x{factor:g} for stabilization"]
    if isinstance(ladder, BetaLadderFixed):
        return BetaLadderFixed(beta=float(ladder.beta) * factor), notes
    if isinstance(ladder, BetaLadderGeometric):
        if ladder.betas is not None:
            betas = [float(b) * factor for b in ladder.betas]
            return BetaLadderGeometric(betas=betas), notes
        return BetaLadderGeometric(
            beta_min=float(ladder.beta_min) * factor,
            beta_max=float(ladder.beta_max) * factor,
            n_temps=int(ladder.n_temps),
        ), notes
    return ladder, notes


def _resize_beta_ladder(ladder: Any, n_temps: int | None) -> tuple[Any, list[str]]:
    if n_temps is None:
        return ladder, []
    if n_temps < 2:
        raise ValueError("auto_opt.pt_ladder_sizes entries must be >= 2")
    notes = [f"set PT ladder size={n_temps} for pilots"]
    if isinstance(ladder, BetaLadderFixed):
        raise ValueError("auto_opt.pt_ladder_sizes requires beta_ladder.kind='geometric' (fixed ladder).")
    if isinstance(ladder, BetaLadderGeometric):
        if ladder.betas is not None:
            betas = [float(b) for b in ladder.betas]
            if n_temps > len(betas):
                raise ValueError(
                    "auto_opt.pt_ladder_sizes exceeds available betas; "
                    "provide a geometric beta_ladder with beta_min/beta_max/n_temps."
                )
            if n_temps == len(betas):
                return ladder, []
            if n_temps == 2:
                idx = [0, len(betas) - 1]
            else:
                idx = [int(round(x)) for x in np.linspace(0, len(betas) - 1, n_temps)]
                idx = sorted(set(idx))
                if len(idx) < n_temps:
                    extras = [i for i in range(len(betas)) if i not in idx]
                    idx.extend(extras[: n_temps - len(idx)])
            resized = [betas[i] for i in idx[:n_temps]]
            return BetaLadderGeometric(betas=resized), notes
        return BetaLadderGeometric(
            beta_min=float(ladder.beta_min),
            beta_max=float(ladder.beta_max),
            n_temps=int(n_temps),
        ), notes
    return ladder, notes


def _format_cooling_summary(cooling: object) -> str:
    if isinstance(cooling, CoolingLinear):
        beta0, beta1 = cooling.beta
        return f"linear({beta0:g}->{beta1:g})"
    if isinstance(cooling, CoolingFixed):
        return f"fixed({cooling.beta:g})"
    if isinstance(cooling, CoolingPiecewise):
        stages = ",".join(f"{stage.sweeps}@{stage.beta:g}" for stage in cooling.stages)
        return f"piecewise([{stages}])"
    return str(cooling)


def _format_beta_ladder_summary(pt_cfg: object) -> str:
    betas, _ = _resolve_beta_ladder(pt_cfg)
    if isinstance(pt_cfg.beta_ladder, BetaLadderFixed):
        return f"fixed({betas[0]:g})"
    beta = ",".join(f"{b:g}" for b in betas)
    return f"geometric([{beta}])"


def _format_move_probs(move_probs: dict[str, float]) -> str:
    parts = [f"{key}={val:.2f}" for key, val in move_probs.items() if val > 0]
    return ",".join(parts) if parts else "none"


def _format_auto_opt_config_summary(cfg: SampleConfig) -> str:
    kind = cfg.optimizer.name
    moves = _format_move_probs(resolve_move_config(cfg.moves).move_probs)
    combine = cfg.objective.combine or ("sum" if cfg.objective.score_scale == "consensus-neglop-sum" else "min")
    if kind == "gibbs":
        cooling = _format_cooling_summary(cfg.optimizers.gibbs.beta_schedule)
        chains = cfg.budget.restarts
    else:
        cooling = _format_beta_ladder_summary(cfg.optimizers.pt)
        chains = len(_resolve_beta_ladder(cfg.optimizers.pt)[0])
        cooling = f"{cooling} swap_prob={cfg.optimizers.pt.swap_prob:g}"
    return (
        f"optimizer={kind} scorer={cfg.objective.score_scale} "
        f"combine={combine} length={cfg.init.length} chains={chains} tune={cfg.budget.tune} draws={cfg.budget.draws} "
        f"cooling={cooling} moves={moves} progress_every={cfg.ui.progress_every}"
    )
