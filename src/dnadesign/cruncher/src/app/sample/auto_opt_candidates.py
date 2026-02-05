"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/auto_opt_candidates.py

Auto-opt candidate construction helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json

from dnadesign.cruncher.app.sample.optimizer_config import _boost_beta_ladder
from dnadesign.cruncher.config.moves import resolve_move_config
from dnadesign.cruncher.config.schema_v2 import AutoOptConfig, SampleConfig


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
    total_sweeps: int,
    cooling_boost: float = 1.0,
    move_profile: str | None = None,
    swap_prob: float | None = None,
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

    pilot.optimizer.name = kind
    if kind != "pt":
        raise ValueError("auto-opt pilots only support optimizer.name='pt'")
    if pilot.budget.restarts != 1:
        pilot.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT pilots")

    if move_profile is not None and pilot.moves.profile != move_profile:
        pilot.moves.profile = move_profile
        notes.append(f"set moves.profile={move_profile} for pilots")
    if move_probs is not None:
        pilot.moves.overrides.move_probs = dict(move_probs)
        notes.append("set moves.overrides.move_probs for pilots")

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

    if cooling_boost != 1:
        ladder_cfg = pilot.optimizers.pt.beta_ladder
        boosted, boost_notes = _boost_beta_ladder(ladder_cfg, cooling_boost)
        pilot.optimizers.pt.beta_ladder = boosted
        notes.extend(boost_notes)

    return pilot, notes


def _build_final_sample_cfg(
    base_cfg: SampleConfig,
    *,
    kind: str,
    cooling_boost: float = 1.0,
    move_profile: str | None = None,
    swap_prob: float | None = None,
    move_probs: dict[str, float] | None = None,
) -> tuple[SampleConfig, list[str]]:
    notes: list[str] = []
    final_cfg = base_cfg.model_copy(deep=True)
    final_cfg.optimizer.name = kind
    if kind != "pt":
        raise ValueError("auto-opt final runs only support optimizer.name='pt'")
    if final_cfg.budget.restarts != 1:
        final_cfg.budget.restarts = 1
        notes.append("forced budget.restarts=1 for PT final run")
    if cooling_boost != 1:
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
    if swap_prob is not None and final_cfg.optimizers.pt.swap_prob != float(swap_prob):
        final_cfg.optimizers.pt.swap_prob = float(swap_prob)
        notes.append(f"set optimizers.pt.swap_prob={final_cfg.optimizers.pt.swap_prob:g} from auto-opt selection")
    return final_cfg, notes
