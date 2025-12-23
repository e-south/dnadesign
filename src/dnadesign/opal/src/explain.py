"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/explain.py

Dry-run planner for a round.

Aggregates what would happen in run --round k without mutating anything:
- effective training set counts (after dedup policy),
- candidate universe size,
- model and selection configs,
- representation info and vector dimension,
- any preflight warnings.

Emits a JSON-ready dict for CLI or programmatic use.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from .preflight import preflight_run


def explain_round(store, df, cfg, round_k: int) -> Dict[str, Any]:
    # Preflight: no writes/backfill during explain
    rep = preflight_run(
        store,
        df,
        round_k,
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
    )
    # Derive counts the same way 'run' does
    train_df = store.training_labels_from_y(df, round_k)
    cand_df = store.candidate_universe(df, round_k)

    info = {
        "round_index": round_k,
        "x_column_name": cfg.data.x_column_name,
        "y_column_name": cfg.data.y_column_name,
        "representation_vector_dimension": rep.x_dim,
        "model": {"name": cfg.model.name, "params": cfg.model.params},
        "training_policy": cfg.training.policy,
        "training_y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        "selection": {
            "strategy": cfg.selection.selection.name,
            "params": cfg.selection.selection.params,
            "objective": {
                "name": cfg.objective.objective.name,
                "params": cfg.objective.objective.params,
            },
        },
        "number_of_training_examples_used_in_round": int(len(train_df)),
        "number_of_candidates_scored_in_round": int(len(cand_df)),
        "warnings": getattr(rep, "warnings", []),
    }
    return info
