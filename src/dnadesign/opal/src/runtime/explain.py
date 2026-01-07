"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/explain.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from ..core.utils import OpalError
from .preflight import preflight_run
from .round_plan import plan_round


def explain_round(store, df, cfg, round_k: int) -> Dict[str, Any]:
    # Preflight: no writes/backfill during explain
    rep = preflight_run(
        store,
        df,
        round_k,
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
    )
    if rep.manual_attach_count:
        raise OpalError(
            f"Detected {rep.manual_attach_count} labels in '{store.y_col}' without label_hist. "
            "Run `opal ingest-y` (preferred) or `opal label-hist attach-from-y` for legacy Y columns."
        )
    store.validate_label_hist(df, require=True)

    # Derive counts the same way 'run' does
    plan = plan_round(store, df, cfg, round_k, warnings=list(rep.warnings or []))

    info = {
        "round_index": round_k,
        "x_column_name": cfg.data.x_column_name,
        "y_column_name": cfg.data.y_column_name,
        "representation_vector_dimension": rep.x_dim,
        "model": {"name": cfg.model.name, "params": cfg.model.params},
        "training_policy": cfg.training.policy,
        "training_label_dedup_policy": plan.training_dedup_policy,
        "training_y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        "selection": {
            "strategy": cfg.selection.selection.name,
            "params": cfg.selection.selection.params,
            "objective": {
                "name": cfg.objective.objective.name,
                "params": cfg.objective.objective.params,
            },
        },
        "number_of_training_examples_used_in_round": int(len(plan.training_df)),
        "number_of_candidates_scored_in_round": int(len(plan.candidate_df)),
        "candidate_pool_total": int(plan.candidate_total_before_filter),
        "candidate_pool_filtered_out": int(plan.candidate_filtered_out),
        "warnings": list(plan.warnings or []),
    }
    return info
