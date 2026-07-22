"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/explain.py

Runtime helpers for explain OPAL runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from ..core.utils import OpalError
from ..storage.label_sources import CampaignHistoryLabelSource, label_source_from_config
from .preflight import preflight_run
from .round_plan import plan_round


def explain_round(store, df, cfg, round_k: int) -> Dict[str, Any]:
    label_source = label_source_from_config(cfg, store)
    uses_campaign_history = isinstance(label_source, CampaignHistoryLabelSource)
    # Preflight: no writes/backfill during explain
    rep = preflight_run(
        store,
        df,
        round_k,
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
        check_manual_attach=uses_campaign_history,
    )
    if uses_campaign_history and rep.manual_attach_count:
        raise OpalError(
            f"Detected {rep.manual_attach_count} labels in '{store.y_col}' without label_hist. "
            "Run `opal ingest-y` (preferred) or explicitly reconcile the current Y column with "
            "`opal label-hist attach-from-y`."
        )
    label_source.validate(df)

    # Derive counts the same way 'run' does
    plan = plan_round(store, df, cfg, round_k, warnings=list(rep.warnings or []), label_source=label_source)
    round_counts: Dict[int, int] = {}
    if "r" in plan.training_df.columns:
        counts = plan.training_df["r"].value_counts(dropna=True).to_dict()
        round_counts = {int(k): int(v) for k, v in counts.items()}

    preflight: Dict[str, Any] = {
        "observed_round_definition": "Labels stamped by ingest-y --round (event time).",
        "labels_as_of_definition": "Training cutoff used by run/explain --round.",
    }
    sfxi_views = [view for view in cfg.selection_views if str(view.objective.name) == "sfxi_v1"]
    if sfxi_views:
        minimums = {
            view.id: int(dict((view.objective.params or {}).get("scaling") or {}).get("min_n", 5))
            for view in sfxi_views
        }
        current_round_labels = int(round_counts.get(int(round_k), 0))
        preflight.update(
            {
                "sfxi_scaling_min_n_by_view": minimums,
                "sfxi_current_round_labels": current_round_labels,
                "sfxi_run_will_fail": bool(current_round_labels < max(minimums.values())),
                "sfxi_fix_command": (f"opal ingest-y --round {int(round_k)} --csv <labels.xlsx> --apply"),
            }
        )

    info = {
        "round_index": round_k,
        "x_column_name": cfg.data.x_column_name,
        "y_column_name": cfg.data.y_column_name,
        "label_source": {"kind": label_source.kind},
        "representation_vector_dimension": rep.x_dim,
        "model": {"name": cfg.model.name, "params": cfg.model.params},
        "training_policy": cfg.training.policy,
        "training_label_dedup_policy": plan.training_dedup_policy,
        "training_y_ops": [{"name": p.name, "params": p.params} for p in (cfg.training.y_ops or [])],
        "selection_views": [
            {
                "id": view.id,
                "objective": {"name": view.objective.name, "params": view.objective.params},
                "selection": {"name": view.selection.name, "params": view.selection.params},
            }
            for view in cfg.selection_views
        ],
        "number_of_training_examples_used_in_round": int(len(plan.training_df)),
        "number_of_candidates_scored_in_round": int(len(plan.candidate_df)),
        "candidate_pool_total": int(plan.candidate_total_before_filter),
        "candidate_pool_filtered_out": int(plan.candidate_filtered_out),
        "training_labels_by_round": round_counts,
        "preflight": preflight,
        "warnings": list(plan.warnings or []),
    }
    return info
