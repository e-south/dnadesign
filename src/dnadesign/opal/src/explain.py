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
    rep = preflight_run(
        store,
        df,
        round_k,
        fail_on_mixed_bio_alphabet=cfg.safety.fail_on_mixed_biotype_or_alphabet,
    )
    info = {
        "round_index": round_k,
        "representation_column_name": cfg.data.representation_column_name,
        "label_source_column_name": cfg.data.label_source_column_name,
        "representation_vector_dimension": rep.x_dim,
        "model": cfg.training["model"].dict(),
        "training_policy": cfg.training["policy"].dict(),
        "selection": cfg.selection.dict(),
        "number_of_training_examples_used_in_round": rep.n_labels,
        "number_of_candidates_scored_in_round": rep.n_candidates,
        "warnings": rep.warnings,
    }
    return info
