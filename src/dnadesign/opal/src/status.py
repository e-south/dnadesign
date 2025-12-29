"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/status.py

Builds a JSON-able status dict from state.json.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class _RoundLite:
    round_index: int
    run_id: str
    number_of_training_examples_used_in_round: int
    number_of_candidates_scored_in_round: int
    selection_top_k_requested: int
    selection_top_k_effective_after_ties: int
    round_dir: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coalesce(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_status(
    state_path: Path,
    round_k: Optional[int] = None,
    show_all: bool = False,
    *,
    ledger_reader=None,
    include_ledger: bool = False,
) -> Dict[str, Any]:
    if not state_path.exists():
        return {"error": f"state.json not found: {state_path}"}

    st = _load_json(state_path)

    # Robust field access (older state files might have slightly different keys)
    campaign_slug = _coalesce(st, "campaign_slug", "slug", default="")
    campaign_name = _coalesce(st, "campaign_name", "name", default="")
    workdir = _coalesce(st, "workdir", default=str(state_path.parent.resolve()))
    x_column_name = _coalesce(st, "x_column_name", default="")
    y_column_name = _coalesce(st, "y_column_name", default="")

    rounds = st.get("rounds", [])
    rounds_sorted = sorted(rounds, key=lambda r: int(r.get("round_index", -1)))

    latest = rounds_sorted[-1] if rounds_sorted else None
    selected = None
    if round_k is not None:
        selected = next(
            (r for r in rounds_sorted if int(r.get("round_index", -1)) == int(round_k)),
            None,
        )

    def _lite(r: Dict[str, Any]) -> _RoundLite:
        return _RoundLite(
            round_index=int(r.get("round_index", -1)),
            run_id=str(r.get("run_id", "")),
            number_of_training_examples_used_in_round=int(r.get("number_of_training_examples_used_in_round", 0)),
            number_of_candidates_scored_in_round=int(r.get("number_of_candidates_scored_in_round", 0)),
            selection_top_k_requested=int(r.get("selection_top_k_requested", 0)),
            selection_top_k_effective_after_ties=int(r.get("selection_top_k_effective_after_ties", 0)),
            round_dir=str(r.get("round_dir", "")),
        )

    out: Dict[str, Any] = {
        "campaign_slug": campaign_slug,
        "campaign_name": campaign_name,
        "workdir": workdir,
        "x_column_name": x_column_name,
        "y_column_name": y_column_name,
        "num_rounds": len(rounds_sorted),
        "latest_round": asdict(_lite(latest)) if latest else None,
    }

    if selected is not None:
        out["selected_round"] = asdict(_lite(selected)) if selected else None

    if show_all:
        out["rounds"] = [asdict(_lite(r)) for r in rounds_sorted]

    if include_ledger:
        if ledger_reader is None:
            raise ValueError("include_ledger=True requires a ledger_reader")

        def _ledger_summary_for_round(as_of_round: int) -> Optional[Dict[str, Any]]:
            try:
                runs = ledger_reader.read_runs(
                    columns=[
                        "run_id",
                        "as_of_round",
                        "model__name",
                        "objective__name",
                        "selection__name",
                        "training__y_ops",
                        "stats__n_train",
                        "stats__n_scored",
                        "objective__summary_stats",
                    ]
                )
            except Exception:
                return None
            if runs.empty:
                return None
            rsel = runs[runs["as_of_round"] == int(as_of_round)]
            if rsel.empty:
                return None
            row = rsel.sort_values(["run_id"]).tail(1).iloc[0]
            return {
                "run_id": str(row.get("run_id", "")),
                "model": row.get("model__name"),
                "objective": row.get("objective__name"),
                "selection": row.get("selection__name"),
                "y_ops": row.get("training__y_ops") or [],
                "stats_n_train": int(row.get("stats__n_train", 0)),
                "stats_n_scored": int(row.get("stats__n_scored", 0)),
                "objective_summary_stats": row.get("objective__summary_stats") or {},
            }

        if latest:
            out["latest_round_ledger"] = _ledger_summary_for_round(int(latest.get("round_index", -1)))
        if selected is not None:
            out["selected_round_ledger"] = (
                _ledger_summary_for_round(int(selected.get("round_index", -1))) if selected else None
            )
        if show_all:
            for rr in out.get("rounds", []):
                rr["ledger"] = _ledger_summary_for_round(int(rr.get("round_index", -1)))

    return out
