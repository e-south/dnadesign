"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/status.py

Reporting helpers for status OPAL reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.utils import OpalError
from ..storage.state import CampaignState, RoundEntry


@dataclass
class _RoundLite:
    round_index: int
    run_id: str
    number_of_training_examples_used_in_round: int
    number_of_candidates_scored_in_round: int
    selection_views: dict
    selection_batch: dict
    round_dir: str


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

    st = CampaignState.load(state_path)
    rounds_sorted = sorted(st.rounds, key=lambda r: int(r.round_index))

    latest = rounds_sorted[-1] if rounds_sorted else None
    selected = None
    if round_k is not None:
        selected = next((r for r in rounds_sorted if int(r.round_index) == int(round_k)), None)

    def _lite(r: RoundEntry) -> _RoundLite:
        return _RoundLite(
            round_index=int(r.round_index),
            run_id=str(r.run_id),
            number_of_training_examples_used_in_round=int(r.number_of_training_examples_used_in_round),
            number_of_candidates_scored_in_round=int(r.number_of_candidates_scored_in_round),
            selection_views=dict(r.selection_views),
            selection_batch=dict(r.selection_batch),
            round_dir=str(r.round_dir),
        )

    out: Dict[str, Any] = {
        "campaign_slug": st.campaign_slug,
        "campaign_name": st.campaign_name,
        "workdir": st.workdir,
        "x_column_name": st.x_column_name,
        "y_column_name": st.y_column_name,
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
                        "objective__defs_json",
                        "selection_views__defs_json",
                        "training__y_ops",
                        "stats__n_train",
                        "stats__n_scored",
                    ]
                )
            except Exception as e:
                raise OpalError("Ledger sinks not found. Run `opal run` to create ledger outputs.") from e
            if runs.empty:
                return None
            rsel = runs[runs["as_of_round"] == int(as_of_round)]
            if rsel.empty:
                return None
            row = rsel.sort_values(["run_id"]).tail(1).iloc[0]
            from .summary import summarize_run_meta

            return summarize_run_meta(row)

        if latest:
            out["latest_round_ledger"] = _ledger_summary_for_round(int(latest.round_index))
        if selected is not None:
            out["selected_round_ledger"] = _ledger_summary_for_round(int(selected.round_index))
        if show_all:
            for rr in out.get("rounds", []):
                rr["ledger"] = _ledger_summary_for_round(int(rr.get("round_index", -1)))

    return out
