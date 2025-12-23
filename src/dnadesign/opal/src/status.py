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


def build_status(state_path: Path, round_k: Optional[int] = None, show_all: bool = False) -> Dict[str, Any]:
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

    return out
