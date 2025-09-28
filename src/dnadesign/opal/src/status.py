"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/status.py

Robust status reader that does not depend on internal CampaignState attributes.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_state(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build_status(
    state_path: Path, *, round_k: Optional[int] = None, show_all: bool = False
) -> Dict[str, Any]:
    st = _read_state(state_path)
    if not st:
        return {"ok": False, "error": f"state.json not found at {state_path}"}

    rounds = st.get("rounds", [])
    rounds_sorted = sorted(rounds, key=lambda r: r.get("round_index", -1))
    latest = rounds_sorted[-1] if rounds_sorted else None

    base = {
        "ok": True,
        "campaign_slug": st.get("campaign_slug"),
        "campaign_name": st.get("campaign_name"),
        "workdir": st.get("workdir"),
        "data_location": st.get("data_location"),
        "x_column_name": st.get("x_column_name"),
        "y_column_name": st.get("y_column_name"),
        "representation_vector_dimension": st.get("representation_vector_dimension"),
        "training_policy": st.get("training_policy"),
        "performance": st.get("performance"),
    }

    if show_all:
        base["rounds"] = rounds_sorted
        return base

    if round_k is not None:
        match = next(
            (r for r in rounds_sorted if r.get("round_index") == int(round_k)), None
        )
        base["round"] = match
        return base

    base["latest_round"] = latest
    return base
