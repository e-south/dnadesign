"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/status.py

Reads state.json and assembles a concise view:
- latest round summary by default,
- specific round or --all on request,
- echoes explicit snake_case fields (counts, metrics, artifacts).

Used by the status command and suitable for dashboards/CI logs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .state import CampaignState


def build_status(
    state_path: Path, round_k: Optional[int] = None, show_all: bool = False
) -> Dict[str, Any]:
    st = CampaignState.load(state_path)
    out: Dict[str, Any] = {
        "campaign_slug": st.campaign_slug,
        "campaign_name": st.campaign_name,
        "workdir": st.workdir,
        "x_column_name": st.x_column_name,
        "y_column_name": st.y_column_name,
        "representation_vector_dimension": st.representation_vector_dimension,
        "rounds_count": len(st.rounds),
    }
    if show_all:
        out["rounds"] = [r.__dict__ for r in st.rounds]
        return out
    if round_k is None:
        out["latest_round"] = st.rounds[-1].__dict__ if st.rounds else None
    else:
        matches = [r for r in st.rounds if r.round_index == round_k]
        out["round"] = matches[0].__dict__ if matches else None
    return out
