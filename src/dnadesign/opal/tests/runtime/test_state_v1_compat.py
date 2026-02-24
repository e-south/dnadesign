"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/runtime/test_state_v1_compat.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.opal.src.reporting.status import build_status
from dnadesign.opal.src.storage.state import CampaignState


def test_state_v1_loads_without_run_id(tmp_path):
    state_path = Path(tmp_path) / "state.json"
    v1_state = {
        "version": 1,
        "campaign_slug": "demo",
        "campaign_name": "Demo",
        "workdir": str(tmp_path),
        "data_location": {
            "kind": "local",
            "path": str(tmp_path),
            "records_path": str(tmp_path / "records.parquet"),
        },
        "x_column_name": "X",
        "y_column_name": "Y",
        "rounds": [
            {
                "round_index": 0,
                "round_name": "round_0",
                "round_dir": str(tmp_path / "outputs" / "rounds" / "round_0"),
                "labels_used_rounds": [0],
                "number_of_training_examples_used_in_round": 2,
                "number_of_candidates_scored_in_round": 3,
                "selection_top_k_requested": 1,
                "selection_top_k_effective_after_ties": 1,
                "model": {},
                "metrics": {},
                "durations_sec": {},
                "seeds": {},
                "artifacts": {},
                "writebacks": {},
                "warnings": [],
            }
        ],
    }
    state_path.write_text(json.dumps(v1_state))
    st = CampaignState.load(state_path)
    assert st.rounds and st.rounds[0].run_id == ""

    status = build_status(state_path)
    assert status.get("latest_round") is not None
