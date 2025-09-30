"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_record_show_events.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from dnadesign.opal.src.record_show import build_record_report


def test_record_show_uses_events(tmp_path):
    rec = pd.DataFrame(
        {"id": ["x"], "sequence": ["AC"], "bio_type": ["dna"], "alphabet": ["dna_4"]}
    )
    ev = pd.DataFrame(
        {
            "event": ["run_pred"],
            "run_id": ["r0-..."],
            "as_of_round": [0],
            "id": ["x"],
            "pred__y_dim": [8],
            "pred__y_hat_model": [list(np.zeros(8))],
            "pred__y_obj_scalar": [0.5],
            "sel__rank_competition": [1],
            "sel__is_selected": [True],
        }
    )
    ep = tmp_path / "events.parquet"
    ev.to_parquet(ep, index=False)
    rep = build_record_report(rec, "demo", id_="x", events_path=ep)
    assert rep["runs"] and rep["runs"][0]["sel__is_selected"] is True
