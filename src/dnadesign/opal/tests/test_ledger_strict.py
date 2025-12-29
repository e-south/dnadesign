"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_ledger_strict.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.ledger import LedgerWriter
from dnadesign.opal.src.utils import LedgerError
from dnadesign.opal.src.workspace import CampaignWorkspace


def test_ledger_rejects_unknown_columns(tmp_path):
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=tmp_path)
    writer = LedgerWriter(ws)
    df = pd.DataFrame(
        {
            "event": ["run_pred"],
            "run_id": ["r0-..."],
            "as_of_round": [0],
            "id": ["x"],
            "pred__y_dim": [1],
            "pred__y_hat_model": [[0.1]],
            "pred__y_obj_scalar": [0.2],
            "sel__rank_competition": [1],
            "sel__is_selected": [True],
            "unexpected": [123],
        }
    )
    with pytest.raises(LedgerError):
        writer.append_run_pred(df)
