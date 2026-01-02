"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_record_show_ledger_dir.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd

from dnadesign.opal.src.reporting.record_show import build_record_report
from dnadesign.opal.src.storage.ledger import LedgerReader
from dnadesign.opal.src.storage.workspace import CampaignWorkspace


def test_record_show_reads_ledger_predictions_dir(tmp_path):
    rec = pd.DataFrame(
        {
            "id": ["x"],
            "sequence": ["AC"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [None],
        }
    )

    workdir = tmp_path
    pred_dir = workdir / "outputs" / "ledger.predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ev = pd.DataFrame(
        {
            "event": ["run_pred"],
            "run_id": ["r0-..."],
            "as_of_round": [0],
            "id": ["x"],
            "sequence": ["AC"],
            "pred__y_dim": [8],
            "pred__y_obj_scalar": [0.5],
            "sel__rank_competition": [1],
            "sel__is_selected": [True],
        }
    )
    ev.to_parquet(pred_dir / "part-000.parquet", index=False)

    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    reader = LedgerReader(ws)
    report = build_record_report(rec, "demo", id_="x", ledger_reader=reader)
    assert report["runs"] and report["runs"][0]["sel__is_selected"] is True
    assert report["latest_rank_competition"] == 1
