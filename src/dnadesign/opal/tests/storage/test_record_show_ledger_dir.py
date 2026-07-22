"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/storage/test_record_show_ledger_dir.py

Regression tests for record show ledger dir OPAL storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd

from dnadesign.opal.src.reporting.record_show import build_record_report
from dnadesign.opal.src.storage.ledger import LedgerReader
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.tests._cli_helpers import write_ledger


def test_record_show_reads_ledger_predictions_dir(tmp_path):
    rec = pd.DataFrame(
        {
            "id": ["a"],
            "sequence": ["AAA"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [None],
        }
    )

    workdir = tmp_path
    write_ledger(workdir, run_id="r0", round_index=0)

    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    reader = LedgerReader(ws)
    report = build_record_report(
        rec,
        "demo",
        id_="a",
        ledger_reader=reader,
        selection_view_id="primary",
    )
    assert report["runs"] and report["runs"][0]["view__is_selected"] is True
    assert report["latest_rank_competition"] == 1
