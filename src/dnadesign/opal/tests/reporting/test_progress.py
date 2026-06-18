"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/reporting/test_progress.py

Regression tests for progress OPAL reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.opal.src.reporting.progress import build_campaign_progress
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state


def test_campaign_progress_clamps_underreported_predict_batch_total(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True, exist_ok=True)
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    log_path = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-05-25T00:00:00+00:00", "stage": "start"}),
                json.dumps(
                    {
                        "ts": "2026-05-25T00:00:01+00:00",
                        "stage": "predict_batch",
                        "batch": 124,
                        "of": 123,
                        "rows": 97,
                    }
                ),
                json.dumps({"ts": "2026-05-25T00:00:02+00:00", "stage": "done"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = build_campaign_progress(campaign, round_selector="latest")

    assert payload["rounds"][0]["predict"] == {"batch": 124, "of": 124, "rows": 97}
