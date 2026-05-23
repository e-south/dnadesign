"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/reporting/test_notebook_set.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting import notebook_set as notebook_set_mod
from dnadesign.opal.src.reporting.notebook_set import (
    build_campaign_set_notebook_view_model,
    build_campaign_set_round_options,
)
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_round_log


def test_campaign_set_notebook_view_model_collects_campaigns(tmp_path: Path) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(
            config_path,
            workdir=workdir,
            records_path=records_path,
            slug=slug,
        )
        config_paths.append(config_path)

    payload = build_campaign_set_notebook_view_model(config_paths, round_selector="latest")

    assert payload["schema_version"] == "opal.notebook_campaign_set_view_model.v1"
    assert payload["campaign_count"] == 2
    assert [row["campaign"]["slug"] for row in payload["campaigns"]] == ["campaign_a", "campaign_b"]
    assert payload["campaigns"][0]["campaign"]["config_path"] == str(config_paths[0])


def test_campaign_set_round_options_include_round_history(tmp_path: Path) -> None:
    config_paths = []
    for slug, rounds in {"campaign_a": [0, 2], "campaign_b": [1]}.items():
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(
            config_path,
            workdir=workdir,
            records_path=records_path,
            slug=slug,
        )
        for round_index in rounds:
            write_round_log(
                workdir / "outputs" / "rounds" / f"round_{round_index}" / "logs" / "round.log.jsonl",
                run_id=f"{slug}-run-{round_index}",
                round_index=round_index,
            )
        config_paths.append(config_path)

    assert build_campaign_set_round_options(config_paths) == ["latest", "all", "0", "1", "2"]


def test_campaign_set_round_options_fail_fast_for_malformed_progress(tmp_path: Path, monkeypatch) -> None:
    config_paths = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
    for path in config_paths:
        path.write_text("campaign: {}\n", encoding="utf-8")

    def _bad_view_model(path: Path, *, round_selector: str):
        return {"progress": {"rounds": [{"round_index": "not-an-int"}]}}

    monkeypatch.setattr(notebook_set_mod, "build_notebook_view_model", _bad_view_model)

    with pytest.raises(OpalError, match="non-integer round_index"):
        build_campaign_set_round_options(config_paths)
