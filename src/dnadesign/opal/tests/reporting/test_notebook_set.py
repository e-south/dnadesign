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
import yaml

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


def test_campaign_set_notebook_view_model_accepts_one_campaign(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign_a"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path, slug="campaign_a")
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(
        config_path,
        workdir=workdir,
        records_path=records_path,
        slug="campaign_a",
    )

    payload = build_campaign_set_notebook_view_model([config_path], round_selector="latest")

    assert payload["campaign_count"] == 1
    assert payload["campaigns"][0]["campaign"]["slug"] == "campaign_a"


def test_campaign_set_notebook_view_model_loads_collection_relationships(tmp_path: Path) -> None:
    config_paths = []
    for target in ["cipro", "ethanol"]:
        for oracle_kind in ["positive", "null"]:
            slug = f"{target}_{oracle_kind}_random_id"
            workdir = tmp_path / slug
            workdir.mkdir(parents=True, exist_ok=True)
            records_path = workdir / "records.parquet"
            write_records(records_path, slug=slug)
            config_path = workdir / "campaign.yaml"
            write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload["campaign"]["metadata"] = {
                "target": target,
                "label_oracle_kind": oracle_kind,
                "label_family_id": "densegen_plan_logic4",
                "label_split_id": "random_id",
                "seed": 7,
            }
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            config_paths.append(config_path)

    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v1",
                "dimensions": ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"],
                "relationships": [
                    {
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                        "replicate_on": ["seed"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = build_campaign_set_notebook_view_model(
        config_paths,
        round_selector="latest",
        collection_manifest_path=collection_path,
    )

    collection = payload["collection"]
    assert collection["schema_version"] == "opal.campaign_collection.v1"
    assert collection["path"] == str(collection_path)
    assert collection["dimensions"] == ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"]
    assert collection["relationships"][0]["role_dimension"] == "label_oracle_kind"
    assert collection["relationships"][0]["pair_count"] == 2
    assert collection["comparison_lenses"] == [
        {
            "kind": "control_pair",
            "label": "Control pair by label oracle kind",
            "group_key": "label_oracle_kind",
            "role_dimension": "label_oracle_kind",
            "left_role": "positive",
            "right_role": "null",
            "match_on": ["target", "label_family_id", "label_split_id", "seed"],
            "replicate_on": ["seed"],
            "pair_count": 2,
            "pairs": [
                {
                    "left": "cipro_positive_random_id",
                    "right": "cipro_null_random_id",
                    "match": {
                        "target": "cipro",
                        "label_family_id": "densegen_plan_logic4",
                        "label_split_id": "random_id",
                        "seed": "7",
                    },
                },
                {
                    "left": "ethanol_positive_random_id",
                    "right": "ethanol_null_random_id",
                    "match": {
                        "target": "ethanol",
                        "label_family_id": "densegen_plan_logic4",
                        "label_split_id": "random_id",
                        "seed": "7",
                    },
                },
            ],
        }
    ]


def test_campaign_collection_lens_display_aliases_probe_metadata(tmp_path: Path) -> None:
    config_paths = []
    for oracle_kind in ["positive", "null"]:
        slug = f"cipro_{oracle_kind}_random_id"
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["campaign"]["metadata"] = {
            "probe_target": "cipro",
            "probe_oracle_kind": oracle_kind,
            "probe_label_family_id": "densegen_plan_logic4",
            "probe_split_id": "random_id",
            "probe_seed": 7,
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths.append(config_path)

    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v1",
                "dimensions": [
                    "probe_target",
                    "probe_oracle_kind",
                    "probe_label_family_id",
                    "probe_split_id",
                    "probe_seed",
                ],
                "relationships": [
                    {
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": [
                            "probe_target",
                            "probe_label_family_id",
                            "probe_split_id",
                            "probe_seed",
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = build_campaign_set_notebook_view_model(
        config_paths,
        round_selector="latest",
        collection_manifest_path=collection_path,
    )

    assert payload["collection"]["comparison_lenses"][0]["label"] == "Control pair by label oracle kind"


def test_campaign_collection_manifest_fails_when_relationship_matches_no_pairs(tmp_path: Path) -> None:
    config_paths = []
    for slug, oracle_kind in {"campaign_positive": "positive", "campaign_null": "null"}.items():
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["campaign"]["metadata"] = {
            "target": slug,
            "label_oracle_kind": oracle_kind,
            "label_family_id": "densegen_plan_logic4",
            "label_split_id": "random_id",
            "seed": 7,
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths.append(config_path)

    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v1",
                "dimensions": ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"],
                "relationships": [
                    {
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(OpalError, match="matched no campaign pairs"):
        build_campaign_set_notebook_view_model(
            config_paths,
            round_selector="latest",
            collection_manifest_path=collection_path,
        )


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
