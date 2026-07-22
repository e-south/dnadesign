"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/reporting/test_notebook_set.py

Regression tests for notebook set OPAL reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import pytest
import yaml

from dnadesign.opal.src.analysis.campaign_set import build_campaign_set_collection_visual_model
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting import notebook_set as notebook_set_mod
from dnadesign.opal.src.reporting.campaign_collection import load_campaign_collection_manifest
from dnadesign.opal.src.reporting.campaign_set_artifacts import materialize_campaign_set_collection_visuals
from dnadesign.opal.src.reporting.collection_visual_index import load_collection_visual_manifest_index
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


def test_campaign_set_notebook_view_model_accepts_one_campaign(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr(
        notebook_set_mod,
        "build_notebook_campaign_set_selection_overlap_choice",
        lambda *_args, **_kwargs: {"visual_id": "must-not-be-built"},
    )

    payload = build_campaign_set_notebook_view_model([config_path], round_selector="latest")

    assert payload["campaign_count"] == 1
    assert payload["campaigns"][0]["campaign"]["slug"] == "campaign_a"
    assert payload["collection_visuals"] == []


def test_campaign_set_notebook_view_model_rejects_collection_inputs_for_one_campaign(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign_a"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    write_records(records_path, slug="campaign_a")
    config_path = workdir / "campaign.yaml"
    write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug="campaign_a")

    with pytest.raises(OpalError, match="at least two distinct campaign configs"):
        build_campaign_set_notebook_view_model(
            [config_path],
            collection_manifest_path=tmp_path / "collection.yaml",
        )


def test_campaign_set_notebook_view_model_does_not_treat_selection_views_as_campaigns(
    tmp_path: Path, monkeypatch
) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["selection_views"] = [
            {**config["selection_views"][0], "id": view_id} for view_id in ["ethanol", "ciprofloxacin", "and"]
        ]
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config_paths.append(config_path)

    monkeypatch.setattr(
        notebook_set_mod,
        "build_notebook_campaign_set_selection_overlap_choice",
        lambda *_args, **_kwargs: {"visual_id": "must-not-be-built"},
    )

    payload = build_campaign_set_notebook_view_model(config_paths, round_selector="latest")

    assert payload["campaign_count"] == 2
    assert payload["collection_visuals"] == []


def test_campaign_set_notebook_view_model_retains_overlap_for_distinct_single_view_campaigns(
    tmp_path: Path, monkeypatch
) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        config_paths.append(config_path)

    overlap = {"visual_id": "pooled_selection_overlap"}
    monkeypatch.setattr(
        notebook_set_mod,
        "build_notebook_campaign_set_selection_overlap_choice",
        lambda *_args, **_kwargs: overlap,
    )

    payload = build_campaign_set_notebook_view_model(config_paths, round_selector="latest")

    assert payload["collection_visuals"] == [overlap]


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
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "densegen_probe_seed7",
                "dimensions": [
                    {"id": "target", "label": "Target"},
                    {"id": "label_oracle_kind", "label": "Label oracle kind"},
                    {"id": "label_family_id", "label": "Label family"},
                    {"id": "label_split_id", "label": "Label split"},
                    {"id": "seed", "label": "Seed"},
                ],
                "relationships": [
                    {
                        "id": "positive_vs_null",
                        "kind": "control_pair",
                        "label": "Positive vs null oracle control",
                        "left_role": "positive",
                        "right_role": "null",
                        "role_dimension": "label_oracle_kind",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                        "replicate_on": ["seed"],
                    }
                ],
                "comparison_views": [
                    {
                        "id": "selected_score_positive_vs_null",
                        "label": "Selected score positive/null trajectory",
                        "kind": "metric_over_rounds_comparison",
                        "relationship_id": "positive_vs_null",
                        "source_plot_name": "score_selected_over_rounds",
                        "source_plot_kind": "metric_over_rounds",
                        "comparison_scope": "comparison_set",
                        "group_key": "label_oracle_kind",
                        "metric": "pred__score_selected",
                        "cohort": "selected",
                        "summary": "mean",
                        "interval_kind": "none",
                        "interpretation_note": "Selected score uses each campaign's configured objective scale.",
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
    assert collection["schema_version"] == "opal.campaign_collection.v2"
    assert collection["collection_id"] == "densegen_probe_seed7"
    assert collection["path"] == str(collection_path)
    assert [row["id"] for row in collection["dimensions"]] == [
        "target",
        "label_oracle_kind",
        "label_family_id",
        "label_split_id",
        "seed",
    ]
    assert collection["dimension_ids"] == ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"]
    assert collection["relationships"][0]["id"] == "positive_vs_null"
    assert collection["relationships"][0]["role_dimension"] == "label_oracle_kind"
    assert collection["relationships"][0]["pair_count"] == 2
    assert collection["comparison_views"] == [
        {
            "id": "selected_score_positive_vs_null",
            "label": "Selected score positive/null trajectory",
            "kind": "metric_over_rounds_comparison",
            "relationship_id": "positive_vs_null",
            "source_plot_name": "score_selected_over_rounds",
            "source_plot_kind": "metric_over_rounds",
            "comparison_scope": "comparison_set",
            "comparison_set_count": 2,
            "match_filters": {},
            "group_key": "label_oracle_kind",
            "metric": "pred__score_selected",
            "cohort": "selected",
            "summary": "mean",
            "interval_kind": "none",
            "confidence_level": None,
            "interpretation_note": "Selected score uses each campaign's configured objective scale.",
            "relationship": collection["relationships"][0],
        }
    ]
    assert collection["comparison_lenses"] == [
        {
            "id": "positive_vs_null",
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
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "probe_aliases",
                "dimensions": [
                    {"id": "probe_target"},
                    {"id": "probe_oracle_kind"},
                    {"id": "probe_label_family_id"},
                    {"id": "probe_split_id"},
                    {"id": "probe_seed"},
                ],
                "relationships": [
                    {
                        "id": "probe_positive_vs_null",
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
                "comparison_views": [
                    {
                        "id": "probe_selected_score",
                        "label": "Probe selected score",
                        "kind": "metric_over_rounds_comparison",
                        "relationship_id": "probe_positive_vs_null",
                        "source_plot_name": "score_selected_over_rounds",
                        "source_plot_kind": "metric_over_rounds",
                        "comparison_scope": "comparison_set",
                        "group_key": "probe_oracle_kind",
                        "metric": "pred__score_selected",
                        "cohort": "selected",
                        "summary": "mean",
                        "interval_kind": "none",
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
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "bad_pairs",
                "dimensions": [
                    {"id": "target"},
                    {"id": "label_oracle_kind"},
                    {"id": "label_family_id"},
                    {"id": "label_split_id"},
                    {"id": "seed"},
                ],
                "relationships": [
                    {
                        "id": "positive_vs_null",
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    }
                ],
                "comparison_views": [
                    {
                        "id": "selected_score",
                        "label": "Selected score",
                        "kind": "metric_over_rounds_comparison",
                        "relationship_id": "positive_vs_null",
                        "source_plot_name": "score_selected_over_rounds",
                        "source_plot_kind": "metric_over_rounds",
                        "comparison_scope": "comparison_set",
                        "group_key": "label_oracle_kind",
                        "metric": "pred__score_selected",
                        "cohort": "selected",
                        "summary": "mean",
                        "interval_kind": "iqr",
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


def test_campaign_collection_manifest_rejects_v1_schema(tmp_path: Path) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        config_paths.append(config_path)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v1",
                "dimensions": ["target"],
                "relationships": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(OpalError, match="Unsupported campaign collection schema_version"):
        build_campaign_set_notebook_view_model(
            config_paths,
            round_selector="latest",
            collection_manifest_path=collection_path,
        )


def test_campaign_collection_manifest_rejects_ci_without_explicit_estimator(tmp_path: Path) -> None:
    config_paths = []
    for oracle_kind in ["positive", "null"]:
        slug = f"campaign_{oracle_kind}"
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["campaign"]["metadata"] = {
            "target": "cipro",
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
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "bad_ci",
                "dimensions": [
                    {"id": "target"},
                    {"id": "label_oracle_kind"},
                    {"id": "label_family_id"},
                    {"id": "label_split_id"},
                    {"id": "seed"},
                ],
                "relationships": [
                    {
                        "id": "positive_vs_null",
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "role_dimension": "label_oracle_kind",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    }
                ],
                "comparison_views": [
                    {
                        "id": "invalid_ci",
                        "label": "Invalid CI",
                        "kind": "metric_over_rounds_comparison",
                        "relationship_id": "positive_vs_null",
                        "source_plot_name": "score_selected_over_rounds",
                        "source_plot_kind": "metric_over_rounds",
                        "comparison_scope": "comparison_set",
                        "group_key": "label_oracle_kind",
                        "metric": "pred__score_selected",
                        "cohort": "selected",
                        "summary": "median",
                        "interval_kind": "student_t_mean_ci",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(OpalError, match="student_t_mean_ci requires summary='mean'"):
        build_campaign_set_notebook_view_model(
            config_paths,
            round_selector="latest",
            collection_manifest_path=collection_path,
        )


def test_collection_visual_model_expands_declared_views_by_campaign_set() -> None:
    campaigns = []
    pairs = []
    for target in ["cipro", "ethanol"]:
        pairs.append(
            {
                "left": f"{target}_positive_random_id",
                "right": f"{target}_null_random_id",
                "match": {
                    "target": target,
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": "7",
                },
            }
        )
        for oracle_kind in ["positive", "null"]:
            campaigns.append(
                {
                    "campaign": {
                        "slug": f"{target}_{oracle_kind}_random_id",
                        "metadata": {
                            "target": target,
                            "label_oracle_kind": oracle_kind,
                            "label_family_id": "densegen_plan_logic4",
                            "label_split_id": "random_id",
                            "seed": 7,
                        },
                    }
                }
            )
    collection = {
        "schema_version": "opal.campaign_collection.v2",
        "collection_id": "fixture",
        "relationships": [
            {
                "id": "positive_vs_null",
                "kind": "control_pair",
                "role_dimension": "label_oracle_kind",
                "left_role": "positive",
                "right_role": "null",
                "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                "replicate_on": ["seed"],
                "pair_count": len(pairs),
                "pairs": pairs,
            }
        ],
        "comparison_views": [
            {
                "id": "selected_score_positive_vs_null",
                "label": "Selected score positive/null trajectory",
                "kind": "metric_over_rounds_comparison",
                "relationship_id": "positive_vs_null",
                "source_plot_name": "score_selected_over_rounds",
                "source_plot_kind": "metric_over_rounds",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "pred__score_selected",
                "cohort": "selected",
                "summary": "mean",
                "interval_kind": "none",
                "interpretation_note": "Selected score uses the configured objective scale.",
                "relationship": {
                    "id": "positive_vs_null",
                    "kind": "control_pair",
                    "role_dimension": "label_oracle_kind",
                    "left_role": "positive",
                    "right_role": "null",
                    "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    "replicate_on": ["seed"],
                    "pairs": pairs,
                },
            }
        ],
    }

    model = build_campaign_set_collection_visual_model(campaigns, collection)

    assert model["visual_count"] == 2
    assert model["comparison_set_count"] == 2
    assert [row["label"] for row in model["comparison_sets"]] == [
        "Cipro | DenseGen plan logic4 | Random ID",
        "Ethanol | DenseGen plan logic4 | Random ID",
    ]
    assert {visual["comparison_set_key"] for visual in model["visuals"]} == {
        "label_family_id=densegen_plan_logic4|label_split_id=random_id|target=cipro",
        "label_family_id=densegen_plan_logic4|label_split_id=random_id|target=ethanol",
    }
    assert all(len(visual["pairs"]) == 1 for visual in model["visuals"])


def test_materialized_collection_visuals_do_not_cross_join_duplicate_seed_slugs(tmp_path: Path) -> None:
    def _campaign(seed: int, oracle_kind: str, value: float) -> dict:
        slug = "opal_densegen_probe_v1_plan_logic4_cipro_" + oracle_kind + "_random_id"
        workdir = tmp_path / f"seed{seed}" / slug
        plots_dir = workdir / "outputs" / "plots"
        plots_dir.mkdir(parents=True)
        tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
        tidy_path.write_text(
            f"round,cohort,metric,summary,value\n0,selected,pred__score_selected,mean,{value}\n",
            encoding="utf-8",
        )
        return {
            "campaign": {
                "slug": slug,
                "workdir": str(workdir),
                "config_path": str(workdir / "configs" / "campaign.yaml"),
                "metadata": {
                    "target": "cipro",
                    "label_oracle_kind": oracle_kind,
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": seed,
                },
            },
            "plot_manifests": [
                {
                    "name": "score_selected_over_rounds",
                    "kind": "metric_over_rounds",
                    "status": "written",
                    "rounds": "all",
                    "params": {
                        "metric_label": "Score = -MSE(y_hat, [0, 0, 1, 1])",
                        "legend_metric_label": "negative MSE score",
                        "collection_visual_label": "Score trajectory: negative MSE to logic4 target",
                    },
                    "tidy_csv": str(tidy_path),
                    "outputs": [{"role": "tidy_csv", "path": str(tidy_path), "exists": True}],
                }
            ],
        }

    campaigns = [
        _campaign(seed=7, oracle_kind="positive", value=0.70),
        _campaign(seed=7, oracle_kind="null", value=0.10),
        _campaign(seed=17, oracle_kind="positive", value=0.90),
        _campaign(seed=17, oracle_kind="null", value=0.20),
    ]
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "duplicate_seed_slug_fixture",
                "dimensions": [
                    {"id": "target"},
                    {"id": "label_oracle_kind"},
                    {"id": "label_family_id"},
                    {"id": "label_split_id"},
                    {"id": "seed"},
                ],
                "relationships": [
                    {
                        "id": "positive_vs_null",
                        "kind": "control_pair",
                        "left_role": "positive",
                        "right_role": "null",
                        "role_dimension": "label_oracle_kind",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                        "replicate_on": ["seed"],
                    }
                ],
                "comparison_views": [
                    {
                        "id": "objective_score_positive_vs_null",
                        "label": "Objective score positive/null trajectory",
                        "kind": "metric_over_rounds_comparison",
                        "relationship_id": "positive_vs_null",
                        "source_plot_name": "score_selected_over_rounds",
                        "source_plot_kind": "metric_over_rounds",
                        "comparison_scope": "comparison_set",
                        "group_key": "label_oracle_kind",
                        "metric": "pred__score_selected",
                        "cohort": "selected",
                        "summary": "mean",
                        "interval_kind": "iqr",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    collection = load_campaign_collection_manifest(collection_path, campaigns)
    pairs = collection["relationships"][0]["pairs"]
    assert len(pairs) == 2
    assert sorted(pair["match"]["seed"] for pair in pairs) == ["17", "7"]
    assert len({pair["left"] for pair in pairs}) == 2
    assert len({pair["right"] for pair in pairs}) == 2

    output_dir = tmp_path / "collection_visuals"
    index = materialize_campaign_set_collection_visuals(
        campaigns,
        collection=collection,
        output_dir=output_dir,
    )

    visual = index["visuals"][0]
    assert visual["row_count"] == 4
    with (output_dir / visual["tidy_csv"]).open("r", encoding="utf-8", newline="") as handle:
        tidy_rows = list(csv.DictReader(handle))
    observed = {
        (row["replicate_key"], row["group"]): float(row["value"])
        for row in tidy_rows
        if row["metric"] == "pred__score_selected"
    }
    assert observed == {
        ("seed=7", "positive"): 0.70,
        ("seed=7", "null"): 0.10,
        ("seed=17", "positive"): 0.90,
        ("seed=17", "null"): 0.20,
    }


def test_materialize_campaign_set_collection_visuals_writes_manifest_backed_outputs(tmp_path: Path) -> None:
    def _campaign(
        slug: str,
        *,
        target: str,
        group: str,
        values: list[float],
        mse_values: list[float],
        vector_values: list[tuple[float, float]],
    ) -> dict:
        workdir = tmp_path / slug
        plots_dir = workdir / "outputs" / "plots"
        plots_dir.mkdir(parents=True)
        tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
        tidy_path.write_text(
            "round,cohort,metric,summary,value\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,mean,{value}" for round_index, value in enumerate(values)
            )
            + "\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,count,12" for round_index, _value in enumerate(values)
            )
            + "\n",
            encoding="utf-8",
        )
        vector_tidy_path = plots_dir / "selected_target_vector_summary_rall.csv"
        vector_tidy_path.write_text(
            "row_type,round,cohort,channel,value,n\n"
            "reference_vector,,Target vector,v00,0,\n"
            "reference_vector,,Target vector,v01,1,\n"
            + "\n".join(
                f"round,{round_index},selected,v00,{channel_values[0]},12\n"
                f"round,{round_index},selected,v01,{channel_values[1]},12"
                for round_index, channel_values in enumerate(vector_values)
            )
            + "\n"
            + "\n".join(
                f"reference_mse,{round_index},selected,mse,{value},12" for round_index, value in enumerate(mse_values)
            )
            + "\n",
            encoding="utf-8",
        )
        media_path = plots_dir / "selected_target_vector_summary_rall.png"
        media_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
            )
        )
        return {
            "campaign": {
                "slug": slug,
                "metadata": {
                    "target": target,
                    "label_oracle_kind": group,
                    "label_family_id": "densegen_plan_logic4",
                    "label_split_id": "random_id",
                    "seed": 7,
                },
            },
            "plot_manifests": [
                {
                    "name": "score_selected_over_rounds",
                    "kind": "metric_over_rounds",
                    "status": "written",
                    "rounds": "all",
                    "params": {
                        "metric_label": "Score = -MSE(y_hat, [0, 0, 1, 1])",
                        "legend_metric_label": "negative MSE score",
                        "collection_visual_label": "Score trajectory: negative MSE to logic4 target",
                        "metric_expression": (
                            "score = -mean((y_hat - [0, 0, 1, 1])^2); loss = mean((y_hat - [0, 0, 1, 1])^2)"
                        ),
                        "y_axis": {
                            "scale_class": "densegen_plan_logic4_negative_mse",
                            "limits": [-0.25, 0.0],
                            "include_zero_tick": True,
                        },
                    },
                    "tidy_csv": str(tidy_path),
                    "outputs": [{"role": "tidy_csv", "path": str(tidy_path), "exists": True}],
                },
                {
                    "name": "selected_target_vector_summary",
                    "kind": "vector_summary_heatmap",
                    "status": "written",
                    "rounds": "all",
                    "params": {
                        "reference_mse_metric_label": "MSE = mean((mean selected y_hat - target)^2)",
                        "reference_mse_expression": "MSE = mean_c((mean selected y_hat_c - target_c)^2)",
                        "reference_mse_y_limits": [0.0, 0.25],
                    },
                    "tidy_csv": str(vector_tidy_path),
                    "outputs": [
                        {"role": "media", "path": str(media_path), "exists": True},
                        {"role": "tidy_csv", "path": str(vector_tidy_path), "exists": True},
                    ],
                },
            ],
        }

    campaigns = [
        _campaign(
            "cipro_positive_random_id",
            target="cipro",
            group="positive",
            values=[0.2, 0.5],
            mse_values=[0.4, 0.2],
            vector_values=[(0.1, 0.7), (0.0, 0.9)],
        ),
        _campaign(
            "cipro_null_random_id",
            target="cipro",
            group="null",
            values=[0.1, 0.15],
            mse_values=[0.8, 0.7],
            vector_values=[(0.4, 0.3), (0.3, 0.4)],
        ),
    ]
    collection = {
        "schema_version": "opal.campaign_collection.v2",
        "collection_id": "fixture",
        "dimensions": [
            {"id": "target"},
            {"id": "label_oracle_kind"},
            {"id": "label_family_id"},
            {"id": "label_split_id"},
            {"id": "seed"},
        ],
        "relationships": [
            {
                "id": "positive_vs_null",
                "kind": "control_pair",
                "label": "Positive vs null",
                "role_dimension": "label_oracle_kind",
                "left_role": "positive",
                "right_role": "null",
                "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                "replicate_on": ["seed"],
                "pair_count": 1,
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
                    }
                ],
            }
        ],
        "comparison_views": [
            {
                "id": "selected_score_positive_vs_null",
                "label": "Selected score positive/null trajectory",
                "kind": "metric_over_rounds_comparison",
                "relationship_id": "positive_vs_null",
                "source_plot_name": "score_selected_over_rounds",
                "source_plot_kind": "metric_over_rounds",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "pred__score_selected",
                "cohort": "selected",
                "summary": "mean",
                "interval_kind": "none",
                "interpretation_note": "Selected score uses the configured objective scale.",
                "relationship": {
                    "id": "positive_vs_null",
                    "kind": "control_pair",
                    "role_dimension": "label_oracle_kind",
                    "left_role": "positive",
                    "right_role": "null",
                    "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    "replicate_on": ["seed"],
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
                        }
                    ],
                },
            },
            {
                "id": "selected_vector_reference_mse",
                "label": "Selected vector reference MSE",
                "kind": "vector_reference_mse_over_rounds_comparison",
                "relationship_id": "positive_vs_null",
                "source_plot_name": "selected_target_vector_summary",
                "source_plot_kind": "vector_summary_heatmap",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "reference_mse",
                "cohort": "selected",
                "summary": "mean",
                "interval_kind": "none",
                "relationship": {
                    "id": "positive_vs_null",
                    "kind": "control_pair",
                    "role_dimension": "label_oracle_kind",
                    "left_role": "positive",
                    "right_role": "null",
                    "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    "replicate_on": ["seed"],
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
                        }
                    ],
                },
            },
            {
                "id": "selected_vector_heatmap",
                "label": "Selected predicted vector heatmap",
                "kind": "vector_heatmap_comparison",
                "relationship_id": "positive_vs_null",
                "source_plot_name": "selected_target_vector_summary",
                "source_plot_kind": "vector_summary_heatmap",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "selected_predicted_vector",
                "cohort": "selected",
                "summary": "mean",
                "interval_kind": "none",
                "relationship": {
                    "id": "positive_vs_null",
                    "kind": "control_pair",
                    "role_dimension": "label_oracle_kind",
                    "left_role": "positive",
                    "right_role": "null",
                    "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    "replicate_on": ["seed"],
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
                        }
                    ],
                },
            },
        ],
    }

    output_dir = tmp_path / "collection_visuals"
    output_dir.mkdir()
    stale_paths = [
        output_dir / "stale.csv",
        output_dir / "stale.png",
        output_dir / "stale.manifest.json",
        output_dir / "collection_visual_manifest.json",
    ]
    for stale_path in stale_paths:
        stale_path.write_text("stale", encoding="utf-8")

    index = materialize_campaign_set_collection_visuals(
        campaigns,
        collection=collection,
        output_dir=output_dir,
    )

    assert index["schema_version"] == "opal.collection_visual_manifest_index.v1"
    assert all(not stale_path.exists() for stale_path in stale_paths[:3])
    assert index["visual_count"] == 3
    assert index["comparison_set_count"] == 1
    assert index["comparison_sets"][0]["label"] == "Cipro | DenseGen plan logic4 | Random ID"
    visual = next(row for row in index["visuals"] if row["visual_id"] == "selected_score_positive_vs_null")
    assert visual["schema_version"] == "opal.collection_visual_artifact.v1"
    assert visual["surface_kind"] == "campaign_set_metric_comparison"
    assert visual["summary"] == "mean"
    assert visual["comparison_set_key"] == "label_family_id=densegen_plan_logic4|label_split_id=random_id|target=cipro"
    assert visual["interval"]["kind"] == "none"
    assert visual["interval"]["is_confidence_interval"] is False
    assert "Values are mean Score = -MSE" in visual["caption"]
    assert "Values are median" not in visual["caption"]
    assert visual["label"] == "Score trajectory: negative MSE to logic4 target"
    assert visual["metric_label"] == "Score = -MSE(y_hat, [0, 0, 1, 1])"
    assert "loss = mean" in visual["metric_expression"]
    assert visual["axis_scale"]["scale_class"] == "densegen_plan_logic4_negative_mse"
    assert visual["axis_scale"]["limits"] == [-0.25, 0.0]
    assert visual["interpretation_note"] == "Selected score uses the configured objective scale."
    assert "configured objective scale" in visual["caption"]
    assert "Score = -MSE(y_hat, [0, 0, 1, 1])" in visual["alt_text"]
    assert Path(visual["outputs"][0]["path"]).exists()
    assert Path(visual["tidy_csv"]).name == visual["tidy_csv"]
    assert Path(visual["path"]).name == visual["path"]
    assert Path(visual["manifest_path"]).name == visual["manifest_path"]
    assert (output_dir / visual["tidy_csv"]).exists()
    assert (output_dir / visual["path"]).exists()
    assert (output_dir / visual["manifest_path"]).exists()
    assert (
        load_collection_visual_manifest_index(
            output_dir / "collection_visual_manifest.json",
            expected_collection_id="fixture",
        )["visual_count"]
        == 3
    )
    mse_visual = next(row for row in index["visuals"] if row["visual_id"] == "selected_vector_reference_mse")
    assert mse_visual["surface_kind"] == "campaign_set_vector_reference_mse_comparison"
    assert mse_visual["metric"] == "reference_mse"
    assert mse_visual["axis_scale"]["limits"] == [0.0, 0.25]
    assert mse_visual["axis_scale"]["reference_lines"] == []
    assert "mean vector" in mse_visual["caption"]
    assert "mean vector" in mse_visual["alt_text"]
    assert (output_dir / mse_visual["path"]).exists()
    heatmap_visual = next(row for row in index["visuals"] if row["visual_id"] == "selected_vector_heatmap")
    assert heatmap_visual["surface_kind"] == "campaign_set_vector_heatmap_comparison"
    assert heatmap_visual["row_count"] == 12
    assert heatmap_visual["axis_scale"]["limits"] == [0.0, 0.25]
    assert heatmap_visual["axis_scale"]["reference_lines"] == []
    assert heatmap_visual["plot_question"] == "Cipro DenseGen plan logic4: positive vs null selected-vector convergence"
    assert heatmap_visual["target_vector_label"] == "target vector [v00=0, v01=1]"
    assert heatmap_visual["mse_formula"] == "MSE = mean_c((mean selected y_hat_c - target_c)^2)"
    assert heatmap_visual["visual_contract"] == {
        "heatmap_cell_geometry": "unit_square_cells",
        "heatmap_cell_edges": "white_cell_edges_only",
        "heatmap_background_grid": "off",
        "mse_panel": "shared_axis_group_lines",
    }
    assert "target vector [v00=0, v01=1]" in heatmap_visual["caption"]
    assert "MSE = mean_c((mean selected y_hat_c - target_c)^2)" in heatmap_visual["caption"]
    assert "shared color scale" in heatmap_visual["caption"]
    assert "Positive" in heatmap_visual["alt_text"]
    assert "Null" in heatmap_visual["alt_text"]
    assert (output_dir / heatmap_visual["path"]).exists()


def test_materialize_campaign_set_collection_visuals_rejects_artifact_stem_collisions(tmp_path: Path) -> None:
    collection = {
        "collection_id": "fixture",
        "comparison_views": [
            {
                "id": "selected_score",
                "label": "Selected score",
                "kind": "metric_over_rounds_comparison",
                "source_plot_name": "score_selected_over_rounds",
                "source_plot_kind": "metric_over_rounds",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "pred__score_selected",
                "cohort": "selected",
                "summary": "mean",
                "interval_kind": "none",
                "relationship": {
                    "id": "positive_vs_null",
                    "kind": "control_pair",
                    "role_dimension": "label_oracle_kind",
                    "match_on": ["target", "seed"],
                    "replicate_on": ["seed"],
                    "pairs": [
                        {
                            "left": "a_positive",
                            "right": "a_null",
                            "match": {"target": "a/b", "seed": "7"},
                        },
                        {
                            "left": "b_positive",
                            "right": "b_null",
                            "match": {"target": "a b", "seed": "7"},
                        },
                    ],
                },
            }
        ],
    }

    with pytest.raises(OpalError, match="artifact stem collision"):
        materialize_campaign_set_collection_visuals([], collection=collection, output_dir=tmp_path / "visuals")


def test_collection_visual_manifest_index_validates_loaded_study_surfaces(tmp_path: Path) -> None:
    index_path = _write_collection_visual_index_fixture(tmp_path)

    with pytest.raises(OpalError, match="without caller approval"):
        load_collection_visual_manifest_index(index_path, expected_collection_id="fixture_collection")

    payload = load_collection_visual_manifest_index(
        index_path,
        expected_collection_id="fixture_collection",
        allowed_surface_kinds=["study_realized_label_review"],
    )

    assert payload["visual_count"] == 1
    assert payload["visuals"][0]["surface_kind"] == "study_realized_label_review"

    with pytest.raises(OpalError, match="collection_id mismatch"):
        load_collection_visual_manifest_index(
            index_path,
            expected_collection_id="other_collection",
            allowed_surface_kinds=["study_realized_label_review"],
        )


def test_campaign_set_notebook_view_model_resolves_collection_visual_paths(tmp_path: Path) -> None:
    config_paths = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records_path = workdir / "records.parquet"
        write_records(records_path, slug=slug)
        config_path = workdir / "campaign.yaml"
        write_campaign_yaml(config_path, workdir=workdir, records_path=records_path, slug=slug)
        config_paths.append(config_path)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "fixture_collection",
                "dimensions": [{"id": "target"}],
                "relationships": [],
                "comparison_views": [],
                "collection_visual_surface_kinds": ["study_realized_label_review"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    index_path = _write_collection_visual_index_fixture(
        tmp_path / "collection_visuals",
        visual_overrides={
            "path": "visual.png",
            "tidy_csv": "visual.csv",
            "manifest_path": "visual.manifest.json",
        },
    )

    payload = build_campaign_set_notebook_view_model(
        config_paths,
        collection_manifest_path=collection_path,
        collection_visual_index_path=index_path,
    )

    visual = payload["collection_visuals"][0]
    assert visual["path"] == str((index_path.parent / "visual.png").resolve())
    assert visual["tidy_csv"] == str((index_path.parent / "visual.csv").resolve())
    assert visual["manifest_path"] == str((index_path.parent / "visual.manifest.json").resolve())


def test_collection_visual_manifest_index_rejects_undeclared_or_stale_entries(tmp_path: Path) -> None:
    undeclared_path = _write_collection_visual_index_fixture(
        tmp_path / "undeclared",
        surface_kinds=[],
    )
    with pytest.raises(OpalError, match="surface_kind is not declared"):
        load_collection_visual_manifest_index(undeclared_path)

    missing_media_path = _write_collection_visual_index_fixture(
        tmp_path / "missing_media",
        path_override="missing.png",
    )
    with pytest.raises(OpalError, match="does not exist"):
        load_collection_visual_manifest_index(
            missing_media_path,
            allowed_surface_kinds=["study_realized_label_review"],
        )

    bad_count_path = _write_collection_visual_index_fixture(
        tmp_path / "bad_count",
        payload_overrides={"visual_count": 2},
    )
    with pytest.raises(OpalError, match="visual_count=2"):
        load_collection_visual_manifest_index(
            bad_count_path,
            allowed_surface_kinds=["study_realized_label_review"],
        )


def test_collection_visual_manifest_index_deep_validates_generic_artifact_manifests(tmp_path: Path) -> None:
    bad_schema_path = _write_collection_visual_index_fixture(
        tmp_path / "bad_generic_schema",
        surface_kinds=["campaign_set_metric_comparison"],
        visual_overrides={"surface_kind": "campaign_set_metric_comparison"},
        manifest_overrides={
            "schema_version": "stress.fixture.owner_manifest.v1",
            "surface_kind": "campaign_set_metric_comparison",
        },
    )
    with pytest.raises(OpalError, match="Unsupported collection visual artifact manifest schema"):
        load_collection_visual_manifest_index(bad_schema_path)

    mismatch_path = _write_collection_visual_index_fixture(
        tmp_path / "generic_manifest_mismatch",
        surface_kinds=["campaign_set_metric_comparison"],
        visual_overrides={"surface_kind": "campaign_set_metric_comparison"},
        manifest_overrides={
            "schema_version": "opal.collection_visual_artifact.v1",
            "surface_kind": "campaign_set_metric_comparison",
            "visual_id": "other_visual",
        },
    )
    with pytest.raises(OpalError, match="visual_id mismatch"):
        load_collection_visual_manifest_index(mismatch_path)


def test_collection_visual_manifest_index_requires_schema_for_extension_manifests(tmp_path: Path) -> None:
    index_path = _write_collection_visual_index_fixture(
        tmp_path / "missing_extension_schema",
        manifest_overrides={"schema_version": ""},
    )

    with pytest.raises(OpalError, match="no schema_version"):
        load_collection_visual_manifest_index(
            index_path,
            allowed_surface_kinds=["study_realized_label_review"],
        )


def test_campaign_set_visual_model_rejects_unknown_view_kind() -> None:
    collection = {
        "collection_id": "fixture",
        "comparison_views": [
            {
                "id": "bad_visual",
                "kind": "not_registered",
                "relationship": {"pairs": []},
            }
        ],
    }

    with pytest.raises(OpalError, match="Unsupported collection comparison view kind"):
        build_campaign_set_collection_visual_model([], collection)


def test_campaign_set_visual_model_requires_manifest_owned_fields() -> None:
    collection = {
        "collection_id": "fixture",
        "comparison_views": [
            {
                "id": "selected_score",
                "kind": "metric_over_rounds_comparison",
                "source_plot_name": "score_selected_over_rounds",
                "source_plot_kind": "metric_over_rounds",
                "comparison_scope": "comparison_set",
                "group_key": "label_oracle_kind",
                "metric": "pred__score_selected",
                "cohort": "selected",
                "summary": "mean",
                "relationship": {"pairs": []},
            }
        ],
    }

    with pytest.raises(OpalError, match="interval_kind"):
        build_campaign_set_collection_visual_model([], collection)


def _write_collection_visual_index_fixture(
    root: Path,
    *,
    surface_kinds: list[str] | None = None,
    path_override: str | None = None,
    payload_overrides: dict[str, object] | None = None,
    visual_overrides: dict[str, object] | None = None,
    manifest_overrides: dict[str, object] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    media = root / "visual.png"
    tidy = root / "visual.csv"
    manifest = root / "visual.manifest.json"
    media.write_bytes(b"png")
    tidy.write_text("round,value\n0,1\n", encoding="utf-8")
    visual: dict[str, object] = {
        "collection_id": "fixture_collection",
        "visual_id": "selected_label_lift",
        "label": "Selected-label lift",
        "surface_kind": "study_realized_label_review",
        "comparison_set_key": "target=lexA",
        "comparison_set_label": "LexA",
        "path": str(media if path_override is None else path_override),
        "tidy_csv": str(tidy),
        "manifest_path": str(manifest),
        "freshness": {"status": "current"},
    }
    if visual_overrides:
        visual.update(visual_overrides)
    manifest_payload: dict[str, object] = {
        "schema_version": "stress.fixture.owner_manifest.v1",
        "collection_id": visual["collection_id"],
        "visual_id": visual["visual_id"],
        "surface_kind": visual["surface_kind"],
        "comparison_set_key": visual["comparison_set_key"],
        "path": visual["path"],
        "tidy_csv": visual["tidy_csv"],
    }
    if manifest_overrides:
        manifest_payload.update(manifest_overrides)
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "opal.collection_visual_manifest_index.v1",
        "generated_at": "2026-06-02T00:00:00+00:00",
        "collection_id": "fixture_collection",
        "output_dir": str(root),
        "surface_kinds": ["study_realized_label_review"] if surface_kinds is None else surface_kinds,
        "comparison_set_count": 1,
        "comparison_sets": [{"key": "target=lexA", "label": "LexA"}],
        "visual_count": 1,
        "visuals": [visual],
    }
    if payload_overrides:
        payload.update(payload_overrides)
    index_path = root / "collection_visual_manifest.json"
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return index_path


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
