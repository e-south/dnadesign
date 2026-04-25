"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/cli/test_workspace_command.py

Workspace command contracts for latentdna CLI.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from os.path import relpath
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr import Dataset
from dnadesign.usr.src.datasets.demo.mock import MockSpec, create_mock_dataset

_RUNNER = CliRunner()


def _build_overlay_usr_sources(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_root"
    register_test_namespace(
        usr_root,
        namespace="infer",
        columns_spec="infer__x_representation:list<float32>,infer__label_vec8:list<float32>",
    )
    register_test_namespace(
        usr_root,
        namespace="construct",
        columns_spec="construct__anchor_id:string,construct__context_id:string",
    )
    create_mock_dataset(
        usr_root,
        "anchor",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="infer"),
        force=True,
    )
    create_mock_dataset(
        usr_root,
        "contexts",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="infer"),
        force=True,
    )
    context_dataset = Dataset(usr_root, "contexts")
    ids = context_dataset.head(n=3, columns=["id"], include_derived=False)["id"].tolist()
    context_dataset.write_overlay_part(
        "construct",
        pa.table(
            {
                "id": ids,
                "construct__anchor_id": ids,
                "construct__context_id": ["ctx0", "ctx1", "ctx2"],
            }
        ),
        key="id",
    )
    return usr_root


def _build_committee_usr_sources(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_root"
    register_test_namespace(
        usr_root,
        namespace="mock",
        columns_spec="mock__x_representation:list<float32>,mock__label_vec8:list<float32>",
    )
    register_test_namespace(
        usr_root,
        namespace="infer",
        columns_spec=",".join(
            [
                "infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean:list<float32>",
                "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean:list<float32>",
                "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token:float32",
                "infer__evo2_7b__anchor_only_7b_features__output_layer_mean__seq_mean:list<float32>",
                "infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean:list<float32>",
                "infer__evo2_7b__template_1kb_7b_features__intermediate_embedding__block26_mlp_out__anchor_mean:list<float32>",
                "infer__evo2_20b__template_1kb_20b_features__intermediate_embedding__block23_mlp_out__anchor_mean:list<float32>",
                "infer__evo2_7b__template_1kb_7b_features__intermediate_embedding__block26_mlp_out__seq_mean:list<float32>",
                "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token:float32",
                "infer__evo2_7b__template_1kb_7b_features__output_layer_mean__seq_mean:list<float32>",
                "infer__evo2_7b__template_1kb_7b_features__output_layer_mean__anchor_mean:list<float32>",
                "infer__evo2_20b__template_1kb_20b_features__intermediate_embedding__block23_mlp_out__seq_mean:list<float32>",
                "infer__evo2_20b__template_1kb_20b_features__output_layer_mean__seq_mean:list<float32>",
                "infer__evo2_20b__template_1kb_20b_features__output_layer_mean__anchor_mean:list<float32>",
            ]
        ),
    )
    register_test_namespace(
        usr_root,
        namespace="construct",
        columns_spec="construct__anchor_id:string,construct__context_id:string,construct__template_id:string",
    )
    register_test_namespace(
        usr_root,
        namespace="densegen",
        columns_spec=",".join(
            [
                "densegen__plan:string",
                "densegen__required_regulators:string",
                "densegen__used_tfbs_detail:string",
            ]
        ),
    )
    register_test_namespace(
        usr_root,
        namespace="usr_label",
        columns_spec="usr_label__primary:string",
    )
    create_mock_dataset(
        usr_root,
        "promoter/test_anchor",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )
    create_mock_dataset(
        usr_root,
        "promoter/test_contexts",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )

    anchor_dataset = Dataset(usr_root, "promoter/test_anchor")
    anchor_records_path = usr_root / "promoter" / "test_anchor" / "records.parquet"
    anchor_records = pq.read_table(anchor_records_path)
    pq.write_table(
        anchor_records.append_column("template_id", pa.array(["tpl0", "tpl1", "tpl2"])),
        anchor_records_path,
    )
    anchor_ids = anchor_dataset.head(n=3, columns=["id"], include_derived=False)["id"].tolist()
    anchor_dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": anchor_ids,
                (
                    "infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean"
                ): pa.array(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    type=pa.list_(pa.float32()),
                ),
                (
                    "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean"
                ): pa.array(
                    [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_7b__anchor_only_7b_features__output_layer_mean__seq_mean"): pa.array(
                    [[1.8, 1.9], [2.0, 2.1], [2.2, 2.3]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token"): pa.array(
                    [-6.1, -5.8, -6.4],
                    type=pa.float32(),
                ),
                ("infer__evo2_20b__anchor_only_20b_features__output_layer_mean__seq_mean"): pa.array(
                    [[2.0, 2.1], [2.2, 2.3], [2.4, 2.5]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )
    anchor_dataset.write_overlay_part(
        "densegen",
        pa.table(
            {
                "id": anchor_ids,
                "densegen__plan": ["ethanol__sigma70_b", "ciprofloxacin__sigma70_c", "background_only__sigma70_d"],
                "densegen__required_regulators": ["cpxR", "lexA", "background"],
                "densegen__used_tfbs_detail": [
                    '[{"part_kind":"fixed_element","spacer_length":17}]',
                    '[{"part_kind":"fixed_element","spacer_length":16}]',
                    '[{"part_kind":"fixed_element","spacer_length":18}]',
                ],
            }
        ),
        key="id",
    )
    anchor_dataset.write_overlay_part(
        "usr_label",
        pa.table({"id": anchor_ids, "usr_label__primary": ["spyP", "sulAp", "J23105"]}),
        key="id",
    )
    context_dataset = Dataset(usr_root, "promoter/test_contexts")
    context_ids = context_dataset.head(n=3, columns=["id"], include_derived=False)["id"].tolist()
    context_dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": context_ids,
                (
                    "infer__evo2_7b__template_1kb_7b_features__intermediate_embedding__block26_mlp_out__anchor_mean"
                ): pa.array(
                    [[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]],
                    type=pa.list_(pa.float32()),
                ),
                (
                    "infer__evo2_20b__template_1kb_20b_features__intermediate_embedding__block23_mlp_out__anchor_mean"
                ): pa.array(
                    [[1.5, 1.4], [1.3, 1.2], [1.1, 1.0]],
                    type=pa.list_(pa.float32()),
                ),
                (
                    "infer__evo2_7b__template_1kb_7b_features__intermediate_embedding__block26_mlp_out__seq_mean"
                ): pa.array(
                    [[0.7, 0.6], [0.5, 0.4], [0.3, 0.2]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_7b__template_1kb_7b_features__output_layer_mean__seq_mean"): pa.array(
                    [[1.9, 1.8], [1.7, 1.6], [1.5, 1.4]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token"): pa.array(
                    [-5.9, -5.6, -6.2],
                    type=pa.float32(),
                ),
                ("infer__evo2_7b__template_1kb_7b_features__output_layer_mean__anchor_mean"): pa.array(
                    [[2.0, 1.9], [1.8, 1.7], [1.6, 1.5]],
                    type=pa.list_(pa.float32()),
                ),
                (
                    "infer__evo2_20b__template_1kb_20b_features__intermediate_embedding__block23_mlp_out__seq_mean"
                ): pa.array(
                    [[1.6, 1.5], [1.4, 1.3], [1.2, 1.1]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_20b__template_1kb_20b_features__output_layer_mean__seq_mean"): pa.array(
                    [[2.4, 2.3], [2.2, 2.1], [2.0, 1.9]],
                    type=pa.list_(pa.float32()),
                ),
                ("infer__evo2_20b__template_1kb_20b_features__output_layer_mean__anchor_mean"): pa.array(
                    [[2.5, 2.4], [2.3, 2.2], [2.1, 2.0]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )
    context_dataset.write_overlay_part(
        "densegen",
        pa.table(
            {
                "id": context_ids,
                "densegen__plan": ["ethanol__sigma70_b", "ciprofloxacin__sigma70_c", "background_only__sigma70_d"],
                "densegen__required_regulators": ["cpxR", "lexA", "background"],
                "densegen__used_tfbs_detail": [
                    '[{"part_kind":"fixed_element","spacer_length":17}]',
                    '[{"part_kind":"fixed_element","spacer_length":16}]',
                    '[{"part_kind":"fixed_element","spacer_length":18}]',
                ],
            }
        ),
        key="id",
    )
    context_dataset.write_overlay_part(
        "usr_label",
        pa.table({"id": context_ids, "usr_label__primary": ["spyP", "sulAp", "J23105"]}),
        key="id",
    )
    context_dataset.write_overlay_part(
        "construct",
        pa.table(
            {
                "id": context_ids,
                "construct__anchor_id": anchor_ids,
                "construct__context_id": ["ctx0", "ctx1", "ctx2"],
                "construct__template_id": ["tpl0", "tpl1", "tpl2"],
            }
        ),
        key="id",
    )
    return usr_root


def test_workspace_where_defaults_to_cwd_when_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LATENTDNA_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["workspace", "where"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_root: {tmp_path.resolve()}" in output
    assert "workspace_root_source: cwd" in output


def test_workspace_init_creates_default_layout_and_config(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--workspace",
            workspace_dir.as_posix(),
            "--template",
            "minimal",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "config.yaml").is_file()
    assert (workspace_dir / "README.md").is_file()
    assert (workspace_dir / "outputs").is_dir()

    payload = yaml.safe_load((workspace_dir / "config.yaml").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "latentdna.workspace.v1"
    assert payload["workspace"]["id"] == "demo_latentdna"

    output = result.stdout or ""
    assert "template: minimal" in output
    assert "latentdna validate workspace" in output


def test_workspace_init_from_study_dir_hydrates_promoter_reference_margin_template(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    study_dir = tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth"
    study_dir.mkdir(parents=True)
    (study_dir / "campaign.yaml").write_text("version: 1\nsteps: []\n", encoding="utf-8")
    (study_dir / "status.md").write_text("## test study\n", encoding="utf-8")
    (study_dir / "ops.study.yaml").write_text("version: 2\nstudy_id: stress_ethanol_cipro_growth\n", encoding="utf-8")
    (study_dir / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "stress_ethanol_cipro_growth",
                "datasets": [
                    {
                        "role": "merged_anchor_source",
                        "dataset": "promoter/test_anchor",
                        "usr_root": usr_root.as_posix(),
                    },
                    {
                        "role": "construct_context",
                        "dataset": "promoter/test_contexts",
                        "usr_root": usr_root.as_posix(),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workspace_dir = tmp_path / "study_bound_latentdna"
    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--workspace",
            workspace_dir.as_posix(),
            "--template",
            "promoter_reference_margin_benchmark",
            "--from-study-dir",
            study_dir.as_posix(),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = yaml.safe_load(result.stdout)
    assert payload["schema_version"] == "latentdna.command_result.v1"
    assert payload["command"] == "workspace init"
    assert payload["artifact_kind"] == "workspace"
    assert payload["artifact_id"] == "study_bound_latentdna"

    config_payload = yaml.safe_load((workspace_dir / "config.yaml").read_text(encoding="utf-8"))
    expected_usr_root = Path(relpath(usr_root.resolve(), workspace_dir.resolve())).as_posix()
    assert config_payload["sources"]["anchor_60bp"]["root"] == expected_usr_root
    assert config_payload["sources"]["anchor_60bp"]["dataset"] == "promoter/test_anchor"
    assert config_payload["sources"]["full_context_1kb"]["root"] == expected_usr_root
    assert config_payload["sources"]["full_context_1kb"]["dataset"] == "promoter/test_contexts"
    assert config_payload["study_binding"]["study_id"] == "stress_ethanol_cipro_growth"
    assert config_payload["study_binding"]["docs_root"] == "src/dnadesign/studies/stress_ethanol_cipro_growth"


def test_workspace_show_reports_workspace_summary(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "metadata": {"include": []},
                "views": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["workspace", "show", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "workspace_id: demo_latentdna" in output
    assert f"workspace_dir: {workspace_dir.resolve()}" in output


def test_validate_workspace_deep_checks_declared_source_schema(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    data_dir = workspace_dir / "inputs"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b"],
                "subject_id": ["s1", "s2"],
                "densegen__plan": ["plan_a", "plan_b"],
                "usr_label__primary": ["spyP", "sulAp"],
                "embedding": [[0.0, 1.0], [1.0, 0.0]],
            }
        ),
        data_dir / "anchor60.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "parquet",
                        "path": "inputs/anchor60.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": ["densegen__plan"]},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
                "landmarks": {
                    "spy_p": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "centroid"},
                    }
                },
                "cohorts": {
                    "plan": {
                        "kind": "column",
                        "source": "anchor60",
                        "column": "densegen__plan",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = yaml.safe_load(result.stdout)
    assert payload["status"] == "ok"
    assert payload["deep"] is True
    assert payload["source_details"][0]["source_id"] == "anchor60"
    assert payload["view_details"][0]["vector_column"] == "embedding"
    assert payload["landmark_details"][0]["selector_column"] == "usr_label__primary"
    assert payload["cohort_details"][0]["column"] == "densegen__plan"


def test_inspect_and_validate_workspace_use_usr_overlay_columns(tmp_path: Path) -> None:
    usr_root = _build_overlay_usr_sources(tmp_path)
    workspace_dir = tmp_path / "overlay_latentdna"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "overlay_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "anchor",
                        "record_key": "id",
                        "subject_key": "id",
                    },
                    "ctx1k": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "contexts",
                        "record_key": "id",
                        "subject_key": "construct__anchor_id",
                        "context_key": "construct__context_id",
                    },
                },
                "metadata": {"include": []},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "infer__x_representation"},
                        "coordinate_space_id": "shared_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    },
                    "intermediate_embedding_20b_full_context_1kb": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "infer__x_representation"},
                        "coordinate_space_id": "shared_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    inspect_result = _RUNNER.invoke(
        app,
        ["inspect", "source", "ctx1k", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout
    inspect_payload = yaml.safe_load(inspect_result.stdout)
    assert "construct__anchor_id" in inspect_payload["data"]["columns"]
    assert "construct__context_id" in inspect_payload["data"]["columns"]
    assert "infer__x_representation" in inspect_payload["data"]["vector_columns"]

    validate_result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )
    assert validate_result.exit_code == 0, validate_result.stdout
    validate_payload = yaml.safe_load(validate_result.stdout)
    assert validate_payload["status"] == "ok"
    ctx_detail = next(detail for detail in validate_payload["source_details"] if detail["source_id"] == "ctx1k")
    assert ctx_detail["required_columns"] == ["id", "construct__anchor_id", "construct__context_id"]


def test_workspace_init_promoter_reference_margin_template_validates_with_realish_promoter_study_data(
    tmp_path: Path,
) -> None:
    usr_root = _build_committee_usr_sources(tmp_path)
    study_dir = tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth"
    study_dir.mkdir(parents=True)
    (study_dir / "campaign.yaml").write_text("version: 1\nsteps: []\n", encoding="utf-8")
    (study_dir / "status.md").write_text("## test study\n", encoding="utf-8")
    (study_dir / "ops.study.yaml").write_text("version: 2\nstudy_id: stress_ethanol_cipro_growth\n", encoding="utf-8")
    (study_dir / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "stress_ethanol_cipro_growth",
                "datasets": [
                    {
                        "role": "merged_anchor_source",
                        "dataset": "promoter/test_anchor",
                        "usr_root": usr_root.as_posix(),
                    },
                    {
                        "role": "construct_context",
                        "dataset": "promoter/test_contexts",
                        "usr_root": usr_root.as_posix(),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workspace_dir = tmp_path / "study_bound_realish_latentdna"
    init_result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--workspace",
            workspace_dir.as_posix(),
            "--template",
            "promoter_reference_margin_benchmark",
            "--from-study-dir",
            study_dir.as_posix(),
            "--json",
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    validate_result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )
    assert validate_result.exit_code == 0, validate_result.stdout
    validate_payload = yaml.safe_load(validate_result.stdout)
    assert validate_payload["status"] == "ok"
    z7_view = next(
        detail
        for detail in validate_payload["view_details"]
        if detail["view_id"] == "intermediate_embedding_7b_anchor_60bp"
    )
    assert z7_view["vector_column"].endswith("block26_mlp_out__seq_mean")

    materialize_result = _RUNNER.invoke(
        app,
        [
            "view",
            "materialize",
            "intermediate_embedding_7b_anchor_60bp",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    post_materialize_validate_result = _RUNNER.invoke(
        app,
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    )
    assert post_materialize_validate_result.exit_code == 0, post_materialize_validate_result.stdout
    post_materialize_validate_payload = yaml.safe_load(post_materialize_validate_result.stdout)
    assert post_materialize_validate_payload["status"] == "ok"
    materialized_view_detail = next(
        detail
        for detail in post_materialize_validate_payload["view_details"]
        if detail["view_id"] == "intermediate_embedding_7b_anchor_60bp"
    )
    assert materialized_view_detail["materialized"] is True
    assert "design_family" in materialized_view_detail["materialized_row_columns"]
    assert "sig35_variant" in materialized_view_detail["materialized_row_columns"]
    assert "densegen__plan" not in materialized_view_detail["materialized_row_columns"]

    sample_result = _RUNNER.invoke(
        app,
        [
            "sample",
            "build",
            "reference_margin_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "intermediate_embedding_7b_anchor_60bp",
            "--strategy",
            "stratified",
            "--group-column",
            "design_family",
            "--n",
            "2",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert sample_result.exit_code == 0, sample_result.stdout


def test_validate_workspace_accepts_minimal_config(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "parquet",
                        "path": "inputs/anchor60.parquet",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": ["cohort"]},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--json"])

    assert result.exit_code == 0, result.stdout
    assert '"status":"ok"' in (result.stdout or "").replace(" ", "")


def test_workspace_refresh_removes_local_outputs_and_preserves_sources(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    source_path = tmp_path / "source_data" / "anchor.parquet"
    source_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": ["a"], "subject_id": ["a"], "embedding": [[0.1, 0.2]]}),
        source_path,
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor60": {
                        "kind": "parquet",
                        "path": source_path.as_posix(),
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": []},
                "views": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "outputs" / "views" / "z20_60").mkdir(parents=True)
    (workspace_dir / "outputs" / "views" / "z20_60" / "manifest.json").write_text("{}", encoding="utf-8")
    (workspace_dir / "outputs" / "plots" / "atlas").mkdir(parents=True)
    (workspace_dir / "outputs" / "plots" / "index.json").write_text("{}", encoding="utf-8")
    (workspace_dir / "outputs" / "catalog.json").write_text("{}", encoding="utf-8")
    (workspace_dir / "outputs" / "runs" / "_staging" / "scratch").mkdir(parents=True)
    (workspace_dir / "outputs" / "runs" / "_staging" / "scratch" / "marker.txt").write_text("tmp", encoding="utf-8")
    (workspace_dir / "outputs" / "status").mkdir(parents=True)
    (workspace_dir / "outputs" / "status" / "workspace_snapshot.json").write_text("{}", encoding="utf-8")
    (workspace_dir / "outputs" / "logs" / "audit").mkdir(parents=True)
    (workspace_dir / "outputs" / "logs" / "audit" / "event.json").write_text("{}", encoding="utf-8")
    result = _RUNNER.invoke(app, ["workspace", "refresh", "--workspace", workspace_dir.as_posix(), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["command"] == "workspace refresh"
    assert payload["inputs"]["targets"] == [
        "agreements",
        "alignments",
        "clusters",
        "distances",
        "enrichments",
        "exports",
        "neighbors",
        "notebooks",
        "plots",
        "projections",
        "reduced_views",
        "reducers",
        "samples",
        "scalars",
        "snapshots",
        "views",
        "runs",
        "logs",
        "catalog",
        "status",
    ]
    assert payload["post_refresh_validation"] == "ok"
    assert not (workspace_dir / "outputs" / "views").exists()
    assert not (workspace_dir / "outputs" / "plots").exists()
    assert not (workspace_dir / "outputs" / "catalog.json").exists()
    assert not (workspace_dir / "outputs" / "runs").exists()
    assert not (workspace_dir / "outputs" / "logs").exists()
    assert not (workspace_dir / "outputs" / "status").exists()
    assert (workspace_dir / "outputs").is_dir()
    assert source_path.is_file()
    assert source_path.as_posix() in payload["protected_paths"]


def test_workspace_refresh_dry_run_leaves_outputs_in_place(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "metadata": {"include": []},
                "views": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "outputs" / "notebooks" / "latent_geometry_browser").mkdir(parents=True)
    (workspace_dir / "outputs" / "notebooks" / "latent_geometry_browser" / "notebook.py").write_text(
        "print('browser')", encoding="utf-8"
    )

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "refresh",
            "--workspace",
            workspace_dir.as_posix(),
            "--target",
            "notebooks",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["post_refresh_validation"] == "skipped"
    assert payload["removed_paths"] == []
    assert (workspace_dir / "outputs" / "notebooks" / "latent_geometry_browser").exists()


def test_workspace_refresh_rejects_noncanonical_output_root(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs/nested"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "metadata": {"include": []},
                "views": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["workspace", "refresh", "--workspace", workspace_dir.as_posix(), "--json"])

    assert result.exit_code != 0
    assert "workspace output_root must resolve to" in result.stdout


def test_workspace_snapshot_emits_machine_readable_status_contract(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "demo_latentdna"
    workspace_dir.mkdir(parents=True)
    usr_root = tmp_path / "usr_root"
    dataset_dir = usr_root / "promoter" / "demo_anchor_set"
    dataset_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "anchor_01",
                    "subject_id": "anchor_01",
                    "usr_label__primary": "spyP",
                    "design_family": "ethanol",
                    "sig35_variant": "b",
                    "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean": [
                        1.0,
                        0.0,
                    ],
                },
                {
                    "id": "anchor_02",
                    "subject_id": "anchor_02",
                    "usr_label__primary": "sulAp",
                    "design_family": "cipro",
                    "sig35_variant": "c",
                    "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__block23_mlp_out__seq_mean": [
                        0.0,
                        1.0,
                    ],
                },
            ]
        ),
        dataset_dir / "records.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "demo_latentdna", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {
                    "anchor_60bp": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_anchor_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    }
                },
                "metadata": {"include": ["usr_label__primary", "design_family", "sig35_variant"]},
                "views": {
                    "intermediate_embedding_20b_anchor_60bp": {
                        "source": "anchor_60bp",
                        "vector": {
                            "kind": "column",
                            "name": (
                                "infer__evo2_20b__anchor_only_20b_features__intermediate_embedding__"
                                "block23_mlp_out__seq_mean"
                            ),
                        },
                        "coordinate_space_id": "evo2_20b_intermediate_block23_mlp_out",
                        "tags": {
                            "encoder": "evo2",
                            "model": "20b",
                            "family": "intermediate_embedding",
                            "scope": "anchor_60bp",
                        },
                    }
                },
                "deliverables": {
                    "dataset_overview": {
                        "title": "Dataset overview",
                        "section": "Dataset",
                        "question": "What was analyzed?",
                        "summary": "Compact dataset inventory for the active workspace.",
                        "recipe": "dataset_overview_recipe",
                        "requires": {"views": ["intermediate_embedding_20b_anchor_60bp"]},
                        "outputs": {"views": ["intermediate_embedding_20b_anchor_60bp"]},
                        "docs_refs": [],
                        "acceptance_checks": [],
                    }
                },
                "notebooks": {
                    "latent_geometry_browser": {
                        "kind": "workspace",
                        "title": "Latent geometry browser",
                        "description": "Read-only geometry browser.",
                        "default_deliverable": "dataset_overview",
                    }
                },
                "recipes": {
                    "dataset_overview_recipe": {
                        "steps": [
                            {
                                "id": "materialize_anchor",
                                "op": "view.materialize",
                                "params": {"view": "intermediate_embedding_20b_anchor_60bp"},
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "snapshot",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["workspace_id"] == "demo_latentdna"
    assert payload["sources"]["anchor_60bp"]["dataset_id"] == "promoter/demo_anchor_set"
    assert payload["sources"]["anchor_60bp"]["row_count"] == 2
    assert payload["model_families"] == ["evo2_20b"]
    assert payload["canonical_views"] == ["intermediate_embedding_20b_anchor_60bp"]
    assert payload["browser"]["default_geometry_ids"] == ["intermediate_embedding_20b_anchor_60bp"]
    assert payload["decision_ladder"] == ["dataset_overview"]
