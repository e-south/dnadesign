"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_notebook_generate.py

Regression tests for CLI notebook generate OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import (
    write_campaign_yaml,
    write_ledger,
    write_ledger_labels,
    write_records,
)


def _literal_assignment_value(text: str, name: str) -> object:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Generated notebook did not assign {name!r}.")


def test_notebook_generate_smoke(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    out_path = workdir / "notebooks" / "opal_demo_analysis.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    smoke_checked: list[Path] = []
    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: smoke_checked.append(Path(path))
        or {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--out",
            str(out_path),
            "--no-validate",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out_path.exists()
    assert smoke_checked == [out_path]

    txt = out_path.read_text()
    assert "marimo.App" in txt
    assert "build_campaign_set_notebook_view_model" in txt
    assert 'label="Campaign"' in txt
    assert "opal" in txt.lower()
    assert "mo.ui.table" in txt
    assert "__generated_with" in txt
    assert 'marimo.App(width="medium")' in txt
    assert 'marimo.App(width="full")' not in txt

    # Optional import check if marimo is installed
    if importlib.util.find_spec("marimo") is not None:
        spec = importlib.util.spec_from_file_location("opal_campaign_nb", out_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "app")


def test_notebook_generate_json_next_commands_are_app_first(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )
    out_path = workdir / "notebooks" / "opal_demo_analysis.py"

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--out",
            str(out_path),
            "--no-validate",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["next_commands"]["run"] == f"uv run opal notebook run -c {campaign} --path {out_path}"
    assert payload["next_commands"]["edit"] == f"uv run opal notebook edit -c {campaign} --path {out_path}"
    assert payload["next_commands"]["marimo_check"] == f"uv run marimo check {out_path}"


def test_notebook_generate_allows_pre_run_campaign_by_default(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code == 0, res.output
    out_path = workdir / "notebooks" / "opal_demo_analysis.py"
    assert out_path.exists()
    text = out_path.read_text()
    helper_text = Path("src/dnadesign/opal/src/analysis/notebook_components/visual_panel.py").read_text()
    assert "build_campaign_set_notebook_view_model" in text
    assert "No OPAL plot deliverables are available" in helper_text


def test_notebook_generate_rejects_unknown_round(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger_labels(workdir, round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--round",
            "7",
        ],
    )
    assert res.exit_code != 0, res.output
    assert "Available rounds" in res.output


def test_notebook_generate_with_name(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--name",
            "custom_demo",
            "--no-validate",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out_path = workdir / "notebooks" / "custom_demo.py"
    assert out_path.exists()


def test_notebook_generate_campaign_set_with_repeated_campaign(tmp_path: Path, monkeypatch) -> None:
    campaigns = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records = workdir / "records.parquet"
        write_records(records, slug=slug)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(
            campaign,
            workdir=workdir,
            records_path=records,
            slug=slug,
        )
        campaigns.append(campaign)

    out_path = tmp_path / "campaign_set.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    smoke_checked: list[Path] = []
    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: smoke_checked.append(Path(path))
        or {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--out",
            str(out_path),
            "--no-validate",
        ],
    )

    assert res.exit_code == 0, res.output
    assert out_path.exists()
    assert smoke_checked == [out_path]
    text = out_path.read_text()
    assert "# OPAL Review Notebook" not in text
    assert "selected_campaign_title_md = mo.md(_header_lines[0])" in text
    assert "selected_campaign_context_panel = mo.accordion(" in text
    assert "build_campaign_set_notebook_view_model" in text
    assert 'label="Campaign"' in text
    assert 'label="Deliverable"' in text


def test_notebook_generate_campaign_set_accepts_all_round_scope(tmp_path: Path, monkeypatch) -> None:
    campaigns = []
    for slug in ["campaign_a", "campaign_b"]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records = workdir / "records.parquet"
        write_records(records, slug=slug)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug=slug)
        write_ledger(workdir, run_id=f"{slug}-run-0", round_index=0)
        campaigns.append(campaign)

    out_path = tmp_path / "campaign_set_all.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--round",
            "all",
            "--out",
            str(out_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "selected_round_selector = 'all'" in out_path.read_text()


def test_notebook_generate_campaign_set_accepts_collection_manifest(tmp_path: Path, monkeypatch) -> None:
    campaigns = []
    for slug, oracle_kind in {"campaign_positive": "positive", "campaign_null": "null"}.items():
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records = workdir / "records.parquet"
        write_records(records, slug=slug)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug=slug)
        payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
        payload["campaign"]["metadata"] = {
            "target": "cipro",
            "label_oracle_kind": oracle_kind,
            "label_family_id": "densegen_plan_logic4",
            "label_split_id": "random_id",
            "seed": 7,
        }
        campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        write_ledger(workdir, run_id=f"{slug}-run-0", round_index=0)
        _write_score_selected_plot_fixture(workdir, slug=slug, value=0.4 if oracle_kind == "positive" else 0.1)
        campaigns.append(campaign)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "cli_fixture",
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
                        "role_dimension": "label_oracle_kind",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
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
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "campaign_set_collection.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--collection",
            str(collection_path),
            "--out",
            str(out_path),
        ],
    )

    assert res.exit_code == 0, res.output
    text = out_path.read_text()
    assert _literal_assignment_value(text, "collection_manifest_path") == str(collection_path)
    assert 'label="View"' in text
    assert "view_mode_ui = mo.ui.dropdown(" in text
    assert "collection_visual_index_path" in text
    collection_visual_index = out_path.parent / "collection_visuals" / "collection_visual_manifest.json"
    assert collection_visual_index.exists()


def test_notebook_generate_campaign_set_accepts_existing_collection_visual_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaigns: list[Path] = []
    for oracle_kind in ("positive", "null"):
        slug = f"tfbs_lexa_{oracle_kind}"
        workdir = tmp_path / slug
        workdir.mkdir(parents=True)
        records = workdir / "records.parquet"
        write_records(records)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(
            campaign,
            workdir=workdir,
            records_path=records,
            slug=slug,
        )
        write_ledger(workdir, run_id=f"{slug}-run-0", round_index=0)
        campaigns.append(campaign)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "cli_fixture_registered",
                "dimensions": [{"id": "target"}],
                "relationships": [],
                "comparison_views": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    visual_index_path = tmp_path / "registered_visuals" / "collection_visual_manifest.json"
    visual_index_path.parent.mkdir()
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v1",
                "generated_at": "2026-06-02T00:00:00+00:00",
                "collection_id": "cli_fixture_registered",
                "output_dir": str(visual_index_path.parent),
                "comparison_set_count": 0,
                "comparison_sets": [],
                "visual_count": 0,
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "campaign_set_registered_visuals.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--collection",
            str(collection_path),
            "--collection-visual-index",
            str(visual_index_path),
            "--no-materialize-collection-visuals",
            "--out",
            str(out_path),
        ],
    )

    assert res.exit_code == 0, res.output
    text = out_path.read_text()
    assert _literal_assignment_value(text, "collection_visual_index_path") == str(visual_index_path)
    assert visual_index_path.exists()
    assert not (out_path.parent / "collection_visuals" / "collection_visual_manifest.json").exists()


def test_notebook_generate_campaign_set_rejects_mismatched_collection_visual_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaigns: list[Path] = []
    for oracle_kind in ("positive", "null"):
        slug = f"tfbs_lexa_{oracle_kind}"
        workdir = tmp_path / slug
        workdir.mkdir(parents=True)
        records = workdir / "records.parquet"
        write_records(records)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug=slug)
        write_ledger(workdir, run_id=f"{slug}-run-0", round_index=0)
        campaigns.append(campaign)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "cli_fixture_registered",
                "dimensions": [{"id": "target"}],
                "relationships": [],
                "comparison_views": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    visual_index_path = tmp_path / "registered_visuals" / "collection_visual_manifest.json"
    visual_index_path.parent.mkdir()
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v1",
                "generated_at": "2026-06-02T00:00:00+00:00",
                "collection_id": "wrong_collection",
                "output_dir": str(visual_index_path.parent),
                "comparison_set_count": 0,
                "comparison_sets": [],
                "visual_count": 0,
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "campaign_set_registered_visuals.py"
    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    smoke_checked: list[Path] = []
    monkeypatch.setattr(
        notebook_cmd,
        "smoke_check_notebook",
        lambda path, *, run_marimo_check=True: smoke_checked.append(Path(path))
        or {"python_parse_ok": True, "marimo_check_ok": True},
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--collection",
            str(collection_path),
            "--collection-visual-index",
            str(visual_index_path),
            "--no-materialize-collection-visuals",
            "--out",
            str(out_path),
        ],
    )

    assert res.exit_code != 0, res.output
    assert "collection_id mismatch" in res.output
    assert not out_path.exists()
    assert smoke_checked == []


def test_notebook_generate_existing_campaign_set_name_does_not_materialize_collection_visuals_without_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaigns = []
    for slug, oracle_kind in [("campaign_positive", "positive"), ("campaign_null", "null")]:
        workdir = tmp_path / slug
        workdir.mkdir(parents=True, exist_ok=True)
        records = workdir / "records.parquet"
        write_records(records, slug=slug)
        campaign = workdir / "campaign.yaml"
        write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug=slug)
        payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
        payload["campaign"]["metadata"] = {
            "target": "cipro",
            "label_oracle_kind": oracle_kind,
            "label_family_id": "densegen_plan_logic4",
            "label_split_id": "random_id",
            "seed": 7,
        }
        campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        campaigns.append(campaign)
    collection_path = tmp_path / "campaign_collection.yaml"
    collection_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "opal.campaign_collection.v2",
                "collection_id": "cli_fixture",
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
                        "role_dimension": "label_oracle_kind",
                        "left_role": "positive",
                        "right_role": "null",
                        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                    }
                ],
                "comparison_views": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "campaign_set_collection.py"
    out_path.write_text("import marimo\n", encoding="utf-8")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    materialized_dirs: list[Path] = []

    def _materialize(*args, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "collection_visual_manifest.json").write_text("{}", encoding="utf-8")
        materialized_dirs.append(output_dir)
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(notebook_cmd, "materialize_campaign_set_collection_visuals", _materialize)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "--campaign",
            str(campaigns[0]),
            "--campaign",
            str(campaigns[1]),
            "--collection",
            str(collection_path),
            "--out",
            str(out_path),
            "--no-validate",
        ],
    )

    assert res.exit_code != 0, res.output
    assert "already exists" in res.output.lower()
    assert materialized_dirs == []
    assert not (out_path.parent / "collection_visuals" / "collection_visual_manifest.json").exists()


def test_notebook_generate_existing_name_requires_force(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    existing = workdir / "notebooks" / "opal_demo_analysis.py"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("import marimo\n")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--no-validate",
        ],
    )
    assert res.exit_code != 0, res.output
    lowered = res.output.lower()
    assert "already exists" in lowered
    assert "--force" in lowered
    assert "--name" in lowered


def test_notebook_run_selects_single_notebook(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    nb_path = workdir / "notebooks" / "only.py"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text("import marimo\n")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    calls: list[dict[str, object]] = []

    def _fake_launch(**kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(notebook_cmd, "launch_marimo_notebook", _fake_launch)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
            "--headless",
            "--host",
            "127.0.0.1",
            "--port",
            "28510",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert calls == [
        {
            "mode": "run",
            "notebook_path": nb_path,
            "host": "127.0.0.1",
            "port": 28510,
            "headless": True,
        }
    ]


def test_notebook_edit_selects_single_notebook(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    nb_path = workdir / "notebooks" / "only.py"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text("import marimo\n")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    calls: list[dict[str, object]] = []

    def _fake_launch(**kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(notebook_cmd, "launch_marimo_notebook", _fake_launch)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "edit",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert calls == [
        {
            "mode": "edit",
            "notebook_path": nb_path,
            "host": None,
            "port": None,
            "headless": False,
        }
    ]


def test_notebook_run_requires_notebook(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "no notebooks found" in res.output.lower()


def test_notebook_run_multiple_notebooks_non_tty(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    nb_dir = workdir / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    (nb_dir / "one.py").write_text("import marimo\n")
    (nb_dir / "two.py").write_text("import marimo\n")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "run",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code != 0, res.output
    lowered = res.output.lower()
    assert "multiple notebooks found" in lowered
    assert "0:" in lowered
    assert "1:" in lowered
    assert "--path" in lowered


def test_notebook_generate_requires_records_even_no_validate(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger_labels(workdir, round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--no-validate",
        ],
    )
    assert res.exit_code != 0, res.output
    assert "records.parquet not found" in res.output


def test_notebook_root_rich_output(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
    )

    nb_dir = workdir / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    (nb_dir / "one.py").write_text("import marimo\n")
    (nb_dir / "two.py").write_text("import marimo\n")

    import dnadesign.opal.src.cli.commands.notebook as notebook_cmd

    calls: list[object] = []

    monkeypatch.setattr(notebook_cmd, "tui_enabled", lambda: True)
    monkeypatch.setattr(
        notebook_cmd,
        "print_rich",
        lambda obj: calls.append(obj) or True,
    )

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "-c",
            str(campaign),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls


def test_notebook_rich_tables_use_rounded_box() -> None:
    from rich import box

    import dnadesign.opal.src.cli.commands.notebook_support as notebook_support

    kv_table = notebook_support.rich_kv_table("Notebook", {"Key": "Value"})
    list_table = notebook_support.rich_list_table("Notebooks", ["0: one.py"])

    assert kv_table.box == box.ROUNDED
    assert list_table.box == box.ROUNDED
    assert str(kv_table.border_style) == "cyan"
    assert str(list_table.border_style) == "cyan"


def _write_score_selected_plot_fixture(workdir: Path, *, slug: str, value: float) -> None:
    plots_dir = workdir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
    tidy_path.write_text(
        "round,cohort,metric,summary,value\n"
        f"0,selected,pred__score_selected,mean,{value}\n"
        "0,selected,pred__score_selected,count,12\n",
        encoding="utf-8",
    )
    manifest_path = plots_dir / "score_selected_over_rounds_rall.manifest.json"
    manifest = {
        "schema_version": "opal.plot_artifact.v1",
        "plot_id": f"{slug}_score_selected_over_rounds",
        "name": "score_selected_over_rounds",
        "kind": "metric_over_rounds",
        "status": "written",
        "started_at": "2026-05-26T00:00:00+00:00",
        "generated_at": "2026-05-26T00:00:00+00:00",
        "run_id": f"{slug}-run-0",
        "rounds": "all",
        "params": {},
        "inputs": [],
        "outputs": [{"role": "tidy_csv", "path": str(tidy_path), "exists": True}],
        "tidy_csv": str(tidy_path),
        "manifest_path": str(manifest_path),
        "metadata": {},
        "caption": "Selected score over rounds",
        "review_purpose": "Selected score over rounds",
        "quality": {},
        "freshness": {"status": "fresh"},
        "warnings": [],
        "error": None,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (plots_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "opal.plot_manifest_index.v1",
                "generated_at": "2026-05-26T00:00:00+00:00",
                "output_dir": str(plots_dir),
                "plot_count": 1,
                "manifests": [manifest],
            }
        ),
        encoding="utf-8",
    )
