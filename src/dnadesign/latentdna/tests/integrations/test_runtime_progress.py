"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/integrations/test_runtime_progress.py

Runtime progress and reference-set contract tests for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.result import CommandResult
from dnadesign.latentdna.src.services import progress_service, recipe_service
from dnadesign.latentdna.src.services.catalog_service import workspace_catalog, workspace_catalog_from_context
from dnadesign.latentdna.src.services.plot_service import render_plot, write_plot_index
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config

_RUNNER = CliRunner()


def _write_workspace_config(workspace_dir: Path) -> None:
    semantics_dir = workspace_dir / "plot_semantics"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    (semantics_dir / "atlas.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "atlas",
                "question": "Do both projection panels retain the required reference set?",
                "decision_role": "debug",
                "encoding": "Two-panel projection grid colored by family.",
                "scope": "Full population.",
                "guardrails": [
                    "Projection geometry is descriptive only.",
                ],
                "caption": "Reference-set completeness check for projection-grid rendering.",
                "alt_text": "Two-panel projection grid used to validate reference-set completeness rules.",
                "preprocessing_md": "Fixture semantics do not declare additional preprocessing.",
                "math_md": "Fixture semantics do not declare a mathematical definition.",
                "rationale_md": "Fixture semantics exist only to validate runtime progress reporting.",
                "limitations_md": "Fixture semantics are not a study-facing scientific contract.",
                "failure_modes_md": "Replace fixture semantics before using the plot outside tests.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "reference_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "pdf", "png"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "metadata": {"include": []},
                "reference_sets": {
                    "promoter_wt_core": {
                        "ids": ["spyP", "sulAp"],
                        "match_column": "usr_label__primary",
                        "label_column": "usr_label__primary",
                        "label_mode": "label_and_highlight",
                    }
                },
                "plots": {
                    "atlas": {
                        "kind": "projection_grid",
                        "semantics_ref": "./plot_semantics/atlas.yaml",
                        "projections": ["p1", "p2"],
                        "color_column": "family",
                        "annotation": {
                            "reference_set": "promoter_wt_core",
                            "require_in_every_panel": True,
                            "missing_policy": "fail",
                            "collision_policy": "repel_then_callout",
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_projection(workspace_dir: Path, projection_id: str, rows: list[dict[str, object]]) -> None:
    projection_dir = workspace_dir / "outputs" / "projections" / projection_id
    projection_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), projection_dir / "coords.parquet")
    (projection_dir / "manifest.json").write_text(
        json.dumps({"artifact_id": projection_id, "artifact_kind": "projection", "status": "ok"}),
        encoding="utf-8",
    )


def _write_progress_workspace_config(workspace_dir: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "progress_demo", "output_root": "./outputs"},
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
                "metadata": {"include": []},
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "embedding"},
                        "coordinate_space_id": "demo_space",
                        "tags": {"model": "demo"},
                        "role": "primary",
                    }
                },
                "recipes": {
                    "slow_recipe": {
                        "steps": [
                            {
                                "id": "slow_step",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_projection_grid_fails_when_reference_set_missing_from_any_panel(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    _write_projection(
        workspace_dir,
        "p1",
        [
            {"x": 0.0, "y": 0.0, "usr_label__primary": "spyP", "family": "control"},
            {"x": 1.0, "y": 0.0, "usr_label__primary": "sulAp", "family": "control"},
        ],
    )
    _write_projection(
        workspace_dir,
        "p2",
        [
            {"x": 0.0, "y": 1.0, "usr_label__primary": "spyP", "family": "control"},
            {"x": 1.0, "y": 1.0, "usr_label__primary": "dense_01", "family": "designed"},
        ],
    )

    with pytest.raises(ContractViolationError, match="missing required ids"):
        render_plot(
            workspace_dir,
            "atlas",
            kind=None,
            projection_ids=[],
            panel_titles=[],
            enrichment_id=None,
            distance_id=None,
            scalar_id=None,
            agreement_id=None,
            reducer_id=None,
            left_cluster_id=None,
            right_cluster_id=None,
            value_column=None,
            x_column=None,
            y_column=None,
            color_column=None,
            render_mode=None,
            label_column=None,
            label_values=[],
        )


def test_projection_grid_records_reference_set_completeness_and_persists_pdf(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    complete_rows = [
        {"x": 0.0, "y": 0.0, "usr_label__primary": "spyP", "family": "control"},
        {"x": 1.0, "y": 0.0, "usr_label__primary": "sulAp", "family": "control"},
    ]
    _write_projection(workspace_dir, "p1", complete_rows)
    _write_projection(workspace_dir, "p2", complete_rows)

    result = render_plot(
        workspace_dir,
        "atlas",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        agreement_id=None,
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert result.outputs == [(workspace_dir / "outputs" / "plots" / "atlas").as_posix()]
    manifest = json.loads((workspace_dir / "outputs" / "plots" / "atlas" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["reference_set_complete"] is True
    assert (workspace_dir / "outputs" / "plots" / "atlas" / "plot.pdf").is_file()


def test_projection_grid_resolves_selector_reference_set_in_every_panel(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_workspace_config(workspace_dir)
    config_path = workspace_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["reference_sets"] = {
        "reference_rows": {
            "match_column": "usr_label__primary",
            "label_column": "usr_label__primary",
            "where": [
                {
                    "column": "source_family",
                    "in_values": ["reference_source", "construct_derived"],
                }
            ],
        }
    }
    config["plots"]["atlas"]["annotation"]["reference_set"] = "reference_rows"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    complete_rows = [
        {
            "x": 0.0,
            "y": 0.0,
            "usr_label__primary": "J23105",
            "source_family": "reference_source",
            "family": "control",
        },
        {
            "x": 1.0,
            "y": 0.0,
            "usr_label__primary": "W1_core60",
            "source_family": "construct_derived",
            "family": "control",
        },
        {
            "x": 0.5,
            "y": 1.0,
            "usr_label__primary": "dense_01",
            "source_family": "densegen_generated",
            "family": "designed",
        },
    ]
    _write_projection(workspace_dir, "p1", complete_rows)
    _write_projection(workspace_dir, "p2", complete_rows)

    render_plot(
        workspace_dir,
        "atlas",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        agreement_id=None,
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    manifest = json.loads((workspace_dir / "outputs" / "plots" / "atlas" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["reference_set_complete"] is True
    assert manifest["stats"]["reference_panels"]["p1"]["expected_ids"] == ["J23105", "W1_core60"]
    assert manifest["stats"]["reference_panels"]["p2"]["matched_ids"] == ["J23105", "W1_core60"]


def test_recipe_run_progress_json_emits_heartbeat_and_final_result(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_progress_workspace_config(workspace_dir)

    def _slow_step(workspace: str | Path, params: dict[str, object], *, force: bool) -> CommandResult:
        del params, force
        time.sleep(0.05)
        return CommandResult(
            command="view materialize",
            workspace_id="progress_demo",
            status="ok",
            artifact_kind="view",
            artifact_id="z20_60",
            outputs=[(Path(workspace) / "outputs" / "views" / "z20_60").as_posix()],
            inputs={"view": "z20_60"},
        )

    monkeypatch.setitem(recipe_service.STEP_EXECUTORS, "view.materialize", _slow_step)
    monkeypatch.setattr(progress_service, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    result = _RUNNER.invoke(
        app,
        [
            "recipe",
            "run",
            "slow_recipe",
            "--workspace",
            workspace_dir.as_posix(),
            "--progress",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stdout

    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    event_types = [line["event_type"] for line in lines]
    assert "run_started" in event_types
    assert "step_started" in event_types
    assert "heartbeat" in event_types
    assert "step_finished" in event_types
    assert "run_succeeded" in event_types
    assert event_types[-1] == "command_result"

    run_id = lines[-1]["result"]["run_id"]
    assert isinstance(run_id, str)
    assert run_id.startswith("recipe__slow_recipe__")
    events_path = workspace_dir / "outputs" / "runs" / run_id / "events.jsonl"
    stored_events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event["event_type"] == "heartbeat" for event in stored_events)


def test_recipe_run_uses_distinct_run_directory_per_invocation(tmp_path: Path, monkeypatch) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_progress_workspace_config(workspace_dir)

    def _fast_step(workspace: str | Path, params: dict[str, object], *, force: bool) -> CommandResult:
        del params, force
        return CommandResult(
            command="view materialize",
            workspace_id="progress_demo",
            status="ok",
            artifact_kind="view",
            artifact_id="z20_60",
            outputs=[(Path(workspace) / "outputs" / "views" / "z20_60").as_posix()],
            inputs={"view": "z20_60"},
        )

    monkeypatch.setitem(recipe_service.STEP_EXECUTORS, "view.materialize", _fast_step)

    first = _RUNNER.invoke(app, ["recipe", "run", "slow_recipe", "--workspace", workspace_dir.as_posix(), "--json"])
    second = _RUNNER.invoke(app, ["recipe", "run", "slow_recipe", "--workspace", workspace_dir.as_posix(), "--json"])

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout

    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    first_run_id = first_payload["run_id"]
    second_run_id = second_payload["run_id"]

    assert first_run_id != second_run_id
    first_run = json.loads((workspace_dir / "outputs" / "runs" / first_run_id / "run.json").read_text(encoding="utf-8"))
    second_run = json.loads(
        (workspace_dir / "outputs" / "runs" / second_run_id / "run.json").read_text(encoding="utf-8")
    )
    assert first_run["started_at"] != second_run["started_at"]
    assert {entry.name for entry in (workspace_dir / "outputs" / "runs").iterdir()} == {first_run_id, second_run_id}


def test_workspace_catalog_prioritizes_missing_over_attention(monkeypatch, tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    fake_context = SimpleNamespace(
        workspace_id="catalog_demo",
        workspace_dir=workspace_dir,
        output_root=outputs_dir,
        config=SimpleNamespace(
            workspace=SimpleNamespace(title="Catalog demo"),
            deliverables={"attention_plot": object(), "missing_plot": object()},
            notebooks={},
            exports={},
        ),
    )

    statuses = {
        "attention_plot": SimpleNamespace(
            model_dump=lambda mode="json": {"deliverable_id": "attention_plot", "status": "attention"},
            docs_refs=[],
        ),
        "missing_plot": SimpleNamespace(
            model_dump=lambda mode="json": {"deliverable_id": "missing_plot", "status": "missing"},
            docs_refs=[],
        ),
    }

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.load_workspace_config",
        lambda workspace: fake_context,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.deliverable_status_from_context",
        lambda workspace, deliverable_id, **kwargs: statuses[deliverable_id],
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.write_plot_index",
        lambda context, **kwargs: {"plots": []},
    )

    payload = workspace_catalog(workspace_dir)

    assert payload["state"] == "missing"
    stored = json.loads((outputs_dir / "catalog.json").read_text(encoding="utf-8"))
    assert stored["state"] == "missing"


def test_workspace_catalog_reuses_one_freshness_cache_for_deliverables_and_plot_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    fake_context = SimpleNamespace(
        workspace_id="catalog_demo",
        workspace_dir=workspace_dir,
        output_root=outputs_dir,
        config=SimpleNamespace(
            workspace=SimpleNamespace(title="Catalog demo"),
            deliverables={"dataset_overview": object()},
            notebooks={},
            exports={},
        ),
    )

    caches_seen: list[object | None] = []

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.load_workspace_config",
        lambda workspace: fake_context,
    )

    def fake_deliverable_status_from_context(context, deliverable_id, *, freshness_cache=None):
        assert context is fake_context
        assert deliverable_id == "dataset_overview"
        caches_seen.append(freshness_cache)
        return SimpleNamespace(
            model_dump=lambda mode="json": {"deliverable_id": deliverable_id, "status": "ok"},
            docs_refs=[],
            status="ok",
        )

    def fake_write_plot_index(context, *, freshness_cache=None):
        assert context is fake_context
        caches_seen.append(freshness_cache)
        return {"plots": []}

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.deliverable_status_from_context",
        fake_deliverable_status_from_context,
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.write_plot_index",
        fake_write_plot_index,
    )

    payload = workspace_catalog(workspace_dir)

    assert payload["state"] == "ok"
    assert len(caches_seen) == 2
    assert caches_seen[0] is not None
    assert caches_seen[0] is caches_seen[1]


def test_workspace_catalog_from_context_avoids_workspace_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    fake_context = SimpleNamespace(
        workspace_id="catalog_demo",
        workspace_dir=workspace_dir,
        output_root=outputs_dir,
        config=SimpleNamespace(
            workspace=SimpleNamespace(title="Catalog demo"),
            deliverables={"dataset_overview": object()},
            notebooks={},
            exports={},
        ),
    )

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.load_workspace_config",
        lambda workspace: (_ for _ in ()).throw(AssertionError("workspace reload not expected")),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.deliverable_status_from_context",
        lambda context, deliverable_id, **kwargs: SimpleNamespace(
            model_dump=lambda mode="json": {"deliverable_id": deliverable_id, "status": "ok"},
            docs_refs=[],
            status="ok",
        ),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.write_plot_index",
        lambda context, **kwargs: {"plots": []},
    )

    payload = workspace_catalog_from_context(fake_context)

    assert payload["workspace_id"] == "catalog_demo"
    stored = json.loads((outputs_dir / "catalog.json").read_text(encoding="utf-8"))
    assert stored["workspace_id"] == "catalog_demo"


def test_workspace_catalog_prunes_retired_managed_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    outputs_dir = workspace_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    for artifact_dir in (
        outputs_dir / "views" / "current_view",
        outputs_dir / "views" / "retired_view",
        outputs_dir / "scalars" / "current_scalar",
        outputs_dir / "scalars" / "retired_scalar",
    ):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")

    fake_context = SimpleNamespace(
        workspace_id="catalog_demo",
        workspace_dir=workspace_dir,
        output_root=outputs_dir,
        config=SimpleNamespace(
            workspace=SimpleNamespace(title="Catalog demo"),
            deliverables={"dataset_overview": object()},
            notebooks={},
            exports={},
            views={"current_view": object()},
            scalars={"current_scalar": object()},
            alignments={},
            plots={},
        ),
    )

    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.deliverable_status_from_context",
        lambda context, deliverable_id, **kwargs: SimpleNamespace(
            model_dump=lambda mode="json": {"deliverable_id": deliverable_id, "status": "ok"},
            docs_refs=[],
            status="ok",
        ),
    )
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.catalog_service.write_plot_index",
        lambda context, **kwargs: {"plots": []},
    )

    payload = workspace_catalog_from_context(fake_context)

    assert payload["pruned_artifacts"] == sorted(
        [
            (outputs_dir / "views" / "retired_view").as_posix(),
        ]
    )
    assert not (outputs_dir / "views" / "retired_view").exists()
    assert (outputs_dir / "scalars" / "retired_scalar").is_dir()
    assert (outputs_dir / "views" / "current_view").is_dir()
    assert (outputs_dir / "scalars" / "current_scalar").is_dir()


def test_write_plot_index_uses_freshness_status_for_catalog_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    semantics_dir = workspace_dir / "plot_semantics"
    semantics_dir.mkdir()
    (semantics_dir / "atlas.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "atlas",
                "question": "Does the plot stay current?",
                "decision_role": "debug",
                "encoding": "Fixture plot.",
                "scope": "Fixture scope.",
                "guardrails": ["Fixture guardrail."],
                "caption": "Fixture caption.",
                "alt_text": "Fixture alt text.",
                "preprocessing_md": "Fixture preprocessing.",
                "math_md": "Fixture math.",
                "rationale_md": "Fixture rationale.",
                "limitations_md": "Fixture limitations.",
                "failure_modes_md": "Fixture failure modes.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "plot_index_demo", "output_root": "./outputs"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg"],
                    "neighbor_backend": "auto",
                },
                "sources": {},
                "metadata": {"include": []},
                "plots": {
                    "atlas": {
                        "kind": "projection_grid",
                        "semantics_ref": "./plot_semantics/atlas.yaml",
                        "projections": ["p1"],
                        "color_column": "family",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plot_dir = workspace_dir / "outputs" / "plots" / "atlas"
    plot_dir.mkdir(parents=True)
    (plot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "plot",
                "artifact_id": "atlas",
                "status": "ok",
                "outputs": [{"path": "plot.svg"}],
                "inputs": [],
                "semantics": {"question": "Fixture question.", "decision_role": "debug"},
            }
        ),
        encoding="utf-8",
    )
    (plot_dir / "plot.svg").write_text("<svg />", encoding="utf-8")

    context = load_workspace_config(workspace_dir)
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.plot_service.evaluate_artifact_freshness",
        lambda *args, **kwargs: {"status": "attention", "reason": "stale plot", "known": True},
    )

    payload = write_plot_index(context)

    assert payload["plots"][0]["status"] == "attention"
    assert payload["plots"][0]["stale"] is True
