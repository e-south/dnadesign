"""
Phase 18 workflow tests for the promoter-atlas addendum surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.latentdna.src.cli import app
from dnadesign.latentdna.src.workspaces.paths import builtin_templates_dir

_RUNNER = CliRunner()


def _write_usr_dataset(root: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "records.parquet")


def _write_addendum_workspace_config(
    workspace_dir: Path, usr_root: Path, *, include_study_binding: bool = True
) -> None:
    template_path = builtin_templates_dir() / "landmark_atlas_committee" / "config.yaml"
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["workspace"] = {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs"}
    payload.setdefault("defaults", {})
    payload["defaults"]["plot_formats"] = ["svg", "png"]
    payload["defaults"]["neighbor_backend"] = "exact"
    payload["sources"]["anchor60"]["root"] = usr_root.as_posix()
    payload["sources"]["anchor60"]["dataset"] = "promoter/demo_anchor_set"
    payload["sources"]["ctx1k"]["root"] = usr_root.as_posix()
    payload["sources"]["ctx1k"]["dataset"] = "promoter/demo_context_set"
    payload["landmarks"]["spyp"]["where"]["equals"] = "spyP"
    payload["reference_sets"]["promoter_wt_core"]["ids"][0] = "spyP"
    for view_id in [
        "z7_60",
        "z20_60",
        "z7_1k_anchor",
        "z20_1k_anchor",
        "z7_1k_seq",
        "z20_1k_seq",
        "logits7_60",
        "logits20_60",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
        "logits7_1k_seq",
        "logits20_1k_seq",
    ]:
        payload["views"][view_id]["vector"]["name"] = view_id
    for recipe in payload["recipes"].values():
        for step in recipe.get("steps", []):
            params = step.get("params", {})
            dims = params.get("dims")
            if isinstance(dims, int) and dims > 3:
                params["dims"] = 3
            k = params.get("k")
            if isinstance(k, int) and k > 7:
                params["k"] = 7
    if not include_study_binding:
        payload.pop("study_binding", None)
        for deliverable in payload.get("deliverables", {}).values():
            if isinstance(deliverable, dict):
                deliverable["docs_refs"] = []
    (workspace_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _anchor_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "anchor_01",
            "subject_id": "subject_01",
            "usr_label__primary": "dense_01",
            "densegen__plan": "background_only__sigma70_b",
            "densegen__required_regulators": [],
            "template_id": "tpl_a",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.42,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.37,
            "z7_60": [0.0, 0.1, 0.0],
            "z20_60": [0.0, 0.0, 0.1],
            "logits7_60": [0.05, 0.0, -0.05],
            "logits20_60": [0.1, 0.0, -0.1],
        },
        {
            "id": "anchor_02",
            "subject_id": "subject_02",
            "usr_label__primary": "dense_02",
            "densegen__plan": "ethanol__cpxR__sigma70_c",
            "densegen__required_regulators": ["cpxR"],
            "template_id": "tpl_a",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.31,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.28,
            "z7_60": [0.1, 0.0, 0.0],
            "z20_60": [0.2, 0.0, 0.0],
            "logits7_60": [0.12, 0.05, -0.08],
            "logits20_60": [0.2, 0.1, -0.1],
        },
        {
            "id": "anchor_03",
            "subject_id": "subject_03",
            "usr_label__primary": "dense_03",
            "densegen__plan": "ciprofloxacin__lexA__sigma70_d",
            "densegen__required_regulators": ["lexA"],
            "template_id": "tpl_b",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.18,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.16,
            "z7_60": [2.8, 2.9, 3.0],
            "z20_60": [3.0, 3.0, 3.1],
            "logits7_60": [2.7, 2.9, 3.1],
            "logits20_60": [2.8, 3.0, 3.2],
        },
        {
            "id": "anchor_04",
            "subject_id": "subject_04",
            "usr_label__primary": "dense_04",
            "densegen__plan": "ethanol_ciprofloxacin__baeR_lexA__sigma70_e",
            "densegen__required_regulators": ["baeR", "lexA"],
            "template_id": "tpl_b",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.14,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.12,
            "z7_60": [3.0, 3.0, 2.9],
            "z20_60": [3.2, 3.1, 3.0],
            "logits7_60": [2.9, 3.0, 2.8],
            "logits20_60": [3.0, 3.1, 2.9],
        },
        {
            "id": "anchor_05",
            "subject_id": "subject_05",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.27,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.24,
            "z7_60": [1.4, 1.5, 1.5],
            "z20_60": [1.5, 1.5, 1.5],
            "logits7_60": [1.4, 1.4, 1.5],
            "logits20_60": [1.5, 1.4, 1.6],
        },
        {
            "id": "anchor_06",
            "subject_id": "subject_06",
            "usr_label__primary": "spyP",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.22,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.19,
            "z7_60": [0.8, 0.9, 0.9],
            "z20_60": [0.9, 1.0, 1.0],
            "logits7_60": [0.75, 0.85, 0.9],
            "logits20_60": [0.82, 0.95, 0.98],
        },
        {
            "id": "anchor_07",
            "subject_id": "subject_07",
            "usr_label__primary": "sulAp",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.15,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.13,
            "z7_60": [2.4, 2.6, 2.7],
            "z20_60": [2.6, 2.7, 2.8],
            "logits7_60": [2.3, 2.5, 2.6],
            "logits20_60": [2.45, 2.6, 2.7],
        },
        {
            "id": "anchor_08",
            "subject_id": "subject_08",
            "usr_label__primary": "soxSp",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token": -0.11,
            "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token": -0.09,
            "z7_60": [3.3, 3.1, 3.2],
            "z20_60": [3.5, 3.3, 3.4],
            "logits7_60": [3.1, 3.0, 3.1],
            "logits20_60": [3.3, 3.15, 3.2],
        },
    ]


def _context_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "ctx_01",
            "subject_id": "subject_01",
            "construct__anchor_id": "anchor_01",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "dense_01",
            "densegen__plan": "background_only__sigma70_b",
            "densegen__required_regulators": [],
            "template_id": "tpl_a",
            "construct__template_id": "tpl_a",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.39,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.35,
            "z7_1k_anchor": [0.0, 0.1, 0.0001],
            "z7_1k_seq": [0.6, 0.1, 0.0],
            "z20_1k_anchor": [0.0, 0.0, 0.1001],
            "z20_1k_seq": [0.9, 0.0, 0.1],
            "logits7_1k_anchor": [0.05, 0.0, -0.03],
            "logits20_1k_anchor": [0.1, 0.0, -0.05],
            "logits7_1k_seq": [0.2, 0.02, -0.01],
            "logits20_1k_seq": [0.3, 0.03, 0.0],
        },
        {
            "id": "ctx_02",
            "subject_id": "subject_02",
            "construct__anchor_id": "anchor_02",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "dense_02",
            "densegen__plan": "ethanol__cpxR__sigma70_c",
            "densegen__required_regulators": ["cpxR"],
            "template_id": "tpl_a",
            "construct__template_id": "tpl_a",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.29,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.25,
            "z7_1k_anchor": [0.1001, 0.0, 0.0],
            "z7_1k_seq": [0.8, 0.1, 0.0],
            "z20_1k_anchor": [0.2001, 0.0, 0.0],
            "z20_1k_seq": [1.0, 0.2, 0.0],
            "logits7_1k_anchor": [0.12, 0.06, -0.06],
            "logits20_1k_anchor": [0.2, 0.1, -0.05],
            "logits7_1k_seq": [0.25, 0.12, -0.02],
            "logits20_1k_seq": [0.34, 0.18, 0.0],
        },
        {
            "id": "ctx_03",
            "subject_id": "subject_03",
            "construct__anchor_id": "anchor_03",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "dense_03",
            "densegen__plan": "ciprofloxacin__lexA__sigma70_d",
            "densegen__required_regulators": ["lexA"],
            "template_id": "tpl_b",
            "construct__template_id": "tpl_b",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.17,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.14,
            "z7_1k_anchor": [2.8001, 2.9, 3.0],
            "z7_1k_seq": [3.7, 2.9, 3.0],
            "z20_1k_anchor": [3.0, 3.0, 3.1001],
            "z20_1k_seq": [4.0, 3.0, 3.1],
            "logits7_1k_anchor": [2.7, 3.0, 3.0],
            "logits20_1k_anchor": [2.8, 3.1, 3.1],
            "logits7_1k_seq": [3.3, 3.0, 3.1],
            "logits20_1k_seq": [3.45, 3.08, 3.15],
        },
        {
            "id": "ctx_04",
            "subject_id": "subject_04",
            "construct__anchor_id": "anchor_04",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "dense_04",
            "densegen__plan": "ethanol_ciprofloxacin__baeR_lexA__sigma70_e",
            "densegen__required_regulators": ["baeR", "lexA"],
            "template_id": "tpl_b",
            "construct__template_id": "tpl_b",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.13,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.10,
            "z7_1k_anchor": [3.0001, 3.0, 2.9],
            "z7_1k_seq": [3.8, 2.9, 3.0],
            "z20_1k_anchor": [3.2001, 3.1, 3.0],
            "z20_1k_seq": [4.2, 3.0, 3.2],
            "logits7_1k_anchor": [2.9, 3.1, 2.9],
            "logits20_1k_anchor": [3.0, 3.2, 3.0],
            "logits7_1k_seq": [3.5, 3.1, 3.0],
            "logits20_1k_seq": [3.7, 3.18, 3.12],
        },
        {
            "id": "ctx_05",
            "subject_id": "subject_05",
            "construct__anchor_id": "anchor_05",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "construct__template_id": "wt",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.23,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.20,
            "z7_1k_anchor": [1.4001, 1.5, 1.5],
            "z7_1k_seq": [1.8, 1.4, 1.6],
            "z20_1k_anchor": [1.5, 1.5, 1.5001],
            "z20_1k_seq": [2.0, 1.4, 1.6],
            "logits7_1k_anchor": [1.4, 1.5, 1.5],
            "logits20_1k_anchor": [1.5, 1.5, 1.55],
            "logits7_1k_seq": [1.6, 1.45, 1.52],
            "logits20_1k_seq": [1.72, 1.5, 1.58],
        },
        {
            "id": "ctx_06",
            "subject_id": "subject_06",
            "construct__anchor_id": "anchor_06",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "spyP",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "construct__template_id": "wt",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.2,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.17,
            "z7_1k_anchor": [0.8001, 0.9, 0.9],
            "z7_1k_seq": [1.3, 0.95, 0.95],
            "z20_1k_anchor": [0.9, 1.0, 1.0001],
            "z20_1k_seq": [1.45, 1.05, 1.05],
            "logits7_1k_anchor": [0.75, 0.85, 0.9],
            "logits20_1k_anchor": [0.82, 0.95, 0.98],
            "logits7_1k_seq": [0.95, 0.9, 0.94],
            "logits20_1k_seq": [1.02, 1.01, 1.0],
        },
        {
            "id": "ctx_07",
            "subject_id": "subject_07",
            "construct__anchor_id": "anchor_07",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "sulAp",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "construct__template_id": "wt",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.14,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.12,
            "z7_1k_anchor": [2.4001, 2.6, 2.7],
            "z7_1k_seq": [3.0, 2.75, 2.8],
            "z20_1k_anchor": [2.6, 2.7, 2.8001],
            "z20_1k_seq": [3.2, 2.85, 2.95],
            "logits7_1k_anchor": [2.3, 2.5, 2.6],
            "logits20_1k_anchor": [2.45, 2.6, 2.7],
            "logits7_1k_seq": [2.65, 2.62, 2.7],
            "logits20_1k_seq": [2.82, 2.75, 2.82],
        },
        {
            "id": "ctx_08",
            "subject_id": "subject_08",
            "construct__anchor_id": "anchor_08",
            "context_id": "ctx_a",
            "construct__context_id": "ctx_a",
            "usr_label__primary": "soxSp",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "construct__template_id": "wt",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.1,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.08,
            "z7_1k_anchor": [3.3001, 3.1, 3.2],
            "z7_1k_seq": [4.0, 3.2, 3.3],
            "z20_1k_anchor": [3.5, 3.3, 3.4001],
            "z20_1k_seq": [4.25, 3.45, 3.55],
            "logits7_1k_anchor": [3.1, 3.0, 3.1],
            "logits20_1k_anchor": [3.3, 3.15, 3.2],
            "logits7_1k_seq": [3.6, 3.25, 3.25],
            "logits20_1k_seq": [3.82, 3.38, 3.32],
        },
    ]


def _context_rows_no_signal() -> list[dict[str, object]]:
    rows = _context_rows()
    for row, anchor_row in zip(rows, _anchor_rows(), strict=True):
        row["z7_1k_anchor"] = [float(value) + 1e-11 for value in anchor_row["z7_60"]]
        row["z7_1k_seq"] = [float(value) + 2e-11 for value in anchor_row["z7_60"]]
        row["z20_1k_anchor"] = [float(value) + 1e-11 for value in anchor_row["z20_60"]]
        row["z20_1k_seq"] = [float(value) + 2e-11 for value in anchor_row["z20_60"]]
        row["logits7_1k_anchor"] = [float(value) + 1e-11 for value in anchor_row["logits7_60"]]
        row["logits7_1k_seq"] = [float(value) + 2e-11 for value in anchor_row["logits7_60"]]
        row["logits20_1k_anchor"] = [float(value) + 1e-11 for value in anchor_row["logits20_60"]]
        row["logits20_1k_seq"] = [float(value) + 2e-11 for value in anchor_row["logits20_60"]]
    return rows


def test_phase18_promoter_addendum_derives_cohorts_and_builds_browser_health(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows())
    _write_addendum_workspace_config(workspace_dir, usr_root)

    materialize_result = _RUNNER.invoke(
        app,
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert materialize_result.exit_code == 0, materialize_result.stdout

    rows = pq.read_table(workspace_dir / "outputs" / "views" / "z20_60" / "rows.parquet").to_pylist()
    by_id = {row["id"]: row for row in rows}
    assert by_id["anchor_01"]["design_family"] == "background_only"
    assert by_id["anchor_01"]["design_regulator_composition"] == "background"
    assert by_id["anchor_01"]["sigma70_variant"] == "b"
    assert by_id["anchor_01"]["campaign_prior"] == "background"
    assert by_id["anchor_01"]["is_control"] is False
    assert by_id["anchor_01"]["source_class"] == "densegen"
    assert by_id["anchor_04"]["design_regulator_composition"] == "baeR+lexA"
    assert by_id["anchor_05"]["design_family"] == "control"
    assert by_id["anchor_05"]["sigma70_variant"] == "control"
    assert by_id["anchor_05"]["campaign_prior"] == "control"
    assert by_id["anchor_05"]["is_control"] is True
    assert by_id["anchor_05"]["source_class"] == "manual_or_wildtype"

    for argv in [
        [
            "deliverable",
            "run",
            "atlas_2x2_intermediate_main",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
        [
            "notebook",
            "smoke",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
    ]:
        result = _RUNNER.invoke(app, argv)
        assert result.exit_code == 0, result.stdout

    browser_path = workspace_dir / "outputs" / "notebooks" / "browser" / "notebook.py"
    controls_path = workspace_dir / "outputs" / "notebooks" / "browser" / "controls.json"
    assert browser_path.is_file()
    assert controls_path.is_file()
    browser_text = browser_path.read_text(encoding="utf-8")
    assert "## Navigation" in browser_text
    assert "## Atlas Viewer" in browser_text
    assert "## Compare Views" in browser_text
    assert "What this shows" in browser_text
    assert "Manifest and QA Details" in browser_text
    assert 'label="Deliverable"' in browser_text
    assert 'label="Model"' in browser_text
    assert 'label="Layout"' in browser_text
    assert 'label="Hue"' in browser_text
    assert 'label="Left geometry"' in browser_text
    assert 'label="Right geometry"' in browser_text

    controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
    assert controls_payload["schema_version"] == "latentdna.workspace_notebook_controls.v2"
    assert controls_payload["runtime_paths"]["workspace_relative_path"] == "../../.."
    assert controls_payload["geometry_switchboard"]["default_model"] == "20b"
    assert controls_payload["context_audit"]["status"] == "missing"
    assert any(
        preset["id"] == "atlas_2x3_model_family"
        for preset in controls_payload["geometry_switchboard"]["layout_presets"]
    )

    plots_index = json.loads((workspace_dir / "outputs" / "plots" / "index.json").read_text(encoding="utf-8"))
    assert plots_index["workspace_id"] == "stress_ethanol_cipro_latent_atlas"
    index_entry = next(item for item in plots_index["plots"] if item["plot_id"] == "atlas_2x2_intermediate_main")
    assert index_entry["deliverable_id"] == "atlas_2x2_intermediate_main"
    assert index_entry["status"] == "ok"
    assert index_entry["rendered_formats"] == ["svg", "png"]
    assert index_entry["stale"] is False

    health_payload = json.loads((workspace_dir / "outputs" / "notebooks" / "health.json").read_text(encoding="utf-8"))
    assert health_payload["status"] == "ok"
    assert health_payload["checks"]["notebook_exists"] is True
    assert health_payload["checks"]["control_plane_loads"] is True
    assert health_payload["checks"]["imports_resolve"] is True
    assert health_payload["checks"]["plot_catalog_loads"] is True
    assert health_payload["checks"]["default_deliverable_ready"] is True
    assert health_payload["checks"]["static_links_resolve"] is True

    inspect_plots = _RUNNER.invoke(
        app,
        ["inspect", "plots", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_plots.exit_code == 0, inspect_plots.stdout
    inspect_plots_payload = json.loads(inspect_plots.stdout)
    plot_ids = {item["plot_id"] for item in inspect_plots_payload["data"]["plots"]}
    assert "atlas_2x2_intermediate_main" in plot_ids

    inspect_health = _RUNNER.invoke(
        app,
        ["inspect", "notebook-health", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_health.exit_code == 0, inspect_health.stdout
    inspect_health_payload = json.loads(inspect_health.stdout)
    assert inspect_health_payload["data"]["health"]["status"] == "ok"


def test_phase18_promoter_addendum_builds_geometry_switchboard_and_context_audit(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows())
    _write_addendum_workspace_config(workspace_dir, usr_root)

    for deliverable_id in ["atlas_2x2_intermediate_main", "geometry_switchboard_20b", "context_audit_primary_20b"]:
        result = _RUNNER.invoke(
            app,
            ["deliverable", "run", deliverable_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    controls_payload = json.loads(
        (workspace_dir / "outputs" / "notebooks" / "browser" / "controls.json").read_text(encoding="utf-8")
    )
    assert controls_payload["geometry_switchboard"]["reference_labels"] == ["spyP", "sulAp", "soxSp", "J23105"]
    geometry_rows = {row["view_id"]: row for row in controls_payload["geometry_switchboard"]["geometries"]}
    assert {
        "z7_60",
        "z20_60",
        "z7_1k_anchor",
        "z20_1k_anchor",
        "z7_1k_seq",
        "z20_1k_seq",
        "logits7_60",
        "logits20_60",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
        "logits7_1k_seq",
        "logits20_1k_seq",
    } <= set(geometry_rows)
    assert geometry_rows["z7_60"]["projection_ids"] == ["umap_z7_60"]
    assert geometry_rows["z20_60"]["projection_ids"] == ["umap_z20_60"]
    assert geometry_rows["z7_1k_seq"]["projection_ids"] == ["umap_z7_1k_seq"]
    assert geometry_rows["z20_1k_seq"]["projection_ids"] == ["umap_z20_1k_seq"]
    assert geometry_rows["logits7_60"]["projection_ids"] == ["umap_logits7_60"]
    assert geometry_rows["logits20_60"]["projection_ids"] == ["umap_logits20_60"]
    assert any(
        preset["id"] == "atlas_2x3_model_family"
        for preset in controls_payload["geometry_switchboard"]["layout_presets"]
    )
    model_pair_preset = next(
        preset for preset in controls_payload["geometry_switchboard"]["layout_presets"] if preset["id"] == "model_pair"
    )
    assert model_pair_preset["view_order"] == [
        "z7_60",
        "z20_60",
        "z7_1k_seq",
        "z20_1k_seq",
        "logits7_60",
        "logits20_60",
        "logits7_1k_seq",
        "logits20_1k_seq",
        "z7_1k_anchor",
        "z20_1k_anchor",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
    ]
    assert any(
        basis["alignment_id"] == "anchor_ctx_seq_20b"
        for basis in controls_payload["geometry_switchboard"]["comparison_bases"]
    )
    assert controls_payload["context_audit"]["status"] == "ok"
    assert controls_payload["context_audit"]["decision"] == "whole_sequence_primary"
    assert controls_payload["context_audit"]["metrics"]["construct_shift20_norm_median"] > 0.0
    assert controls_payload["context_audit"]["metrics"]["construct_self_cosine20_median"] < 1.0
    assert controls_payload["context_audit"]["metrics"]["anchor20_log_likelihood_per_token_median"] < 0.0
    assert controls_payload["context_audit"]["metrics"]["expanded_context20_log_likelihood_per_token_median"] < 0.0
    assert controls_payload["context_audit"]["metrics"]["mean_knn_overlap"] >= 0.0
    assert geometry_rows["z20_1k_seq"]["label"] == "20B intermediate 1 kb expanded context"

    audit_table = pq.read_table(
        workspace_dir / "outputs" / "scalars" / "context_audit_20b" / "table.parquet"
    ).to_pylist()
    assert len(audit_table) == len(_anchor_rows())
    assert {
        "construct_shift20_norm",
        "construct_self_cosine20",
        "construct__anchor_id",
        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token",
        "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token",
    } <= set(audit_table[0])

    anchor_rows = pq.read_table(workspace_dir / "outputs" / "views" / "z20_60" / "rows.parquet").to_pylist()
    seq_rows = pq.read_table(workspace_dir / "outputs" / "views" / "z20_1k_seq" / "rows.parquet").to_pylist()
    assert "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token" in anchor_rows[0]
    assert "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token" in seq_rows[0]

    for projection_id in [
        "umap_z7_60",
        "umap_z20_60",
        "umap_z7_1k_anchor",
        "umap_z20_1k_anchor",
        "umap_z7_1k_seq",
        "umap_z20_1k_seq",
        "umap_logits7_60",
        "umap_logits20_60",
        "umap_logits7_1k_anchor",
        "umap_logits20_1k_anchor",
        "umap_logits7_1k_seq",
        "umap_logits20_1k_seq",
    ]:
        assert (workspace_dir / "outputs" / "projections" / projection_id / "coords.parquet").is_file()

    for plot_id in [
        "atlas_2x3_model_family",
        "context_shift_primary_distribution",
        "context_shift_self_cosine_primary",
        "context_shift_vs_drag_primary",
        "context_geometry_primary_summary",
    ]:
        assert (workspace_dir / "outputs" / "plots" / plot_id / "plot.svg").is_file()

    smoke_result = _RUNNER.invoke(
        app,
        ["notebook", "smoke", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert smoke_result.exit_code == 0, smoke_result.stdout

    health_payload = json.loads((workspace_dir / "outputs" / "notebooks" / "health.json").read_text(encoding="utf-8"))
    assert health_payload["status"] == "ok"
    assert health_payload["checks"]["control_plane_loads"] is True


def test_phase18_promoter_addendum_marks_numerically_null_context_lane(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows_no_signal())
    _write_addendum_workspace_config(workspace_dir, usr_root)

    for deliverable_id in ["atlas_2x2_intermediate_main", "context_audit_primary_20b"]:
        result = _RUNNER.invoke(
            app,
            ["deliverable", "run", deliverable_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    controls_payload = json.loads(
        (workspace_dir / "outputs" / "notebooks" / "browser" / "controls.json").read_text(encoding="utf-8")
    )
    assert controls_payload["context_audit"]["status"] == "ok"
    assert controls_payload["context_audit"]["decision"] == "no_context_signal"
    assert controls_payload["context_audit"]["metrics"]["construct_shift20_norm_median"] < 1e-8
    assert controls_payload["context_audit"]["metrics"]["construct_self_cosine20_median"] > 0.999999


def test_phase18_promoter_addendum_runs_reference_alignment_and_x2_whole_sequence_export(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows())
    _write_addendum_workspace_config(workspace_dir, usr_root)

    for deliverable_id in ["reference_alignment_primary_20b", "x2_primary_20b"]:
        result = _RUNNER.invoke(
            app,
            ["deliverable", "run", deliverable_id, "--workspace", workspace_dir.as_posix(), "--json"],
        )
        assert result.exit_code == 0, result.stdout

    output_root = workspace_dir / "outputs"
    reference_table = pq.read_table(output_root / "distances" / "seq20_reference_distances" / "table.parquet")
    assert {"d_spyp", "d_sulap", "d_soxsp", "d_j23105"} <= set(reference_table.column_names)
    assert (output_root / "plots" / "reference_alignment_seq20b" / "plot.svg").is_file()
    assert (output_root / "plots" / "reference_alignment_anchor20b" / "plot.svg").is_file()

    x2_dir = output_root / "exports" / "x2_primary_20b"
    x2_matrix = pq.read_table(x2_dir / "features.parquet").to_pylist()
    feature_names = [row["feature_name"] for row in x2_matrix]
    assert "z20_60_pc_001" in feature_names
    assert "z20_1k_seq_pc_001" in feature_names
    assert "anchor_ref_d_spyp_centered" in feature_names
    assert "seq_ref_d_spyp_centered" in feature_names
    assert "construct_shift20_norm" in feature_names
    assert "construct_self_cosine20" in feature_names
    assert "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token" in feature_names
    assert "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token" in feature_names
    assert len(feature_names) == len(set(feature_names))


def test_phase18_promoter_addendum_runs_without_study_binding(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows())
    _write_addendum_workspace_config(workspace_dir, usr_root, include_study_binding=False)

    for argv in [
        ["deliverable", "run", "atlas_2x2_intermediate_main", "--workspace", workspace_dir.as_posix(), "--json"],
        ["deliverable", "run", "x2_primary_20b", "--workspace", workspace_dir.as_posix(), "--json"],
        ["notebook", "smoke", "--workspace", workspace_dir.as_posix(), "--json"],
        ["validate", "workspace", "--workspace", workspace_dir.as_posix(), "--deep", "--json"],
    ]:
        result = _RUNNER.invoke(app, argv)
        assert result.exit_code == 0, result.stdout

    controls_payload = json.loads(
        (workspace_dir / "outputs" / "notebooks" / "browser" / "controls.json").read_text(encoding="utf-8")
    )
    assert controls_payload["geometry_switchboard"]["reference_labels"] == ["spyP", "sulAp", "soxSp", "J23105"]


def test_phase18_promoter_addendum_supports_leiden_xy_curve_and_correspondence_plots(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    usr_root = tmp_path / "usr_root"
    _write_usr_dataset(usr_root, "promoter/demo_anchor_set", _anchor_rows())
    _write_usr_dataset(usr_root, "promoter/demo_context_set", _context_rows())
    _write_addendum_workspace_config(workspace_dir, usr_root)

    for argv in [
        ["view", "materialize", "z20_60", "--workspace", workspace_dir.as_posix(), "--json"],
        ["view", "materialize", "z20_1k_anchor", "--workspace", workspace_dir.as_posix(), "--json"],
        ["alignment", "build", "anchor_ctx_20b", "--workspace", workspace_dir.as_posix(), "--json"],
        [
            "view",
            "reduce",
            "z20_60",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z20_60_anchor_ctx_pca",
            "--alignment",
            "anchor_ctx_20b",
            "--dims",
            "2",
            "--reduced-view-id",
            "z20_60_anchor_ctx_pc2",
            "--json",
        ],
        [
            "view",
            "reduce",
            "z20_1k_anchor",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z20_1k_anchor_anchor_ctx_pca",
            "--alignment",
            "anchor_ctx_20b",
            "--dims",
            "2",
            "--reduced-view-id",
            "z20_1k_anchor_anchor_ctx_pc2",
            "--json",
        ],
        [
            "neighbors",
            "fit",
            "z20_60_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--reduced-view",
            "z20_60_anchor_ctx_pc2",
            "--k",
            "2",
            "--backend",
            "exact",
            "--json",
        ],
        [
            "neighbors",
            "fit",
            "z20_1k_anchor_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--reduced-view",
            "z20_1k_anchor_anchor_ctx_pc2",
            "--k",
            "2",
            "--backend",
            "exact",
            "--json",
        ],
        [
            "cluster",
            "fit",
            "leiden_z20_60",
            "--workspace",
            workspace_dir.as_posix(),
            "--reduced-view",
            "z20_60_anchor_ctx_pc2",
            "--method",
            "leiden",
            "--neighbor-set",
            "z20_60_knn",
            "--k",
            "2",
            "--resolution",
            "0.5",
            "--json",
        ],
        [
            "cluster",
            "fit",
            "leiden_z20_1k_anchor",
            "--workspace",
            workspace_dir.as_posix(),
            "--reduced-view",
            "z20_1k_anchor_anchor_ctx_pc2",
            "--method",
            "leiden",
            "--neighbor-set",
            "z20_1k_anchor_knn",
            "--k",
            "2",
            "--resolution",
            "0.5",
            "--json",
        ],
        [
            "distance",
            "score",
            "primary_landmark_distances",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
            "--landmark",
            "spyp",
            "--landmark",
            "sulap",
            "--json",
        ],
        [
            "view",
            "derive",
            "delta20",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
        [
            "scalar",
            "derive",
            "delta20_norm",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
        [
            "view",
            "reduce",
            "z20_60",
            "--workspace",
            workspace_dir.as_posix(),
            "--run-id",
            "z20_60_pca",
            "--dims",
            "2",
            "--json",
        ],
    ]:
        result = _RUNNER.invoke(app, argv)
        assert result.exit_code == 0, result.stdout

    xy_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "distance_margin_hexbin",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "xy_scatter",
            "--distance",
            "primary_landmark_distances",
            "--x-column",
            "d_spyp",
            "--y-column",
            "d_sulap",
            "--render-mode",
            "hexbin",
            "--json",
        ],
    )
    assert xy_result.exit_code == 0, xy_result.stdout

    curve_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "scree_curve",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "curve",
            "--reducer",
            "z20_60_pca",
            "--json",
        ],
    )
    assert curve_result.exit_code == 0, curve_result.stdout

    correspondence_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "cluster_correspondence_manual",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "correspondence_heatmap",
            "--left-cluster",
            "leiden_z20_60",
            "--right-cluster",
            "leiden_z20_1k_anchor",
            "--json",
        ],
    )
    assert correspondence_result.exit_code == 0, correspondence_result.stdout

    distribution_result = _RUNNER.invoke(
        app,
        [
            "plot",
            "render",
            "context_shift_ecdf",
            "--workspace",
            workspace_dir.as_posix(),
            "--kind",
            "distribution",
            "--scalar",
            "delta20_norm",
            "--value-column",
            "delta20_norm",
            "--color-column",
            "design_family",
            "--render-mode",
            "ecdf",
            "--json",
        ],
    )
    assert distribution_result.exit_code == 0, distribution_result.stdout

    cluster_summary = json.loads(
        (workspace_dir / "outputs" / "clusters" / "leiden_z20_60" / "summary.json").read_text(encoding="utf-8")
    )
    assert cluster_summary["method"] == "leiden"
    assert cluster_summary["reduced_view_id"] == "z20_60_anchor_ctx_pc2"
    assert cluster_summary["k"] == 2
    assert cluster_summary["resolution"] == 0.5
    assert (workspace_dir / "outputs" / "clusters" / "leiden_z20_60" / "cluster_sizes.parquet").is_file()
    assert (workspace_dir / "outputs" / "clusters" / "leiden_z20_60" / "medoids.parquet").is_file()
    assert (workspace_dir / "outputs" / "clusters" / "leiden_z20_60" / "nearest_landmarks.parquet").is_file()

    for plot_id in [
        "distance_margin_hexbin",
        "scree_curve",
        "cluster_correspondence_manual",
        "context_shift_ecdf",
    ]:
        plot_dir = workspace_dir / "outputs" / "plots" / plot_id
        assert (plot_dir / "plot.svg").is_file()
        assert (plot_dir / "manifest.json").is_file()
