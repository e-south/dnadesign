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

from dnadesign.latentdna.cli import app

_RUNNER = CliRunner()


def _write_usr_dataset(root: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "records.parquet")


def _write_addendum_workspace_config(workspace_dir: Path, usr_root: Path) -> None:
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs/latentdna"},
                "defaults": {
                    "analysis_dtype": "float32",
                    "metric": "cosine",
                    "random_seed": 17,
                    "plot_formats": ["svg", "png"],
                    "neighbor_backend": "exact",
                },
                "sources": {
                    "anchor60": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_anchor_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                    },
                    "ctx1k": {
                        "kind": "usr",
                        "root": usr_root.as_posix(),
                        "dataset": "promoter/demo_context_set",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": "context_id",
                    },
                },
                "metadata": {
                    "include": [
                        "usr_label__primary",
                        "densegen__plan",
                        "densegen__required_regulators",
                        "template_id",
                    ]
                },
                "alignments": {
                    "anchor_ctx_20b": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    }
                },
                "views": {
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "z20_60"},
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"model": "20B", "family": "intermediate", "context": "60bp"},
                        "role": "primary",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "z20_1k_anchor"},
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"model": "20B", "family": "intermediate", "context": "1kb_anchor"},
                        "role": "primary",
                    },
                    "delta20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_anchor",
                            "right": "z20_60",
                            "alignment": "anchor_ctx_20b",
                        },
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"family": "intermediate", "context": "delta"},
                        "role": "primary",
                    },
                },
                "scalars": {"delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}}},
                "landmarks": {
                    "spyp": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "spyP"},
                        "representation": {"mode": "centroid"},
                    },
                    "sulap": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "sulAp"},
                        "representation": {"mode": "centroid"},
                    },
                    "soxsp": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "soxSp"},
                        "representation": {"mode": "centroid"},
                    },
                    "j23105": {
                        "source": "anchor60",
                        "where": {"column": "usr_label__primary", "equals": "J23105"},
                        "representation": {"mode": "centroid"},
                    },
                },
                "cohorts": {
                    "design_family": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "design_family",
                    },
                    "design_regulator_composition": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "design_regulator_composition",
                    },
                    "sigma70_variant": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "sigma70_variant",
                    },
                    "campaign_prior": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "campaign_prior",
                    },
                    "is_control": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "is_control",
                    },
                    "source_class": {
                        "kind": "promoter_metadata",
                        "source": "anchor60",
                        "derive": "source_class",
                    },
                },
                "plots": {
                    "atlas_primary": {
                        "kind": "projection_scatter",
                        "projection": "umap_z20_60",
                        "color_column": "design_family",
                        "label_column": "usr_label__primary",
                        "label_values": ["spyP", "sulAp", "soxSp", "J23105"],
                    }
                },
                "notebooks": {
                    "browser": {
                        "kind": "workspace_browser",
                        "title": "Promoter atlas browser",
                        "description": "Read-only promoter atlas browser.",
                        "default_deliverable": "atlas_2x2_intermediate_main",
                    }
                },
                "deliverables": {
                    "atlas_2x2_intermediate_main": {
                        "kind": "projection_scatter",
                        "description": "Primary atlas panel.",
                        "question": "Do the design families separate in latent space at all?",
                        "section": "Atlas",
                        "recipe": "atlas_recipe",
                        "outputs": {
                            "samples": ["atlas_sample"],
                            "projections": ["umap_z20_60"],
                            "plots": ["atlas_primary"],
                            "notebooks": ["browser"],
                        },
                    }
                },
                "recipes": {
                    "atlas_recipe": {
                        "steps": [
                            {"id": "materialize_z20_60", "op": "view.materialize", "params": {"view": "z20_60"}},
                            {
                                "id": "sample_all",
                                "op": "sample.build",
                                "depends_on": ["materialize_z20_60"],
                                "params": {"sample": "atlas_sample", "view": "z20_60", "strategy": "all"},
                            },
                            {
                                "id": "fit_projection",
                                "op": "projection.fit",
                                "depends_on": ["sample_all"],
                                "params": {"view": "z20_60", "sample": "atlas_sample", "run_id": "umap_z20_60"},
                            },
                            {
                                "id": "render_atlas",
                                "op": "plot.render",
                                "depends_on": ["fit_projection"],
                                "params": {"plot": "atlas_primary"},
                            },
                            {
                                "id": "generate_browser",
                                "op": "notebook.generate",
                                "depends_on": ["render_atlas"],
                                "params": {"notebook": "browser"},
                            },
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _anchor_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "anchor_01",
            "subject_id": "subject_01",
            "usr_label__primary": "dense_01",
            "densegen__plan": "background_only__sigma70_b",
            "densegen__required_regulators": [],
            "template_id": "tpl_a",
            "z20_60": [0.0, 0.0, 0.1],
        },
        {
            "id": "anchor_02",
            "subject_id": "subject_02",
            "usr_label__primary": "dense_02",
            "densegen__plan": "ethanol__cpxR__sigma70_c",
            "densegen__required_regulators": ["cpxR"],
            "template_id": "tpl_a",
            "z20_60": [0.2, 0.0, 0.0],
        },
        {
            "id": "anchor_03",
            "subject_id": "subject_03",
            "usr_label__primary": "dense_03",
            "densegen__plan": "ciprofloxacin__lexA__sigma70_d",
            "densegen__required_regulators": ["lexA"],
            "template_id": "tpl_b",
            "z20_60": [3.0, 3.0, 3.1],
        },
        {
            "id": "anchor_04",
            "subject_id": "subject_04",
            "usr_label__primary": "dense_04",
            "densegen__plan": "ethanol_ciprofloxacin__baeR_lexA__sigma70_e",
            "densegen__required_regulators": ["baeR", "lexA"],
            "template_id": "tpl_b",
            "z20_60": [3.2, 3.1, 3.0],
        },
        {
            "id": "anchor_05",
            "subject_id": "subject_05",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "z20_60": [1.5, 1.5, 1.5],
        },
    ]


def _context_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "ctx_01",
            "subject_id": "subject_01",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_01",
            "densegen__plan": "background_only__sigma70_b",
            "densegen__required_regulators": [],
            "template_id": "tpl_a",
            "z20_1k_anchor": [0.0, 0.0, 0.2],
        },
        {
            "id": "ctx_02",
            "subject_id": "subject_02",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_02",
            "densegen__plan": "ethanol__cpxR__sigma70_c",
            "densegen__required_regulators": ["cpxR"],
            "template_id": "tpl_a",
            "z20_1k_anchor": [0.3, 0.1, 0.0],
        },
        {
            "id": "ctx_03",
            "subject_id": "subject_03",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_03",
            "densegen__plan": "ciprofloxacin__lexA__sigma70_d",
            "densegen__required_regulators": ["lexA"],
            "template_id": "tpl_b",
            "z20_1k_anchor": [3.0, 3.1, 3.2],
        },
        {
            "id": "ctx_04",
            "subject_id": "subject_04",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_04",
            "densegen__plan": "ethanol_ciprofloxacin__baeR_lexA__sigma70_e",
            "densegen__required_regulators": ["baeR", "lexA"],
            "template_id": "tpl_b",
            "z20_1k_anchor": [3.1, 3.0, 3.2],
        },
        {
            "id": "ctx_05",
            "subject_id": "subject_05",
            "context_id": "ctx_a",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "z20_1k_anchor": [1.6, 1.5, 1.4],
        },
    ]


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

    rows = pq.read_table(workspace_dir / "outputs" / "latentdna" / "views" / "z20_60" / "rows.parquet").to_pylist()
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
            "sample",
            "build",
            "atlas_sample",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
            "--strategy",
            "all",
            "--json",
        ],
        [
            "projection",
            "fit",
            "z20_60",
            "--workspace",
            workspace_dir.as_posix(),
            "--sample",
            "atlas_sample",
            "--run-id",
            "umap_z20_60",
            "--json",
        ],
        [
            "plot",
            "render",
            "atlas_primary",
            "--workspace",
            workspace_dir.as_posix(),
            "--json",
        ],
        [
            "notebook",
            "generate",
            "browser",
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

    browser_path = workspace_dir / "outputs" / "latentdna" / "notebooks" / "browser.py"
    assert browser_path.is_file()
    browser_text = browser_path.read_text(encoding="utf-8")
    assert "Overview" in browser_text
    assert "Atlas" in browser_text
    assert "Landmarks" in browser_text
    assert "Clusters" in browser_text
    assert "deliverable selector" not in browser_text.lower() or "Deliverable" in browser_text

    plots_index = json.loads(
        (workspace_dir / "outputs" / "latentdna" / "plots" / "index.json").read_text(encoding="utf-8")
    )
    assert plots_index["workspace_id"] == "stress_ethanol_cipro_latent_atlas"
    index_entry = next(item for item in plots_index["plots"] if item["plot_id"] == "atlas_primary")
    assert index_entry["deliverable_id"] == "atlas_2x2_intermediate_main"
    assert index_entry["status"] == "ok"
    assert index_entry["rendered_formats"] == ["svg", "png"]
    assert index_entry["stale"] is False

    health_payload = json.loads(
        (workspace_dir / "outputs" / "latentdna" / "notebooks" / "health.json").read_text(encoding="utf-8")
    )
    assert health_payload["status"] == "ok"
    assert health_payload["checks"]["notebook_exists"] is True
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
    assert inspect_plots_payload["data"]["plots"][0]["plot_id"] == "atlas_primary"

    inspect_health = _RUNNER.invoke(
        app,
        ["inspect", "notebook-health", "--workspace", workspace_dir.as_posix(), "--json"],
    )
    assert inspect_health.exit_code == 0, inspect_health.stdout
    inspect_health_payload = json.loads(inspect_health.stdout)
    assert inspect_health_payload["data"]["health"]["status"] == "ok"


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
            "neighbors",
            "fit",
            "z20_60_knn",
            "--workspace",
            workspace_dir.as_posix(),
            "--view",
            "z20_60",
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
            "--view",
            "z20_60",
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
            "--view",
            "z20_1k_anchor",
            "--method",
            "leiden",
            "--alignment",
            "anchor_ctx_20b",
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
            "cluster_correspondence_primary",
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
        (workspace_dir / "outputs" / "latentdna" / "clusters" / "leiden_z20_60" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert cluster_summary["method"] == "leiden"
    assert cluster_summary["k"] == 2
    assert cluster_summary["resolution"] == 0.5
    assert (workspace_dir / "outputs" / "latentdna" / "clusters" / "leiden_z20_60" / "cluster_sizes.parquet").is_file()
    assert (
        workspace_dir
        / "outputs"
        / "latentdna"
        / "clusters"
        / "leiden_z20_60"
        / "cluster_enrichment__design_family.parquet"
    ).is_file()
    assert (workspace_dir / "outputs" / "latentdna" / "clusters" / "leiden_z20_60" / "medoids.parquet").is_file()
    assert (
        workspace_dir / "outputs" / "latentdna" / "clusters" / "leiden_z20_60" / "nearest_landmarks.parquet"
    ).is_file()

    for plot_id in [
        "distance_margin_hexbin",
        "scree_curve",
        "cluster_correspondence_primary",
        "context_shift_ecdf",
    ]:
        plot_dir = workspace_dir / "outputs" / "latentdna" / "plots" / plot_id
        assert (plot_dir / "plot.svg").is_file()
        assert (plot_dir / "manifest.json").is_file()
