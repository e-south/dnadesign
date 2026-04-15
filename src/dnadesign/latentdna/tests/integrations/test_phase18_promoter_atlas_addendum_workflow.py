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
                "workspace": {"id": "stress_ethanol_cipro_latent_atlas", "output_root": "./outputs"},
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
                        "infer__evo2_7b__anchor_only_7b_features__log_likelihood__mean_per_token",
                        "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token",
                        "infer__evo2_20b__anchor_only_20b_features__log_likelihood__mean_per_token",
                        "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token",
                    ]
                },
                "alignments": {
                    "anchor_ctx_20b": {
                        "left": "z20_1k_anchor",
                        "right": "z20_60",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                    "anchor_seq_20b": {
                        "left": "z20_1k_seq",
                        "right": "z20_1k_anchor",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                    "anchor_ctx_7b": {
                        "left": "z7_1k_anchor",
                        "right": "z7_60",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                    "anchor_seq_7b": {
                        "left": "z7_1k_seq",
                        "right": "z7_1k_anchor",
                        "on": "subject_key",
                        "support": "intersection",
                    },
                },
                "views": {
                    "z7_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "z7_60"},
                        "coordinate_space_id": "evo2_7b_intermediate",
                        "tags": {"model": "7B", "family": "intermediate", "context": "60bp"},
                        "role": "committee_member",
                    },
                    "z20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "z20_60"},
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"model": "20B", "family": "intermediate", "context": "60bp"},
                        "role": "primary",
                    },
                    "z7_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "z7_1k_anchor"},
                        "coordinate_space_id": "evo2_7b_intermediate",
                        "tags": {"model": "7B", "family": "intermediate", "context": "1kb_anchor"},
                        "role": "committee_member",
                    },
                    "z20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "z20_1k_anchor"},
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"model": "20B", "family": "intermediate", "context": "1kb_anchor"},
                        "role": "primary",
                    },
                    "z7_1k_seq": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "z7_1k_seq"},
                        "coordinate_space_id": "evo2_7b_intermediate",
                        "tags": {"model": "7B", "family": "intermediate", "context": "1kb_seq"},
                        "role": "challenger",
                    },
                    "z20_1k_seq": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "z20_1k_seq"},
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"model": "20B", "family": "intermediate", "context": "1kb_seq"},
                        "role": "challenger",
                    },
                    "logits7_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "logits7_60"},
                        "coordinate_space_id": "evo2_7b_pooled_logits",
                        "tags": {"model": "7B", "family": "pooled_logits", "context": "60bp"},
                        "role": "challenger",
                    },
                    "logits20_60": {
                        "source": "anchor60",
                        "vector": {"kind": "column", "name": "logits20_60"},
                        "coordinate_space_id": "evo2_20b_pooled_logits",
                        "tags": {"model": "20B", "family": "pooled_logits", "context": "60bp"},
                        "role": "challenger",
                    },
                    "logits7_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "logits7_1k_anchor"},
                        "coordinate_space_id": "evo2_7b_pooled_logits",
                        "tags": {"model": "7B", "family": "pooled_logits", "context": "1kb_anchor"},
                        "role": "challenger",
                    },
                    "logits20_1k_anchor": {
                        "source": "ctx1k",
                        "vector": {"kind": "column", "name": "logits20_1k_anchor"},
                        "coordinate_space_id": "evo2_20b_pooled_logits",
                        "tags": {"model": "20B", "family": "pooled_logits", "context": "1kb_anchor"},
                        "role": "challenger",
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
                    "drag20": {
                        "derive": {
                            "kind": "vector_difference",
                            "left": "z20_1k_seq",
                            "right": "z20_1k_anchor",
                            "alignment": "anchor_seq_20b",
                        },
                        "coordinate_space_id": "evo2_20b_intermediate",
                        "tags": {"family": "intermediate", "context": "drag"},
                        "role": "challenger",
                    },
                },
                "scalars": {
                    "delta20_norm": {"derive": {"kind": "vector_norm", "view": "delta20", "norm": "l2"}},
                    "drag20_norm": {"derive": {"kind": "vector_norm", "view": "drag20", "norm": "l2"}},
                    "drag20_norm_for_context_audit": {
                        "derive": {
                            "kind": "select_columns",
                            "source": "drag20_norm",
                            "columns": ["subject_id", "drag20_norm"],
                        }
                    },
                    "context_audit_20b": {
                        "derive": {
                            "kind": "join_tables",
                            "sources": ["delta20_norm", "drag20_norm_for_context_audit"],
                            "on": ["subject_id"],
                        }
                    },
                },
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
                    },
                    "atlas_2x3_model_family": {
                        "kind": "projection_grid",
                        "projections": [
                            "umap_z7_60",
                            "umap_z20_60",
                            "umap_z7_1k_anchor",
                            "umap_z20_1k_anchor",
                            "umap_logits7_1k_anchor",
                            "umap_logits20_1k_anchor",
                        ],
                        "panel_titles": [
                            "7B anchor-only intermediate",
                            "20B anchor-only intermediate",
                            "7B anchor-aware intermediate",
                            "20B anchor-aware intermediate",
                            "7B anchor-aware pooled logits",
                            "20B anchor-aware pooled logits",
                        ],
                        "color_column": "design_family",
                        "label_column": "usr_label__primary",
                        "label_values": ["spyP", "sulAp", "soxSp", "J23105"],
                    },
                    "drag_qc_distribution": {
                        "kind": "distribution",
                        "scalar": "drag20_norm",
                        "value_column": "drag20_norm",
                        "color_column": "design_family",
                        "render_mode": "ecdf",
                    },
                    "context_shift_vs_drag_primary": {
                        "kind": "xy_scatter",
                        "scalar": "context_audit_20b",
                        "x_column": "delta20_norm",
                        "y_column": "drag20_norm",
                        "color_column": "design_family",
                        "render_mode": "hexbin",
                    },
                },
                "notebooks": {
                    "browser": {
                        "kind": "workspace",
                        "title": "Promoter atlas browser",
                        "description": "Read-only promoter atlas browser.",
                        "default_deliverable": "atlas_2x2_intermediate_main",
                    }
                },
                "deliverables": {
                    "atlas_2x2_intermediate_main": {
                        "recipe": "atlas_recipe",
                        "title": "Atlas 2x2 intermediate main",
                        "section": "atlas",
                        "question": "Do the design families separate in latent space at all?",
                        "summary": "Primary atlas panel for the promoter addendum browser and companion artifacts.",
                        "requires": {
                            "sources": ["anchor60"],
                            "views": ["z20_60"],
                            "recipes": ["atlas_recipe"],
                        },
                        "outputs": {
                            "views": ["z20_60"],
                            "samples": ["atlas_sample"],
                            "projections": ["umap_z20_60"],
                            "plots": ["atlas_primary"],
                            "notebooks": ["browser"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [
                            {"kind": "required_plot_kind", "value": "projection_scatter"},
                        ],
                    },
                    "geometry_switchboard_20b": {
                        "recipe": "geometry_switchboard_recipe",
                        "title": "Geometry switchboard multiview",
                        "section": "atlas",
                        "question": (
                            "Can the browser switch across the core 7B and 20B geometries "
                            "without one static plot per hue?"
                        ),
                        "summary": (
                            "Persist the 7B and 20B intermediate and pooled-logit projections "
                            "needed by the browser atlas viewer."
                        ),
                        "requires": {
                            "sources": ["anchor60", "ctx1k"],
                            "views": [
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
                            ],
                        },
                        "outputs": {
                            "views": [
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
                            ],
                            "samples": ["atlas_sample", "context_sample"],
                            "projections": [
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
                            ],
                            "plots": ["atlas_2x3_model_family"],
                            "notebooks": ["browser"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [],
                    },
                    "context_audit_primary_20b": {
                        "recipe": "context_audit_recipe",
                        "title": "Context audit primary 20B",
                        "section": "context",
                        "question": "Should delta20 remain in x2, or be demoted behind drag-aware QC?",
                        "summary": (
                            "Persist delta20, drag20, and the joined audit table that drives "
                            "the browser decision summary."
                        ),
                        "requires": {
                            "sources": ["anchor60", "ctx1k"],
                            "views": ["z20_60", "z20_1k_anchor", "z20_1k_seq"],
                        },
                        "outputs": {
                            "views": ["delta20", "drag20"],
                            "scalars": ["delta20_norm", "drag20_norm", "context_audit_20b"],
                            "plots": ["drag_qc_distribution", "context_shift_vs_drag_primary"],
                            "notebooks": ["browser"],
                        },
                        "docs_refs": [],
                        "acceptance_checks": [],
                    },
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
                    },
                    "geometry_switchboard_recipe": {
                        "steps": [
                            {
                                "id": "materialize_z7_60_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z7_60"},
                            },
                            {
                                "id": "materialize_z20_60_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z20_60"},
                            },
                            {
                                "id": "materialize_z7_1k_anchor_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z7_1k_anchor"},
                            },
                            {
                                "id": "materialize_z20_1k_anchor_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_anchor"},
                            },
                            {
                                "id": "materialize_z7_1k_seq_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z7_1k_seq"},
                            },
                            {
                                "id": "materialize_z20_1k_seq_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_seq"},
                            },
                            {
                                "id": "materialize_logits7_60_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "logits7_60"},
                            },
                            {
                                "id": "materialize_logits20_60_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "logits20_60"},
                            },
                            {
                                "id": "materialize_logits7_1k_anchor_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "logits7_1k_anchor"},
                            },
                            {
                                "id": "materialize_logits20_1k_anchor_switchboard",
                                "op": "view.materialize",
                                "params": {"view": "logits20_1k_anchor"},
                            },
                            {
                                "id": "sample_anchor_switchboard",
                                "op": "sample.build",
                                "depends_on": ["materialize_z20_60_switchboard", "materialize_logits20_60_switchboard"],
                                "params": {"sample": "atlas_sample", "view": "z20_60", "strategy": "all"},
                            },
                            {
                                "id": "sample_context_switchboard",
                                "op": "sample.build",
                                "depends_on": ["materialize_z20_1k_anchor_switchboard"],
                                "params": {"sample": "context_sample", "view": "z20_1k_anchor", "strategy": "all"},
                            },
                            {
                                "id": "fit_projection_z7_60_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_anchor_switchboard"],
                                "params": {"view": "z7_60", "sample": "atlas_sample", "run_id": "umap_z7_60"},
                            },
                            {
                                "id": "fit_projection_z20_60_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_anchor_switchboard"],
                                "params": {"view": "z20_60", "sample": "atlas_sample", "run_id": "umap_z20_60"},
                            },
                            {
                                "id": "fit_projection_z7_1k_anchor_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {
                                    "view": "z7_1k_anchor",
                                    "sample": "context_sample",
                                    "run_id": "umap_z7_1k_anchor",
                                },
                            },
                            {
                                "id": "fit_projection_z20_1k_anchor_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {
                                    "view": "z20_1k_anchor",
                                    "sample": "context_sample",
                                    "run_id": "umap_z20_1k_anchor",
                                },
                            },
                            {
                                "id": "fit_projection_z7_1k_seq_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {"view": "z7_1k_seq", "sample": "context_sample", "run_id": "umap_z7_1k_seq"},
                            },
                            {
                                "id": "fit_projection_z20_1k_seq_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {
                                    "view": "z20_1k_seq",
                                    "sample": "context_sample",
                                    "run_id": "umap_z20_1k_seq",
                                },
                            },
                            {
                                "id": "fit_projection_logits7_60_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_anchor_switchboard"],
                                "params": {"view": "logits7_60", "sample": "atlas_sample", "run_id": "umap_logits7_60"},
                            },
                            {
                                "id": "fit_projection_logits20_60_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_anchor_switchboard"],
                                "params": {
                                    "view": "logits20_60",
                                    "sample": "atlas_sample",
                                    "run_id": "umap_logits20_60",
                                },
                            },
                            {
                                "id": "fit_projection_logits7_1k_anchor_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {
                                    "view": "logits7_1k_anchor",
                                    "sample": "context_sample",
                                    "run_id": "umap_logits7_1k_anchor",
                                },
                            },
                            {
                                "id": "fit_projection_logits20_1k_anchor_switchboard",
                                "op": "projection.fit",
                                "depends_on": ["sample_context_switchboard"],
                                "params": {
                                    "view": "logits20_1k_anchor",
                                    "sample": "context_sample",
                                    "run_id": "umap_logits20_1k_anchor",
                                },
                            },
                            {
                                "id": "render_atlas_2x3_model_family",
                                "op": "plot.render",
                                "depends_on": [
                                    "fit_projection_z7_60_switchboard",
                                    "fit_projection_z20_60_switchboard",
                                    "fit_projection_z7_1k_anchor_switchboard",
                                    "fit_projection_z20_1k_anchor_switchboard",
                                    "fit_projection_logits7_1k_anchor_switchboard",
                                    "fit_projection_logits20_1k_anchor_switchboard",
                                ],
                                "params": {"plot_id": "atlas_2x3_model_family"},
                            },
                            {
                                "id": "generate_browser_switchboard",
                                "op": "notebook.generate",
                                "depends_on": [
                                    "render_atlas_2x3_model_family",
                                    "fit_projection_z7_60_switchboard",
                                    "fit_projection_z20_60_switchboard",
                                    "fit_projection_z7_1k_anchor_switchboard",
                                    "fit_projection_z20_1k_anchor_switchboard",
                                    "fit_projection_z7_1k_seq_switchboard",
                                    "fit_projection_z20_1k_seq_switchboard",
                                    "fit_projection_logits7_60_switchboard",
                                    "fit_projection_logits20_60_switchboard",
                                    "fit_projection_logits7_1k_anchor_switchboard",
                                    "fit_projection_logits20_1k_anchor_switchboard",
                                ],
                                "params": {"notebook": "browser", "force": True},
                            },
                        ]
                    },
                    "context_audit_recipe": {
                        "steps": [
                            {"id": "materialize_z20_60_audit", "op": "view.materialize", "params": {"view": "z20_60"}},
                            {
                                "id": "materialize_z20_1k_anchor_audit",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_anchor"},
                            },
                            {
                                "id": "materialize_z20_1k_seq_audit",
                                "op": "view.materialize",
                                "params": {"view": "z20_1k_seq"},
                            },
                            {
                                "id": "build_anchor_ctx_audit",
                                "op": "alignment.build",
                                "depends_on": ["materialize_z20_60_audit", "materialize_z20_1k_anchor_audit"],
                                "params": {"alignment": "anchor_ctx_20b"},
                            },
                            {
                                "id": "build_anchor_seq_audit",
                                "op": "alignment.build",
                                "depends_on": ["materialize_z20_1k_anchor_audit", "materialize_z20_1k_seq_audit"],
                                "params": {"alignment": "anchor_seq_20b"},
                            },
                            {
                                "id": "derive_delta20_audit",
                                "op": "view.derive",
                                "depends_on": ["build_anchor_ctx_audit"],
                                "params": {"view": "delta20"},
                            },
                            {
                                "id": "derive_drag20_audit",
                                "op": "view.derive",
                                "depends_on": ["build_anchor_seq_audit"],
                                "params": {"view": "drag20"},
                            },
                            {
                                "id": "derive_delta20_norm_audit",
                                "op": "scalar.derive",
                                "depends_on": ["derive_delta20_audit"],
                                "params": {"scalar": "delta20_norm"},
                            },
                            {
                                "id": "derive_drag20_norm_audit",
                                "op": "scalar.derive",
                                "depends_on": ["derive_drag20_audit"],
                                "params": {"scalar": "drag20_norm"},
                            },
                            {
                                "id": "derive_drag20_norm_selected_audit",
                                "op": "scalar.derive",
                                "depends_on": ["derive_drag20_norm_audit"],
                                "params": {"scalar": "drag20_norm_for_context_audit"},
                            },
                            {
                                "id": "derive_context_audit_join",
                                "op": "scalar.derive",
                                "depends_on": [
                                    "derive_delta20_norm_audit",
                                    "derive_drag20_norm_selected_audit",
                                ],
                                "params": {"scalar": "context_audit_20b"},
                            },
                            {
                                "id": "render_drag_distribution",
                                "op": "plot.render",
                                "depends_on": ["derive_drag20_norm_audit"],
                                "params": {"plot_id": "drag_qc_distribution"},
                            },
                            {
                                "id": "render_context_shift_vs_drag",
                                "op": "plot.render",
                                "depends_on": ["derive_context_audit_join"],
                                "params": {"plot_id": "context_shift_vs_drag_primary"},
                            },
                            {
                                "id": "generate_browser_audit",
                                "op": "notebook.generate",
                                "depends_on": ["render_context_shift_vs_drag"],
                                "params": {"notebook": "browser", "force": True},
                            },
                        ]
                    },
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
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.39,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.35,
            "z7_1k_anchor": [0.0, 0.1, 0.0001],
            "z7_1k_seq": [0.6, 0.1, 0.0],
            "z20_1k_anchor": [0.0, 0.0, 0.1001],
            "z20_1k_seq": [0.9, 0.0, 0.1],
            "logits7_1k_anchor": [0.05, 0.0, -0.03],
            "logits20_1k_anchor": [0.1, 0.0, -0.05],
        },
        {
            "id": "ctx_02",
            "subject_id": "subject_02",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_02",
            "densegen__plan": "ethanol__cpxR__sigma70_c",
            "densegen__required_regulators": ["cpxR"],
            "template_id": "tpl_a",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.29,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.25,
            "z7_1k_anchor": [0.1001, 0.0, 0.0],
            "z7_1k_seq": [0.8, 0.1, 0.0],
            "z20_1k_anchor": [0.2001, 0.0, 0.0],
            "z20_1k_seq": [1.0, 0.2, 0.0],
            "logits7_1k_anchor": [0.12, 0.06, -0.06],
            "logits20_1k_anchor": [0.2, 0.1, -0.05],
        },
        {
            "id": "ctx_03",
            "subject_id": "subject_03",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_03",
            "densegen__plan": "ciprofloxacin__lexA__sigma70_d",
            "densegen__required_regulators": ["lexA"],
            "template_id": "tpl_b",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.17,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.14,
            "z7_1k_anchor": [2.8001, 2.9, 3.0],
            "z7_1k_seq": [3.7, 2.9, 3.0],
            "z20_1k_anchor": [3.0, 3.0, 3.1001],
            "z20_1k_seq": [4.0, 3.0, 3.1],
            "logits7_1k_anchor": [2.7, 3.0, 3.0],
            "logits20_1k_anchor": [2.8, 3.1, 3.1],
        },
        {
            "id": "ctx_04",
            "subject_id": "subject_04",
            "context_id": "ctx_a",
            "usr_label__primary": "dense_04",
            "densegen__plan": "ethanol_ciprofloxacin__baeR_lexA__sigma70_e",
            "densegen__required_regulators": ["baeR", "lexA"],
            "template_id": "tpl_b",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.13,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.10,
            "z7_1k_anchor": [3.0001, 3.0, 2.9],
            "z7_1k_seq": [3.8, 2.9, 3.0],
            "z20_1k_anchor": [3.2001, 3.1, 3.0],
            "z20_1k_seq": [4.2, 3.0, 3.2],
            "logits7_1k_anchor": [2.9, 3.1, 2.9],
            "logits20_1k_anchor": [3.0, 3.2, 3.0],
        },
        {
            "id": "ctx_05",
            "subject_id": "subject_05",
            "context_id": "ctx_a",
            "usr_label__primary": "J23105",
            "densegen__plan": None,
            "densegen__required_regulators": None,
            "template_id": "wt",
            "infer__evo2_7b__template_1kb_7b_features__log_likelihood__mean_per_token": -0.23,
            "infer__evo2_20b__template_1kb_20b_features__log_likelihood__mean_per_token": -0.20,
            "z7_1k_anchor": [1.4001, 1.5, 1.5],
            "z7_1k_seq": [1.8, 1.4, 1.6],
            "z20_1k_anchor": [1.5, 1.5, 1.5001],
            "z20_1k_seq": [2.0, 1.4, 1.6],
            "logits7_1k_anchor": [1.4, 1.5, 1.5],
            "logits20_1k_anchor": [1.5, 1.5, 1.55],
        },
    ]


def _context_rows_no_signal() -> list[dict[str, object]]:
    rows = _context_rows()
    for row, anchor_row in zip(rows, _anchor_rows(), strict=True):
        base = list(anchor_row["z20_60"])
        row["z20_1k_anchor"] = [value + 1e-11 for value in base]
        row["z20_1k_seq"] = [value + 2e-11 for value in base]
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
    assert controls_payload["schema_version"] == "latentdna.workspace_notebook_controls.v1"
    assert controls_payload["geometry_switchboard"]["default_model"] == "20b"
    assert controls_payload["context_audit"]["status"] == "missing"
    assert any(
        preset["id"] == "atlas_2x3_model_family"
        for preset in controls_payload["geometry_switchboard"]["layout_presets"]
    )

    plots_index = json.loads((workspace_dir / "outputs" / "plots" / "index.json").read_text(encoding="utf-8"))
    assert plots_index["workspace_id"] == "stress_ethanol_cipro_latent_atlas"
    index_entry = next(item for item in plots_index["plots"] if item["plot_id"] == "atlas_primary")
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
    assert inspect_plots_payload["data"]["plots"][0]["plot_id"] == "atlas_primary"

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
        "z7_1k_anchor",
        "z20_1k_anchor",
        "z7_1k_seq",
        "z20_1k_seq",
        "logits7_60",
        "logits20_60",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
    ]
    assert any(
        basis["alignment_id"] == "anchor_ctx_20b"
        for basis in controls_payload["geometry_switchboard"]["comparison_bases"]
    )
    assert controls_payload["context_audit"]["status"] == "ok"
    assert controls_payload["context_audit"]["decision"] == "demote_delta_in_x2"
    assert (
        controls_payload["context_audit"]["metrics"]["drag20_median"]
        > controls_payload["context_audit"]["metrics"]["delta20_median"]
    )

    audit_table = pq.read_table(
        workspace_dir / "outputs" / "scalars" / "context_audit_20b" / "table.parquet"
    ).to_pylist()
    assert len(audit_table) == 5
    assert {"delta20_norm", "drag20_norm", "subject_id"} <= set(audit_table[0])

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
    ]:
        assert (workspace_dir / "outputs" / "projections" / projection_id / "coords.parquet").is_file()

    for plot_id in ["atlas_2x3_model_family", "drag_qc_distribution", "context_shift_vs_drag_primary"]:
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
    assert controls_payload["context_audit"]["metrics"]["delta20_median"] < 1e-8
    assert controls_payload["context_audit"]["metrics"]["drag20_median"] < 1e-8


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
        "cluster_correspondence_primary",
        "context_shift_ecdf",
    ]:
        plot_dir = workspace_dir / "outputs" / "plots" / plot_id
        assert (plot_dir / "plot.svg").is_file()
        assert (plot_dir / "manifest.json").is_file()
