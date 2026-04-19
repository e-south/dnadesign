from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.latentdna.src.workspaces.loader import load_workspace_config
from dnadesign.latentdna.src.workspaces.plot_semantics import resolve_plot_semantics


def test_plot_semantics_loads_sidecar_and_returns_required_fields(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "plot_semantics").mkdir()
    (workspace_dir / "plot_semantics" / "dataset_overview.yaml").write_text(
        yaml.safe_dump(
            {
                "plot_id": "dataset_overview",
                "question": "What is in the dataset?",
                "decision_role": "primary",
                "encoding": "Multi-panel bar chart of dataset counts.",
                "scope": "Full population.",
                "guardrails": ["Counts are descriptive, not a model-quality signal."],
                "caption": "Dataset composition across scopes and biological axes.",
                "alt_text": "Multi-panel bar chart summarizing dataset counts.",
                "preprocessing_md": (
                    "Counts use the persisted cohort-inventory table without additional vector preprocessing."
                ),
                "math_md": "Bar heights equal observed row counts or proportions for each cohort partition.",
                "rationale_md": "This plot sets the shared denominator before comparing candidate representations.",
                "limitations_md": "Inventory counts are descriptive and do not measure representation quality.",
                "failure_modes_md": "Small metadata drift or stale cohort labels can misstate the study denominator.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "latentdna.workspace.v1",
                "workspace": {"id": "plot_semantics_demo", "output_root": "./outputs"},
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
                    "dataset_overview": {
                        "kind": "categorical_count",
                        "scalar": "dataset_overview_counts",
                        "category_column": "panel_id",
                        "label_column": "label",
                        "value_column": "row_count",
                        "panel_column": "category",
                        "semantics_ref": "plot_semantics/dataset_overview.yaml",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir)
    semantics = resolve_plot_semantics(context, plot_id="dataset_overview")

    assert semantics.plot_id == "dataset_overview"
    assert semantics.decision_role == "primary"
    assert semantics.scope == "Full population."
    assert semantics.caption == "Dataset composition across scopes and biological axes."
    assert semantics.alt_text == "Multi-panel bar chart summarizing dataset counts."
