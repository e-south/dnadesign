from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa

from dnadesign.latentdna.src.io.parquet_io import write_table
from dnadesign.latentdna.src.scalars.build import _representation_scorecard_table


def test_representation_scorecard_keeps_dual_promoters_positive_and_emits_joint_dual_metrics(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    wildtype_dir = output_root / "scalars" / "wildtype_reference_margins_demo"
    synthetic_dir = output_root / "scalars" / "synthetic_reference_margins_demo"
    wildtype_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)

    rows = [
        {
            "id": "ethanol_only",
            "design_family": "ethanol",
            "wildtype_margin_ethanol_vs_control": 0.9,
            "wildtype_margin_cipro_vs_control": -0.3,
            "synthetic_margin_ethanol_vs_background": 0.8,
            "synthetic_margin_cipro_vs_background": -0.2,
        },
        {
            "id": "cipro_only",
            "design_family": "ciprofloxacin",
            "wildtype_margin_ethanol_vs_control": -0.2,
            "wildtype_margin_cipro_vs_control": 0.85,
            "synthetic_margin_ethanol_vs_background": -0.1,
            "synthetic_margin_cipro_vs_background": 0.75,
        },
        {
            "id": "dual",
            "design_family": "ethanol_ciprofloxacin",
            "wildtype_margin_ethanol_vs_control": 0.88,
            "wildtype_margin_cipro_vs_control": 0.83,
            "synthetic_margin_ethanol_vs_background": 0.78,
            "synthetic_margin_cipro_vs_background": 0.73,
        },
        {
            "id": "background",
            "design_family": "background_only",
            "wildtype_margin_ethanol_vs_control": -0.4,
            "wildtype_margin_cipro_vs_control": -0.35,
            "synthetic_margin_ethanol_vs_background": -0.3,
            "synthetic_margin_cipro_vs_background": -0.25,
        },
    ]

    write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": row["id"],
                    "design_family": row["design_family"],
                    "wildtype_margin_ethanol_vs_control": row["wildtype_margin_ethanol_vs_control"],
                    "wildtype_margin_cipro_vs_control": row["wildtype_margin_cipro_vs_control"],
                }
                for row in rows
            ]
        ),
        wildtype_dir / "table.parquet",
    )
    write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": row["id"],
                    "design_family": row["design_family"],
                    "synthetic_margin_ethanol_vs_background": row["synthetic_margin_ethanol_vs_background"],
                    "synthetic_margin_cipro_vs_background": row["synthetic_margin_cipro_vs_background"],
                }
                for row in rows
            ]
        ),
        synthetic_dir / "table.parquet",
    )

    context = SimpleNamespace(output_root=output_root)
    table, _, _ = _representation_scorecard_table(
        context,
        candidates=[
            {
                "candidate_id": "demo_view",
                "wildtype_source": "wildtype_reference_margins_demo",
                "synthetic_source": "synthetic_reference_margins_demo",
            }
        ],
        label_column="design_family",
        ethanol_values={"ethanol", "ethanol_ciprofloxacin"},
        cipro_values={"ciprofloxacin", "ethanol_ciprofloxacin"},
        dual_values={"ethanol_ciprofloxacin"},
    )

    metrics = {row["metric_id"]: float(row["value"]) for row in table.to_pylist() if row["candidate_id"] == "demo_view"}

    assert metrics["wildtype_margin_ethanol_auroc"] == 1.0
    assert metrics["wildtype_margin_cipro_auroc"] == 1.0
    assert metrics["wildtype_margin_dual_joint_auroc"] == 1.0
    assert metrics["wildtype_margin_dual_joint_auprc"] == 1.0
    assert metrics["synthetic_margin_dual_joint_auroc"] == 1.0
    assert metrics["synthetic_margin_dual_joint_auprc"] == 1.0
