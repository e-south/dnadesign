"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/reporting/test_reporting_projection_contract.py

Contracts for projected parquet reads in DenseGen reporting data collection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core import reporting_data as reporting_data_module
from dnadesign.densegen.src.core.reporting_data import collect_report_data
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_collect_report_data_reads_projected_parquet_columns(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    write_minimal_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    tables_root = run_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    (tables_root / "attempts.parquet").write_text("placeholder")
    (tables_root / "solutions.parquet").write_text("placeholder")
    (tables_root / "composition.parquet").write_text("placeholder")

    def _fake_load_records_from_config(*_args, **_kwargs):
        return (
            pd.DataFrame(
                [
                    {
                        "id": "sol-1",
                        "sequence": "ATGCATGCAT",
                        "densegen__plan": "demo_plan",
                        "densegen__input_name": "demo_input",
                        "densegen__sampling_library_index": 1,
                        "densegen__sampling_library_hash": "hash1",
                        "densegen__used_tfbs_detail": [],
                        "densegen__required_regulators": [],
                        "densegen__covers_all_tfs_in_solution": True,
                        "densegen__used_tf_counts": [],
                        "densegen__min_count_by_regulator": [],
                    }
                ]
            ),
            "parquet:outputs/tables/records.parquet",
        )

    requested_columns: dict[str, list[str] | None] = {}

    def _fake_read_parquet(path: Path, *args, **kwargs) -> pd.DataFrame:
        name = Path(path).name
        columns = kwargs.get("columns")
        requested_columns[name] = list(columns) if columns is not None else None
        if name == "attempts.parquet":
            return pd.DataFrame(
                [
                    {
                        "status": "success",
                        "sampling_library_hash": "hash1",
                        "sampling_library_index": 1,
                        "input_name": "demo_input",
                        "plan_name": "demo_plan",
                        "library_tfbs": ["AAAA"],
                        "library_tfs": ["TF_A"],
                        "library_site_ids": ["s1"],
                        "library_sources": ["inputs.csv"],
                    }
                ]
            )
        if name == "solutions.parquet":
            return pd.DataFrame([{"solution_id": "sol-1", "sequence": "ATGCATGCAT"}])
        if name == "composition.parquet":
            return pd.DataFrame([{"input_name": "demo_input", "plan_name": "demo_plan", "tf": "TF_A", "tfbs": "AAAA"}])
        raise AssertionError(f"Unexpected parquet read: {path}")

    monkeypatch.setattr(reporting_data_module, "load_records_from_config", _fake_load_records_from_config)
    monkeypatch.setattr(reporting_data_module.pd, "read_parquet", _fake_read_parquet)
    loaded = load_config(cfg_path)
    bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
    assert bundle.tables
    assert requested_columns.get("attempts.parquet") is not None
    assert requested_columns.get("solutions.parquet") is not None
    assert requested_columns.get("composition.parquet") is not None


def test_read_composition_projection_maps_regulator_sequence_columns(tmp_path: Path) -> None:
    composition_path = tmp_path / "composition.parquet"
    pd.DataFrame(
        [
            {
                "solution_id": "sol-1",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "regulator": "TF_A",
                "sequence": "AAAA",
                "length": 4,
            }
        ]
    ).to_parquet(composition_path, index=False)

    frame = reporting_data_module._read_composition_parquet(
        composition_path,
        columns=["solution_id", "input_name", "plan_name", "tf", "tfbs", "length"],
    )

    assert "tf" in frame.columns
    assert "tfbs" in frame.columns
    assert frame.loc[0, "tf"] == "TF_A"
    assert frame.loc[0, "tfbs"] == "AAAA"
