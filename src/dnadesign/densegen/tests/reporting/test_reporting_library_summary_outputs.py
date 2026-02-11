"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/reporting/test_reporting_library_summary_outputs.py

Report-table coverage for library summary outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.reporting import collect_report_data
from dnadesign.densegen.src.core.reporting_render import _render_report_md
from dnadesign.densegen.tests.meta_fixtures import output_meta

PLAN_POOL_LABEL = "plan_pool__demo_plan"


def _write_config(path: Path) -> None:
    path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_input
              type: binding_sites
              path: inputs.csv
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/dense_arrays.parquet
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                quota: 1
                sampling:
                  include_inputs: [demo_input]
                regulator_constraints:
                  groups: []
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )


def _write_usr_config(path: Path) -> None:
    path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_input
              type: binding_sites
              path: inputs.csv
          output:
            targets: [usr]
            schema:
              bio_type: dna
              alphabet: dna_4
            usr:
              root: outputs/usr_datasets
              dataset: densegen/demo_usr
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                quota: 1
                sampling:
                  include_inputs: [demo_input]
                regulator_constraints:
                  groups: []
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )


def test_library_summary_outputs_filled(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)

    out_file = run_root / "outputs" / "tables" / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    meta = output_meta(library_hash="abc123", library_index=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGCATGCAT",
        meta=meta,
        source="densegen:demo",
        bio_type="dna",
        alphabet="dna_4",
    )
    sink.add(rec)
    sink.finalize()

    attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash1",
                "solution_id": "out1",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["lexA", "cpxR"],
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["lexA", "cpxR"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            },
            {
                "attempt_id": "a2",
                "attempt_index": 2,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:02+00:00",
                "status": "failed",
                "reason": "no_solution",
                "detail_json": "{}",
                "sequence": None,
                "sequence_hash": "",
                "solution_id": "",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["lexA", "cpxR"],
                "sampling_library_index": 2,
                "sampling_library_hash": "def456",
                "solver_status": "infeasible",
                "solver_objective": None,
                "solver_solve_time_s": 0.2,
                "dense_arrays_version": None,
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["lexA", "cpxR"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            },
        ]
    )
    attempts_df.to_parquet(attempts_path, index=False)

    loaded = load_config(cfg_path)
    bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
    library_summary = bundle.tables["library_summary"]

    assert not library_summary.empty
    row = library_summary.iloc[0]
    assert int(row["libraries"]) == 2
    assert int(row["library_size_min"]) == 2
    assert int(row["library_size_max"]) == 2


def test_report_outputs_section_uses_usr_records_path(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_usr_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\nTF_A,AAAA\n")

    records_df = pd.DataFrame(
        [
            {
                "id": "sol-1",
                "sequence": "ATGCATGCAT",
                "densegen__plan": "demo_plan",
                "densegen__input_name": PLAN_POOL_LABEL,
                "densegen__sampling_library_hash": "abc123",
                "densegen__sampling_library_index": 1,
                "densegen__used_tfbs_detail": [{"tf": "lexA", "tfbs": "AAAA"}],
                "densegen__required_regulators": ["lexA"],
                "densegen__covers_all_tfs_in_solution": True,
                "densegen__used_tf_counts": [{"tf": "lexA", "count": 1}],
                "densegen__min_count_by_regulator": [{"tf": "lexA", "min_count": 1}],
            }
        ]
    )

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.reporting_data.load_records_from_config",
        lambda *_args, **_kwargs: (records_df.copy(), "usr:densegen/demo_usr"),
    )

    loaded = load_config(cfg_path)
    bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
    assert bundle.run_report["outputs_path"] == "outputs/usr_datasets/densegen/demo_usr/records.parquet"

    report_md = _render_report_md(bundle)
    assert "- outputs/usr_datasets/densegen/demo_usr/records.parquet" in report_md
    assert "- outputs/tables/dense_arrays.parquet" not in report_md


def test_collect_report_data_maps_used_rows_when_record_library_hash_missing(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")

    attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash1",
                "solution_id": "out1",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["TF1", "TF2"],
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["TF1", "TF2"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            }
        ]
    ).to_parquet(attempts_path, index=False)

    # Dense-array records may omit sampling_library_hash after schema curation;
    # reporting still needs to map usage back to attempts by input/plan/library_index.
    records_df = pd.DataFrame(
        [
            {
                "id": "out1",
                "sequence": "ATGCATGCAT",
                "densegen__plan": "demo_plan",
                "densegen__input_name": PLAN_POOL_LABEL,
                "densegen__sampling_library_index": 1,
                "densegen__used_tfbs_detail": [
                    {"tf": "TF1", "tfbs": "AAA"},
                    {"tf": "TF2", "tfbs": "CCC"},
                ],
                "densegen__required_regulators": [],
                "densegen__covers_all_tfs_in_solution": True,
                "densegen__used_tf_counts": [{"tf": "TF1", "count": 1}, {"tf": "TF2", "count": 1}],
                "densegen__min_count_by_regulator": [],
            }
        ]
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.core.reporting_data.load_records_from_config",
        lambda *_args, **_kwargs: (records_df.copy(), "parquet:outputs/tables/dense_arrays.parquet"),
    )

    loaded = load_config(cfg_path)
    bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
    offered_vs_used_tf = bundle.tables["offered_vs_used_tf"]
    assert not offered_vs_used_tf.empty

    tf1 = offered_vs_used_tf[offered_vs_used_tf["tf"] == "TF1"].iloc[0]
    tf2 = offered_vs_used_tf[offered_vs_used_tf["tf"] == "TF2"].iloc[0]
    assert int(tf1["used_placements"]) == 1
    assert int(tf2["used_placements"]) == 1
