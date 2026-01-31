from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.reporting import collect_report_data
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _write_config(path: Path) -> None:
    path.write_text(
        """
        densegen:
          schema_version: "2.8"
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
            quota: 1
            plan:
              - name: demo_plan
                quota: 1
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
                "input_name": "demo_input",
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
                "input_name": "demo_input",
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
