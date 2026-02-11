"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_sampling_feedback_runtime.py

Runtime coverage tests for Stage-B sampling feedback across resamples.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml

from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs.base import SinkBase
from dnadesign.densegen.src.adapters.sources import data_source_factory
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.pipeline.deps import PipelineDeps
from dnadesign.densegen.src.core.pipeline.orchestrator import _process_plan_for_source
from dnadesign.densegen.src.core.pipeline.plan_context import PlanExecutionState, PlanRunContext


class _DummySink(SinkBase):
    def __init__(self) -> None:
        self.records = []

    def add(self, record):
        self.records.append(record)
        return True

    def flush(self) -> None:
        return None


class _DummyOpt:
    def forbid(self, _sol) -> None:
        return None


class _DummySol:
    def __init__(self, *, sequence: str, library: list[str], indices: list[int]) -> None:
        self.sequence = sequence
        self.library = list(library)
        self._indices = list(indices)
        self.compression_ratio = 1.0

    def offset_indices_in_order(self):
        return [(idx * 3, motif_idx) for idx, motif_idx in enumerate(self._indices)]

    def __str__(self) -> str:
        return "visual"


class _SequencedAdapter:
    def __init__(self) -> None:
        self.build_calls = 0

    def probe_solver(self, backend: str, *, test_length: int = 10) -> None:
        _ = (backend, test_length)
        return None

    def build(
        self,
        *,
        library,
        sequence_length,
        solver,
        strategy,
        fixed_elements,
        strands="double",
        regulator_by_index=None,
        required_regulators=None,
        min_count_by_regulator=None,
        min_required_regulators=None,
        solver_time_limit_seconds=None,
        solver_threads=None,
        extra_label=None,
    ):
        _ = (
            sequence_length,
            solver,
            strategy,
            fixed_elements,
            strands,
            regulator_by_index,
            required_regulators,
            min_count_by_regulator,
            min_required_regulators,
            solver_time_limit_seconds,
            solver_threads,
            extra_label,
        )
        self.build_calls += 1
        opt = _DummyOpt()
        if self.build_calls == 1:
            seqs = [_DummySol(sequence="AAA", library=library, indices=[0])]
        else:
            seqs = [_DummySol(sequence="AAACCC", library=library, indices=[0, 1])]

        def _gen():
            for seq in seqs:
                yield seq

        return OptimizerRun(optimizer=opt, generator=_gen())


def test_runtime_iterative_sampling_uses_failure_feedback_across_libraries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    csv_path = run_dir / "sites.csv"
    csv_path.write_text("tf,tfbs,tfbs_core\nTF1,AAA,AAA\nTF2,CCC,CCC\n")

    cfg = {
        "densegen": {
            "schema_version": "2.9",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo",
                    "type": "binding_sites",
                    "path": str(csv_path),
                    "format": "csv",
                }
            ],
            "output": {
                "targets": ["parquet"],
                "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
            },
            "generation": {
                "sequence_length": 6,
                "sampling": {
                    "pool_strategy": "iterative_subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "coverage_weighted",
                    "avoid_failed_motifs": True,
                    "cover_all_regulators": False,
                    "unique_binding_sites": True,
                    "unique_binding_cores": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                    "iterative_max_libraries": 3,
                    "iterative_min_new_solutions": 0,
                },
                "plan": [
                    {
                        "name": "default",
                        "quota": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 1,
                "min_count_per_tf": 1,
                "max_duplicate_solutions": 5,
                "stall_seconds_before_resample": 10,
                "stall_warning_every_seconds": 10,
                "max_consecutive_failures": 25,
                "max_seconds_per_plan": 0,
                "max_failed_solutions": 0,
                "random_seed": 1,
            },
            "postprocess": {"pad": {"mode": "off"}},
            "logging": {"log_dir": "outputs/logs", "level": "INFO"},
        }
    }

    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(cfg_path)
    adapter = _SequencedAdapter()

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=adapter,
        pad=lambda *args, **kwargs: "",
    )
    plan_context = PlanRunContext(
        global_cfg=loaded.root.densegen,
        sinks=[sink],
        chosen_solver="CBC",
        deps=deps,
        rng=random.Random(1),
        np_rng=np.random.default_rng(1),
        cfg_path=loaded.path,
        run_id=str(loaded.root.densegen.run.id),
        run_root=str(run_dir),
        run_config_path="config.yaml",
        run_config_sha256="sha",
        random_seed=1,
        dense_arrays_version=None,
        dense_arrays_version_source="test",
        show_tfbs=False,
        show_solutions=False,
        output_bio_type="dna",
        output_alphabet="dna_4",
    )
    events_path = run_dir / "outputs" / "meta" / "events.jsonl"
    execution_state = PlanExecutionState(
        inputs_manifest={},
        site_failure_counts={},
        events_path=events_path,
    )

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    produced, _stats = _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=False,
        already_generated=0,
        execution_state=execution_state,
    )

    assert produced == 1
    assert adapter.build_calls == 2
    assert execution_state.site_failure_counts
    min_count_failures = sum(
        int(reasons.get("min_count_per_tf", 0)) for reasons in execution_state.site_failure_counts.values()
    )
    assert min_count_failures >= 2

    assert events_path.exists()
    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert rows
    pressure_rows = [row for row in rows if row.get("event") == "LIBRARY_SAMPLING_PRESSURE"]
    if pressure_rows:
        by_index = {int(row.get("library_index", -1)): row for row in pressure_rows}
        assert 1 in by_index
        assert 2 in by_index
        first_failure_counts = by_index[1].get("failure_count_by_tf") or {}
        second_failure_counts = by_index[2].get("failure_count_by_tf") or {}
        first_total = sum(int(v) for v in first_failure_counts.values())
        second_total = sum(int(v) for v in second_failure_counts.values())
        assert first_total == 0
        assert second_total > 0
    else:
        built_rows = [row for row in rows if row.get("event") == "LIBRARY_BUILT"]
        assert len(built_rows) >= 2
