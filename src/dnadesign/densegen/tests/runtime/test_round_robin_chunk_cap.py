"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_round_robin_chunk_cap.py

Round-robin runtime guardrail tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pytest
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
    def __init__(self, sequence: str, library: list[str]) -> None:
        self.sequence = sequence
        self.library = library
        self._indices = [0]
        self.compression_ratio = 1.0

    def offset_indices_in_order(self):
        return [(0, idx) for idx in self._indices]


class _DummyAdapter:
    def probe_solver(self, backend: str, *, test_length: int = 10) -> None:
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
        opt = _DummyOpt()
        seqs = ["AAA", "CCC", "GGG", "TTT", "AAC", "CCA"]

        def _gen():
            for seq in seqs:
                yield _DummySol(sequence=seq, library=library)

        return OptimizerRun(optimizer=opt, generator=_gen())


def test_round_robin_chunk_cap_subsample(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    csv_path = run_dir / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")

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
                "sequence_length": 3,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": False,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                },
                "plan": [
                    {
                        "name": "default",
                        "quota": 5,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {
                            "groups": [
                                {
                                    "name": "all",
                                    "members": ["TF1", "TF2"],
                                    "min_required": 1,
                                }
                            ]
                        },
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": True,
                "arrays_generated_before_resample": 2,
                "min_count_per_tf": 0,
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

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_DummyAdapter(),
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
    execution_state = PlanExecutionState(inputs_manifest={})

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    produced, _stats = _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=True,
        already_generated=0,
        execution_state=execution_state,
    )

    assert produced <= loaded.root.densegen.runtime.arrays_generated_before_resample


def test_stall_detected_with_no_solutions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    csv_path = run_dir / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")

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
                "sequence_length": 3,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": False,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                },
                "plan": [
                    {
                        "name": "default",
                        "quota": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {
                            "groups": [
                                {
                                    "name": "all",
                                    "members": ["TF1", "TF2"],
                                    "min_required": 1,
                                }
                            ]
                        },
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "stall_seconds_before_resample": 1,
                "stall_warning_every_seconds": 0,
                "max_consecutive_failures": 1,
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

    class _EmptyAdapter:
        def probe_solver(self, backend: str, *, test_length: int = 10) -> None:
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
            opt = _DummyOpt()

            def _gen():
                time.sleep(1.1)
                if False:
                    yield None

            return OptimizerRun(optimizer=opt, generator=_gen())

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_EmptyAdapter(),
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
    execution_state = PlanExecutionState(inputs_manifest={})

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    with pytest.raises(RuntimeError, match="max_consecutive_failures"):
        _process_plan_for_source(
            loaded.root.densegen.inputs[0],
            plan_item,
            plan_context,
            one_subsample_only=True,
            already_generated=0,
            execution_state=execution_state,
        )
