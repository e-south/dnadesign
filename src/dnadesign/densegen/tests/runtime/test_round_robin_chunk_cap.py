"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_round_robin_chunk_cap.py

Round-robin runtime guardrail tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


class _CountingSink(_DummySink):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


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
        solver_attempt_timeout_seconds=None,
        solver_threads=None,
        extra_label=None,
    ):
        opt = _DummyOpt()
        seqs = ["AAA", "CCC", "GGG", "TTT", "AAC", "CCA"]

        def _gen():
            for seq in seqs:
                yield _DummySol(sequence=seq, library=library)

        return OptimizerRun(optimizer=opt, generator=_gen())


class _NoAcceptProgressAdapter:
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
        solver_attempt_timeout_seconds=None,
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
            solver_attempt_timeout_seconds,
            solver_threads,
            extra_label,
        )
        opt = _DummyOpt()

        def _gen():
            for i in range(8):
                sol = _DummySol(sequence=f"AAA{i}", library=library)
                sol._indices = [0]
                yield sol

        return OptimizerRun(optimizer=opt, generator=_gen())


class _SingleLateSolutionAdapter:
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
        solver_attempt_timeout_seconds=None,
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
            solver_attempt_timeout_seconds,
            solver_threads,
            extra_label,
        )
        opt = _DummyOpt()

        def _gen():
            yield _DummySol(sequence="AAA", library=library)

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
                "parquet": {"path": "outputs/tables/records.parquet"},
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
                        "sequences": 5,
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
                "max_accepted_per_library": 2,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "no_progress_seconds_before_resample": 10,
                "max_consecutive_no_progress_resamples": 25,
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

    assert produced <= loaded.root.densegen.runtime.max_accepted_per_library


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
                "parquet": {"path": "outputs/tables/records.parquet"},
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
                        "sequences": 1,
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
                "max_accepted_per_library": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "no_progress_seconds_before_resample": 1,
                "max_consecutive_no_progress_resamples": 1,
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
            solver_attempt_timeout_seconds=None,
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
    with pytest.raises(RuntimeError, match="max_consecutive_no_progress_resamples"):
        _process_plan_for_source(
            loaded.root.densegen.inputs[0],
            plan_item,
            plan_context,
            one_subsample_only=True,
            already_generated=0,
            execution_state=execution_state,
        )


def test_stall_detected_when_no_accepted_progress_even_with_solver_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "outputs" / "meta").mkdir(parents=True)
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
                "parquet": {"path": "outputs/tables/records.parquet"},
            },
            "generation": {
                "sequence_length": 8,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                },
                "plan": [
                    {
                        "name": "default",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "max_accepted_per_library": 1,
                "min_count_per_tf": 1,
                "max_duplicate_solutions": 10,
                "no_progress_seconds_before_resample": 1,
                "max_consecutive_no_progress_resamples": 1,
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

    monotonic_time = {"value": 0.0}

    def _fake_monotonic() -> float:
        value = monotonic_time["value"]
        monotonic_time["value"] = value + 0.35
        return value

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.stage_b_runtime_callbacks.time.monotonic",
        _fake_monotonic,
    )

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_NoAcceptProgressAdapter(),
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
    execution_state = PlanExecutionState(inputs_manifest={}, events_path=events_path)

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    with pytest.raises(RuntimeError, match="max_consecutive_no_progress_resamples"):
        _process_plan_for_source(
            loaded.root.densegen.inputs[0],
            plan_item,
            plan_context,
            one_subsample_only=True,
            already_generated=0,
            execution_state=execution_state,
        )

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("event") == "STALL_DETECTED" for event in events)


def test_stall_guard_checks_after_candidate_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "outputs" / "meta").mkdir(parents=True)
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
                "parquet": {"path": "outputs/tables/records.parquet"},
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
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "max_accepted_per_library": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 10,
                "no_progress_seconds_before_resample": 1,
                "max_consecutive_no_progress_resamples": 0,
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

    monotonic_time = {"value": 0.0}

    def _fake_monotonic() -> float:
        value = monotonic_time["value"]
        monotonic_time["value"] = value + 1.2
        return value

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.stage_b_runtime_callbacks.time.monotonic",
        _fake_monotonic,
    )

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_SingleLateSolutionAdapter(),
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
    execution_state = PlanExecutionState(inputs_manifest={}, events_path=events_path)

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    produced, _stats = _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=True,
        already_generated=0,
        execution_state=execution_state,
    )
    assert produced == 1
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert not any(event.get("event") == "STALL_DETECTED" for event in events)


def test_no_solution_attempt_records_solver_diagnostics(tmp_path: Path) -> None:
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
                "parquet": {"path": "outputs/tables/records.parquet"},
            },
            "generation": {
                "sequence_length": 8,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                },
                "plan": [
                    {
                        "name": "default",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "max_accepted_per_library": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "no_progress_seconds_before_resample": 10,
                "max_consecutive_no_progress_resamples": 0,
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

    class _NoSolutionAdapter:
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
            solver_attempt_timeout_seconds=None,
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
                solver_attempt_timeout_seconds,
                solver_threads,
                extra_label,
            )
            opt = _DummyOpt()

            def _gen():
                time.sleep(0.05)
                if False:
                    yield _DummySol(sequence="AAA", library=library)

            return OptimizerRun(optimizer=opt, generator=_gen())

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_NoSolutionAdapter(),
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
    assert produced == 0

    attempts_parts = sorted((run_dir / "outputs" / "tables").glob("attempts_part-*.parquet"))
    assert attempts_parts
    attempts = pd.concat([pd.read_parquet(path) for path in attempts_parts], ignore_index=True)
    assert len(attempts) == 1
    row = attempts.iloc[0]
    assert str(row["status"]) == "failed"
    assert str(row["reason"]) == "no_solution"
    assert str(row["solver_status"]) == "no_solution"
    assert float(row["solver_solve_time_s"]) >= 0.01
    detail = json.loads(str(row["detail_json"]))
    assert detail.get("solver_status") == "no_solution"
    assert float(detail.get("solver_solve_time_s", 0.0)) >= 0.01


def test_one_subsample_flushes_all_sinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
                "parquet": {"path": "outputs/tables/records.parquet"},
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
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": True,
                "max_accepted_per_library": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "no_progress_seconds_before_resample": 10,
                "max_consecutive_no_progress_resamples": 25,
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

    class _SamplerSinglePass:
        def __init__(self, **_kwargs) -> None:
            return None

        def run(
            self,
            *,
            build_next_library,
            run_library,
            on_no_solution,
            on_resample,
            already_generated,
            one_subsample_only,
            initial_consecutive_failures=0,
            initial_no_progress_seconds=0.0,
        ):
            _ = (
                build_next_library,
                run_library,
                on_no_solution,
                on_resample,
                already_generated,
                one_subsample_only,
                initial_consecutive_failures,
                initial_no_progress_seconds,
            )
            return SimpleNamespace(generated=0, consecutive_failures_end=0, no_progress_seconds_end=0.0)

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.stage_b_runtime_runner.StageBSampler",
        _SamplerSinglePass,
    )

    sink_a = _CountingSink()
    sink_b = _CountingSink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink_a, sink_b],
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )
    plan_context = PlanRunContext(
        global_cfg=loaded.root.densegen,
        sinks=[sink_a, sink_b],
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
    _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=True,
        already_generated=0,
        execution_state=execution_state,
    )

    assert sink_a.flush_calls == 1
    assert sink_b.flush_calls == 1


def test_attempts_flushed_when_sampler_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "outputs" / "meta").mkdir(parents=True)
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
                "parquet": {"path": "outputs/tables/records.parquet"},
            },
            "generation": {
                "sequence_length": 8,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                },
                "plan": [
                    {
                        "name": "default",
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "max_accepted_per_library": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "no_progress_seconds_before_resample": 1,
                "max_consecutive_no_progress_resamples": 25,
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

    class _SamplerRaisesAfterNoSolution:
        def __init__(self, **_kwargs) -> None:
            return None

        def run(
            self,
            *,
            build_next_library,
            run_library,
            on_no_solution,
            on_resample,
            already_generated,
            one_subsample_only,
            initial_consecutive_failures=0,
            initial_no_progress_seconds=0.0,
        ):
            _ = (
                run_library,
                on_resample,
                already_generated,
                one_subsample_only,
                initial_consecutive_failures,
                initial_no_progress_seconds,
            )
            library = build_next_library()
            if on_no_solution is not None:
                on_no_solution(library, "no_solution")
            raise RuntimeError("Exceeded max_consecutive_no_progress_resamples=60.")

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.stage_b_runtime_runner.StageBSampler",
        _SamplerRaisesAfterNoSolution,
    )

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
    with pytest.raises(RuntimeError, match="Exceeded max_consecutive_no_progress_resamples=60"):
        _process_plan_for_source(
            loaded.root.densegen.inputs[0],
            plan_item,
            plan_context,
            one_subsample_only=False,
            already_generated=0,
            execution_state=execution_state,
        )

    attempts_parts = sorted((run_dir / "outputs" / "tables").glob("attempts_part-*.parquet"))
    assert attempts_parts
    attempts = pd.concat([pd.read_parquet(path) for path in attempts_parts], ignore_index=True)
    assert len(attempts) == 1
    assert str(attempts.iloc[0]["status"]) == "failed"
    assert str(attempts.iloc[0]["reason"]) == "no_solution"
