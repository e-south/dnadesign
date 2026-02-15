"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_resume_pool_reuse.py

Resume-mode reuse and failure-count parsing tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs import ParquetSink, USRSink
from dnadesign.densegen.src.adapters.outputs.usr_writer import USRWriter
from dnadesign.densegen.src.adapters.sources import data_source_factory
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.pipeline import resume_state as resume_state_module
from dnadesign.densegen.src.core.pipeline.attempts import _load_failure_counts_from_attempts
from dnadesign.densegen.src.core.pipeline.deps import PipelineDeps
from dnadesign.densegen.src.core.pipeline.orchestrator import run_pipeline
from dnadesign.densegen.src.core.pipeline.resume_state import load_resume_state


class _DummyOpt:
    def forbid(self, _sol) -> None:
        return None


class _DummySol:
    def __init__(self, sequence: str, library: list[str], used_indices: list[int]) -> None:
        self.sequence = sequence
        self.library = library
        self._indices = used_indices
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
        sol1 = _DummySol(sequence="AAA", library=library, used_indices=[0])

        def _gen():
            yield sol1

        return OptimizerRun(optimizer=opt, generator=_gen())


def _write_config(path: Path, input_path: Path) -> None:
    cfg = {
        "densegen": {
            "schema_version": "2.9",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo",
                    "type": "binding_sites",
                    "path": str(input_path),
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
                    "pool_strategy": "full",
                    "library_size": 1,
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
                                    "members": ["TF1"],
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
                "max_duplicate_solutions": 1,
                "stall_seconds_before_resample": 1,
                "stall_warning_every_seconds": 1,
                "max_consecutive_failures": 25,
                "max_seconds_per_plan": 0,
                "max_failed_solutions": 0,
                "random_seed": 1,
            },
            "postprocess": {"pad": {"mode": "off"}},
            "logging": {"log_dir": "outputs/logs", "level": "INFO"},
        }
    }
    path.write_text(yaml.safe_dump(cfg))


def test_resume_uses_existing_pool_without_inputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)

    loaded = load_config(cfg_path)

    def _sink_factory(_cfg, _path):
        tables_root = tmp_path / "outputs" / "tables"
        tables_root.mkdir(parents=True, exist_ok=True)
        out_file = tables_root / "dense_arrays.parquet"
        return [ParquetSink(path=str(out_file), chunk_size=1)]

    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=_sink_factory,
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )

    run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)

    csv_path.unlink()
    with pytest.raises(RuntimeError, match="Stage-A pools missing or stale"):
        run_pipeline(loaded, deps=deps, resume=True, build_stage_a=False)


def test_run_pipeline_fails_when_effective_config_write_fails(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)
    loaded = load_config(cfg_path)

    def _sink_factory(_cfg, _path):
        tables_root = tmp_path / "outputs" / "tables"
        tables_root.mkdir(parents=True, exist_ok=True)
        out_file = tables_root / "dense_arrays.parquet"
        return [ParquetSink(path=str(out_file), chunk_size=1)]

    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=_sink_factory,
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )

    def _raise_write_error(**_kwargs) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.orchestrator._write_effective_config",
        _raise_write_error,
    )

    with pytest.raises(RuntimeError, match="effective_config"):
        run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)


def test_run_pipeline_fails_when_stage_b_event_emit_fails(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)
    loaded = load_config(cfg_path)

    def _sink_factory(_cfg, _path):
        tables_root = tmp_path / "outputs" / "tables"
        tables_root.mkdir(parents=True, exist_ok=True)
        out_file = tables_root / "dense_arrays.parquet"
        return [ParquetSink(path=str(out_file), chunk_size=1)]

    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=_sink_factory,
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )

    class _AlwaysResampleSampler:
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
            plan_start,
        ):
            on_resample(
                SimpleNamespace(sampling_library_index=1, sampling_library_hash="forced-resample"),
                "forced-test",
                0,
            )
            return SimpleNamespace(generated=0)

    def _raise_emit_error(*_args, **_kwargs) -> None:
        raise OSError("event stream unavailable")

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.orchestrator.StageBSampler",
        _AlwaysResampleSampler,
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.orchestrator._emit_event",
        _raise_emit_error,
    )

    with pytest.raises(RuntimeError, match="RESAMPLE_TRIGGERED"):
        run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)


def test_run_pipeline_emits_terminal_failure_health_event_for_usr_sink(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)
    loaded = load_config(cfg_path)

    usr_root = tmp_path / "outputs" / "usr_datasets"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = usr_root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")

    def _sink_factory(_cfg, _path):
        writer = USRWriter(
            dataset="demo",
            root=usr_root,
            namespace="densegen",
            chunk_size=1,
            allow_overwrite=False,
        )
        return [USRSink(writer)]

    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=_sink_factory,
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )

    def _raise_run_failure(**_kwargs):
        raise RuntimeError("simulated run failure")

    monkeypatch.setattr(
        "dnadesign.densegen.src.core.pipeline.orchestrator.run_plan_schedule",
        _raise_run_failure,
    )

    with pytest.raises(RuntimeError, match="simulated run failure"):
        run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)

    events_path = usr_root / "demo" / ".events.log"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    terminal_failure = [
        event
        for event in events
        if event.get("action") == "densegen_health" and str((event.get("args") or {}).get("status")) == "failed"
    ]
    assert terminal_failure


def test_load_failure_counts_handles_numpy_arrays(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "status": "failed",
                "reason": "constraint",
                "input_name": "plan_pool__demo_plan",
                "plan_name": "demo_plan",
                "library_tfbs": np.array(["AAA", "CCC"]),
                "library_tfs": np.array(["TF1", "TF2"]),
                "library_site_ids": np.array(["site1", "site2"]),
            }
        ]
    )
    df.to_parquet(tables_root / "attempts.parquet", index=False)
    counts = _load_failure_counts_from_attempts(outputs_root)
    key = ("plan_pool__demo_plan", "demo_plan", "TF1", "AAA", "site1")
    assert counts[key]["constraint"] == 1


def test_load_resume_state_uses_streaming_scan(monkeypatch, tmp_path: Path) -> None:
    loaded = SimpleNamespace(
        root=SimpleNamespace(
            densegen=SimpleNamespace(
                run=SimpleNamespace(id="run-1"),
                output=SimpleNamespace(targets=["parquet"]),
            )
        ),
        path=tmp_path / "config.yaml",
    )

    rows = [
        {
            "densegen__run_id": "run-1",
            "densegen__input_name": "input_a",
            "densegen__plan": "plan_x",
            "densegen__used_tfbs_detail": [{"tf": "TF1", "tfbs": "AAA"}],
        },
        {
            "densegen__run_id": "run-1",
            "densegen__input_name": "input_a",
            "densegen__plan": "plan_x",
            "densegen__used_tfbs_detail": '[{"tf":"TF1","tfbs":"AAA"},{"tf":"TF2","tfbs":"CCC"}]',
        },
        {
            "densegen__run_id": "run-1",
            "densegen__input_name": "input_b",
            "densegen__plan": "plan_y",
            "densegen__used_tfbs_detail": None,
        },
    ]

    def _raise_dataframe_load(*_args, **_kwargs):
        raise AssertionError("resume state should not use dataframe loading")

    def _scan_records_from_config(*_args, **_kwargs):
        return iter(rows), "parquet:/tmp/dense_arrays.parquet"

    monkeypatch.setattr(
        resume_state_module,
        "_load_failure_counts_from_attempts",
        lambda _tables_root: {},
    )
    monkeypatch.setattr(
        resume_state_module,
        "_load_existing_attempt_index_by_plan",
        lambda _tables_root: {},
    )
    monkeypatch.setattr(
        resume_state_module,
        "load_records_from_config",
        _raise_dataframe_load,
        raising=False,
    )
    monkeypatch.setattr(
        resume_state_module,
        "scan_records_from_config",
        _scan_records_from_config,
        raising=False,
    )

    state = load_resume_state(
        resume=True,
        loaded=loaded,
        tables_root=tmp_path / "outputs" / "tables",
        config_sha="abc123",
        allowed_config_sha256=None,
    )

    assert state.existing_counts == {
        ("input_a", "plan_x"): 2,
        ("input_b", "plan_y"): 1,
    }
    assert state.existing_usage_by_plan[("input_a", "plan_x")] == {
        ("TF1", "AAA"): 2,
        ("TF2", "CCC"): 1,
    }
    assert state.existing_usage_by_plan[("input_b", "plan_y")] == {}


def test_load_resume_state_rejects_stream_records_with_mismatched_run_id(monkeypatch, tmp_path: Path) -> None:
    loaded = SimpleNamespace(
        root=SimpleNamespace(
            densegen=SimpleNamespace(
                run=SimpleNamespace(id="run-1"),
                output=SimpleNamespace(targets=["parquet"]),
            )
        ),
        path=tmp_path / "config.yaml",
    )

    def _scan_records_from_config(*_args, **_kwargs):
        rows = [
            {
                "densegen__run_id": "run-2",
                "densegen__input_name": "input_a",
                "densegen__plan": "plan_x",
                "densegen__used_tfbs_detail": [],
            }
        ]
        return iter(rows), "parquet:/tmp/dense_arrays.parquet"

    monkeypatch.setattr(
        resume_state_module,
        "_load_failure_counts_from_attempts",
        lambda _tables_root: {},
    )
    monkeypatch.setattr(
        resume_state_module,
        "_load_existing_attempt_index_by_plan",
        lambda _tables_root: {},
    )
    monkeypatch.setattr(
        resume_state_module,
        "scan_records_from_config",
        _scan_records_from_config,
    )

    with pytest.raises(RuntimeError, match="different run_id"):
        load_resume_state(
            resume=True,
            loaded=loaded,
            tables_root=tmp_path / "outputs" / "tables",
            config_sha="abc123",
            allowed_config_sha256=None,
        )
