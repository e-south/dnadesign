"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_solver_strategy_runtime.py

Runtime coverage for non-iterate solver strategy wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs.base import SinkBase
from dnadesign.densegen.src.adapters.sources import data_source_factory
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.pipeline import PipelineDeps, run_pipeline


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
    def __init__(self, sequence: str, library: list[str], used_indices: list[int]) -> None:
        self.sequence = sequence
        self.library = library
        self._indices = used_indices
        self.compression_ratio = 1.0

    def offset_indices_in_order(self):
        return [(0, idx) for idx in self._indices]


class _RecordingAdapter:
    def __init__(self) -> None:
        self.strategies: list[str] = []

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
        solver_attempt_timeout_seconds=None,
        solver_threads=None,
        extra_label=None,
    ):
        _ = (
            sequence_length,
            solver,
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
        self.strategies.append(str(strategy))
        opt = _DummyOpt()
        sol = _DummySol(sequence="AAAAAA", library=library, used_indices=[0])

        def _gen():
            yield sol

        return OptimizerRun(optimizer=opt, generator=_gen())


@pytest.mark.parametrize("solver_strategy", ["diverse", "optimal"])
def test_runtime_uses_non_iterate_solver_strategy(
    tmp_path: Path,
    solver_strategy: str,
) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\n")
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
                "sequence_length": 6,
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
                        "sequences": 1,
                        "sampling": {"include_inputs": ["demo"]},
                        "regulator_constraints": {"groups": []},
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": solver_strategy},
            "runtime": {
                "round_robin": False,
                "max_accepted_per_library": 10,
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
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(cfg_path)

    sink = _DummySink()
    adapter = _RecordingAdapter()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=adapter,
        pad=lambda *args, **kwargs: "",
    )
    summary = run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)

    assert summary.total_generated == 1
    assert adapter.strategies == [solver_strategy]
    assert len(sink.records) == 1

    run_manifest = json.loads((tmp_path / "outputs" / "meta" / "run_manifest.json").read_text())
    assert run_manifest["solver_strategy"] == solver_strategy
