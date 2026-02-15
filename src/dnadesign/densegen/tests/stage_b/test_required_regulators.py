"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/densegen/tests/stage_b/test_required_regulators.py

Regulator group constraint coverage for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
        sol1 = _DummySol(sequence="AAAAAA", library=library, used_indices=[0])
        sol2 = _DummySol(sequence="GGGGGG", library=library, used_indices=[0, 2])

        def _gen():
            yield sol1
            yield sol2

        return OptimizerRun(optimizer=opt, generator=_gen())


def test_regulator_groups_filtering(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\nTF3,GGG\n")
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
                    "library_size": 3,
                    "cover_all_regulators": True,
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
                                    "name": "group_a",
                                    "members": ["TF1"],
                                    "min_required": 1,
                                },
                                {
                                    "name": "group_b",
                                    "members": ["TF3"],
                                    "min_required": 1,
                                },
                            ],
                            "min_count_by_regulator": {"TF3": 1},
                        },
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 10,
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
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(cfg_path)
    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )
    summary = run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)
    assert summary.total_generated == 1
    assert len(sink.records) == 1
    assert sink.records[0].sequence == "GGGGGG"


def test_runtime_min_count_per_tf_rejection_then_acceptance(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")

    class _MinCountAdapter:
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
            opt = _DummyOpt()
            sol_missing_tf2 = _DummySol(sequence="AAAAAA", library=library, used_indices=[0])
            sol_ok = _DummySol(sequence="GGGGGG", library=library, used_indices=[0, 1])

            def _gen():
                yield sol_missing_tf2
                yield sol_ok

            return OptimizerRun(optimizer=opt, generator=_gen())

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
                    "library_size": 2,
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
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
                "arrays_generated_before_resample": 10,
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
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(cfg_path)

    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_MinCountAdapter(),
        pad=lambda *args, **kwargs: "",
    )
    summary = run_pipeline(loaded, deps=deps, resume=False, build_stage_a=True)

    assert summary.total_generated == 1
    assert len(sink.records) == 1
    assert sink.records[0].sequence == "GGGGGG"

    attempts = pd.read_parquet(tmp_path / "outputs" / "tables" / "attempts.parquet")
    rejected = attempts[attempts["status"] == "rejected"]
    assert "min_count_per_tf" in set(rejected["reason"].tolist())
