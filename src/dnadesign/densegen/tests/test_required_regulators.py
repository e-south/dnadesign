from __future__ import annotations

from pathlib import Path

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
        solver_options,
        fixed_elements,
        strands="double",
        regulator_by_index=None,
        required_regulators=None,
        min_count_by_regulator=None,
        min_required_regulators=None,
    ):
        opt = _DummyOpt()
        sol1 = _DummySol(sequence="AAA", library=library, used_indices=[0])
        sol2 = _DummySol(sequence="CCC", library=library, used_indices=[1])

        def _gen():
            yield sol1
            yield sol2

        return OptimizerRun(optimizer=opt, generator=_gen())


def test_required_regulators_filtering(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")
    cfg = {
        "densegen": {
            "schema_version": "2.4",
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
                "parquet": {"path": str(tmp_path / "out.parquet")},
            },
            "generation": {
                "sequence_length": 3,
                "quota": 1,
                "sampling": {
                    "pool_strategy": "full",
                    "library_size": 2,
                    "subsample_over_length_budget_by": 0,
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                    "allow_incomplete_coverage": False,
                    "iterative_max_libraries": 1,
                    "iterative_min_new_solutions": 0,
                },
                "plan": [
                    {
                        "name": "default",
                        "quota": 1,
                        "required_regulators": ["TF2"],
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate", "options": []},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 10,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "stall_seconds_before_resample": 10,
                "stall_warning_every_seconds": 10,
                "max_resample_attempts": 1,
                "max_total_resamples": 1,
                "max_seconds_per_plan": 0,
                "max_failed_solutions": 0,
                "random_seed": 1,
            },
            "postprocess": {"gap_fill": {"mode": "off", "end": "5prime", "gc_min": 0.4, "gc_max": 0.6}},
            "logging": {"log_dir": str(tmp_path / "logs"), "level": "INFO"},
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
        gap_fill=lambda *args, **kwargs: "",
    )
    summary = run_pipeline(loaded, deps=deps, resume=False)
    assert summary.total_generated == 1
    assert len(sink.records) == 1
    assert sink.records[0].sequence == "CCC"


def test_required_regulators_k_of_n(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")
    cfg = {
        "densegen": {
            "schema_version": "2.4",
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
                "parquet": {"path": str(tmp_path / "out.parquet")},
            },
            "generation": {
                "sequence_length": 3,
                "quota": 1,
                "sampling": {
                    "pool_strategy": "full",
                    "library_size": 2,
                    "subsample_over_length_budget_by": 0,
                    "cover_all_regulators": True,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                    "allow_incomplete_coverage": False,
                    "iterative_max_libraries": 1,
                    "iterative_min_new_solutions": 0,
                },
                "plan": [
                    {
                        "name": "default",
                        "quota": 1,
                        "required_regulators": ["TF1", "TF2"],
                        "min_required_regulators": 1,
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate", "options": []},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 10,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 5,
                "stall_seconds_before_resample": 10,
                "stall_warning_every_seconds": 10,
                "max_resample_attempts": 1,
                "max_total_resamples": 1,
                "max_seconds_per_plan": 0,
                "max_failed_solutions": 0,
                "random_seed": 1,
            },
            "postprocess": {"gap_fill": {"mode": "off", "end": "5prime", "gc_min": 0.4, "gc_max": 0.6}},
            "logging": {"log_dir": str(tmp_path / "logs"), "level": "INFO"},
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
        gap_fill=lambda *args, **kwargs: "",
    )
    summary = run_pipeline(loaded, deps=deps, resume=False)
    assert summary.total_generated == 1
    assert len(sink.records) == 1
    assert sink.records[0].sequence == "AAA"
