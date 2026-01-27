from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs import ParquetSink
from dnadesign.densegen.src.adapters.sources import data_source_factory
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.pipeline import PipelineDeps, _load_failure_counts_from_attempts, run_pipeline


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
    ):
        opt = _DummyOpt()
        sol1 = _DummySol(sequence="AAA", library=library, used_indices=[0])

        def _gen():
            yield sol1

        return OptimizerRun(optimizer=opt, generator=_gen())


def _write_config(path: Path, input_path: Path) -> None:
    cfg = {
        "densegen": {
            "schema_version": "2.6",
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
                "quota": 1,
                "sampling": {
                    "pool_strategy": "full",
                    "library_size": 1,
                    "subsample_over_length_budget_by": 0,
                    "cover_all_regulators": False,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                    "allow_incomplete_coverage": True,
                    "iterative_max_libraries": 1,
                    "iterative_min_new_solutions": 0,
                },
                "plan": [{"name": "default", "quota": 1}],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": False,
                "arrays_generated_before_resample": 1,
                "min_count_per_tf": 0,
                "max_duplicate_solutions": 1,
                "stall_seconds_before_resample": 1,
                "stall_warning_every_seconds": 1,
                "max_resample_attempts": 1,
                "max_total_resamples": 1,
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
    run_pipeline(loaded, deps=deps, resume=True, build_stage_a=False)


def test_load_failure_counts_handles_numpy_arrays(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "status": "failed",
                "reason": "constraint",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "library_tfbs": np.array(["AAA", "CCC"]),
                "library_tfs": np.array(["TF1", "TF2"]),
                "library_site_ids": np.array(["site1", "site2"]),
            }
        ]
    )
    df.to_parquet(tables_root / "attempts.parquet", index=False)
    counts = _load_failure_counts_from_attempts(outputs_root)
    key = ("demo_input", "demo_plan", "TF1", "AAA", "site1")
    assert counts[key]["constraint"] == 1
