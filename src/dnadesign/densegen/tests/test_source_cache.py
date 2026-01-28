"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_source_cache.py

Pipeline source cache behavior tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs.base import SinkBase
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import PoolData
from dnadesign.densegen.src.core.pipeline import PipelineDeps, _process_plan_for_source


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
    ):
        opt = _DummyOpt()
        seqs = ["AAA", "CCC"]

        def _gen():
            for seq in seqs:
                yield _DummySol(sequence=seq, library=library)

        return OptimizerRun(optimizer=opt, generator=_gen())


class _DummySource:
    def __init__(self, entries: list[str]) -> None:
        self.entries = entries
        self.calls = 0

    def load_data(self, *, rng, outputs_root, run_id=None):
        self.calls += 1
        return self.entries, None, None


def test_source_cache_reuses_loaded_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    seq_path = run_dir / "seqs.csv"
    seq_path.write_text("sequence\nAAA\nCCC\nGGG\nTTT\n")

    cfg = {
        "densegen": {
            "schema_version": "2.5",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo",
                    "type": "sequence_library",
                    "path": str(seq_path),
                    "format": "csv",
                    "sequence_column": "sequence",
                }
            ],
            "output": {
                "targets": ["parquet"],
                "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
            },
            "generation": {
                "sequence_length": 3,
                "quota": 2,
                "sampling": {
                    "pool_strategy": "subsample",
                    "library_size": 2,
                    "subsample_over_length_budget_by": 0,
                    "library_sampling_strategy": "tf_balanced",
                    "cover_all_regulators": False,
                    "unique_binding_sites": True,
                    "max_sites_per_regulator": None,
                    "relax_on_exhaustion": False,
                    "allow_incomplete_coverage": False,
                    "iterative_max_libraries": 2,
                    "iterative_min_new_solutions": 0,
                },
                "plan": [{"name": "default", "quota": 2}],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {
                "round_robin": True,
                "arrays_generated_before_resample": 1,
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
            "postprocess": {"pad": {"mode": "off"}},
            "logging": {"log_dir": "outputs/logs", "level": "INFO"},
        }
    }

    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(cfg_path)

    dummy_source = _DummySource(entries=["AAA", "CCC", "GGG", "TTT"])
    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=lambda _cfg, _path: dummy_source,
        sink_factory=lambda _cfg, _path: [sink],
        optimizer=_DummyAdapter(),
        pad=lambda *args, **kwargs: "",
    )

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    source_cache: dict[str, PoolData] = {}

    _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        loaded.root.densegen,
        [sink],
        chosen_solver="CBC",
        deps=deps,
        rng=random.Random(1),
        np_rng=np.random.default_rng(1),
        cfg_path=loaded.path,
        run_id=loaded.root.densegen.run.id,
        run_root=str(run_dir),
        run_config_path="config.yaml",
        run_config_sha256="sha",
        random_seed=1,
        dense_arrays_version=None,
        dense_arrays_version_source="test",
        output_bio_type="dna",
        output_alphabet="dna_4",
        one_subsample_only=True,
        already_generated=0,
        inputs_manifest={},
        source_cache=source_cache,
    )

    _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        loaded.root.densegen,
        [sink],
        chosen_solver="CBC",
        deps=deps,
        rng=random.Random(1),
        np_rng=np.random.default_rng(1),
        cfg_path=loaded.path,
        run_id=loaded.root.densegen.run.id,
        run_root=str(run_dir),
        run_config_path="config.yaml",
        run_config_sha256="sha",
        random_seed=1,
        dense_arrays_version=None,
        dense_arrays_version_source="test",
        output_bio_type="dna",
        output_alphabet="dna_4",
        one_subsample_only=True,
        already_generated=0,
        inputs_manifest={},
        source_cache=source_cache,
    )

    assert dummy_source.calls == 1
