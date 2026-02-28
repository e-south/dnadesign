"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_source_cache.py

Pipeline source cache behavior tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.densegen.src.adapters.optimizer import OptimizerRun
from dnadesign.densegen.src.adapters.outputs.base import SinkBase
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import PoolData
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
        solver_attempt_timeout_seconds=None,
        solver_threads=None,
        extra_label=None,
    ):
        opt = _DummyOpt()
        seqs = ["AAA", "CCC"]

        def _gen():
            for seq in seqs:
                yield _DummySol(sequence=seq, library=library)

        return OptimizerRun(optimizer=opt, generator=_gen())


class _DummySource:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.calls = 0

    def load_data(self, *, rng, outputs_root, run_id=None):
        self.calls += 1
        entries = list(zip(self.df["tf"].tolist(), self.df["tfbs"].tolist(), self.df["source"].tolist()))
        return entries, self.df.copy(), None


def test_source_cache_reuses_loaded_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outputs" / "parquet").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    sites_path = run_dir / "sites.csv"
    sites_path.write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\nTF3,GGG\nTF4,TTT\n")

    cfg = {
        "densegen": {
            "schema_version": "2.9",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo",
                    "type": "binding_sites",
                    "path": str(sites_path),
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
                        "sequences": 2,
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

    dummy_source = _DummySource(
        pd.DataFrame(
            {
                "tf": ["TF1", "TF2", "TF3", "TF4"],
                "tfbs": ["AAA", "CCC", "GGG", "TTT"],
                "tfbs_core": ["AAA", "CCC", "GGG", "TTT"],
                "source": ["demo_source"] * 4,
                "motif_id": ["motif_1", "motif_2", "motif_3", "motif_4"],
                "tfbs_id": ["tfbs_1", "tfbs_2", "tfbs_3", "tfbs_4"],
            }
        )
    )
    sink = _DummySink()
    deps = PipelineDeps(
        source_factory=lambda _cfg, _path: dummy_source,
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

    plan_item = loaded.root.densegen.generation.resolve_plan()[0]
    source_cache: dict[str, PoolData] = {}
    execution_state = PlanExecutionState(inputs_manifest={}, source_cache=source_cache)

    _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=True,
        already_generated=0,
        execution_state=execution_state,
    )

    _process_plan_for_source(
        loaded.root.densegen.inputs[0],
        plan_item,
        plan_context,
        one_subsample_only=True,
        already_generated=0,
        execution_state=execution_state,
    )

    assert dummy_source.calls == 1
