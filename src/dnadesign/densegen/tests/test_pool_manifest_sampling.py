"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pool_manifest_sampling.py

Stage-A sampling metadata is persisted in pool manifests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.densegen.src.adapters.sources.base import BaseDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMSamplingSummary
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact
from dnadesign.densegen.src.core.pipeline import PipelineDeps, default_deps


class _DummySource(BaseDataSource):
    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        df = pd.DataFrame(
            {
                "tf": ["regA", "regA", "regB"],
                "tfbs": ["AAAA", "AAAAT", "CCCC"],
                "source": ["dummy", "dummy", "dummy"],
                "motif_id": ["m1", "m1", "m2"],
                "tfbs_id": ["id1", "id2", "id3"],
            }
        )
        summary = PWMSamplingSummary(
            input_name="demo_pwm",
            regulator="regA",
            backend="fimo",
            generated=10,
            target=10,
            target_sites=2,
            candidates_with_hit=9,
            eligible_total=3,
            eligible=2,
            retained=1,
            retained_len_min=4,
            retained_len_median=4.0,
            retained_len_mean=4.0,
            retained_len_max=4,
            retained_score_min=1.0,
            retained_score_median=1.0,
            retained_score_mean=1.0,
            retained_score_max=1.0,
            eligible_tier_counts=[1, 1, 0, 0],
            retained_tier_counts=[1, 0, 0, 0],
            tier0_score=2.0,
            tier1_score=1.5,
            tier2_score=1.0,
            eligible_score_hist_edges=[0.0, 1.0, 2.0],
            eligible_score_hist_counts=[1, 1],
        )
        return [("regA", "AAAA", "dummy")], df, [summary]


def test_pool_manifest_includes_stage_a_sampling(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.5",
                    "run": {"id": "demo", "root": "."},
                    "inputs": [
                        {
                            "name": "demo_pwm",
                            "type": "pwm_meme",
                            "path": str(tmp_path / "motifs.meme"),
                            "sampling": {
                                "scoring_backend": "fimo",
                                "n_sites": 2,
                                "oversample_factor": 2,
                                "bgfile": "inputs/bg.txt",
                            },
                        }
                    ],
                    "output": {
                        "targets": ["parquet"],
                        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                        "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
                    },
                    "generation": {
                        "sequence_length": 10,
                        "quota": 1,
                        "plan": [{"name": "default", "quota": 1}],
                    },
                    "solver": {"backend": "CBC", "strategy": "iterate"},
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
                        "checkpoint_every": 0,
                        "leaderboard_every": 50,
                    },
                    "logging": {"log_dir": "outputs/logs", "level": "INFO"},
                    "postprocess": {"pad": {"mode": "off"}},
                }
            }
        )
    )
    (tmp_path / "motifs.meme").write_text("MEME version 4\n")

    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    out_dir = tmp_path / "outputs" / "pools"
    outputs_root = tmp_path / "outputs"
    base = default_deps()
    deps = PipelineDeps(
        source_factory=lambda _inp, _cfg_path: _DummySource(),
        sink_factory=base.sink_factory,
        optimizer=base.optimizer,
        pad=base.pad,
    )
    build_pool_artifact(
        cfg=cfg,
        cfg_path=cfg_path,
        deps=deps,
        rng=np.random.default_rng(0),
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    manifest = json.loads((out_dir / "pool_manifest.json").read_text())
    entry = manifest["inputs"][0]
    stage_a_sampling = entry.get("stage_a_sampling")
    assert stage_a_sampling is not None
    assert stage_a_sampling["backend"] == "fimo"
    assert stage_a_sampling["tier_scheme"] == "pct_0.1_1_9"
    assert stage_a_sampling["eligibility_rule"].startswith("best_hit_score")
    assert stage_a_sampling["retention_rule"] == "top_n_sites_by_best_hit_score"
    assert stage_a_sampling["fimo_thresh"] == 1.0
    assert stage_a_sampling["bgfile"] == "inputs/bg.txt"
    assert stage_a_sampling["background_source"] == "bgfile"
    hist = stage_a_sampling["eligible_score_hist"]
    assert hist[0]["regulator"] == "regA"
    assert hist[0]["edges"] == [0.0, 1.0, 2.0]
    assert hist[0]["counts"] == [1, 1]
    assert hist[0]["tier0_score"] == 2.0
    assert hist[0]["tier1_score"] == 1.5
    assert hist[0]["tier2_score"] == 1.0
    assert hist[0]["generated"] == 10
    assert hist[0]["candidates_with_hit"] == 9
    assert hist[0]["eligible"] == 3
    assert hist[0]["unique_eligible"] == 2
    assert hist[0]["retained"] == 1
