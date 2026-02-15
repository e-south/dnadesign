"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_pool_manifest_sampling.py

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
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_metrics import (
    CoreHammingSummary,
    DiversitySummary,
    EntropyBlock,
    EntropySummary,
    KnnBlock,
    KnnSummary,
    PairwiseBlock,
    PairwiseSummary,
    ScoreQuantiles,
    ScoreQuantilesBlock,
    ScoreSummary,
    ScoreSummaryBlock,
)
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_summary import PWMSamplingSummary
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
        top_knn = KnnSummary(
            bins=[0.0, 1.0],
            counts=[0, 1],
            median=1.0,
            p05=1.0,
            p95=1.0,
            frac_le_1=1.0,
            n=2,
            subsampled=False,
            k=1,
        )
        diversified_knn = KnnSummary(
            bins=[0.0, 1.0],
            counts=[0, 1],
            median=1.0,
            p05=1.0,
            p95=1.0,
            frac_le_1=1.0,
            n=2,
            subsampled=False,
            k=1,
        )
        top_pairwise = PairwiseSummary(
            bins=[0.0, 1.0],
            counts=[0, 1],
            median=1.0,
            mean=1.0,
            p10=1.0,
            p90=1.0,
            n_pairs=1,
            total_pairs=1,
            subsampled=False,
        )
        diversified_pairwise = PairwiseSummary(
            bins=[0.0, 1.0],
            counts=[0, 1],
            median=1.0,
            mean=1.0,
            p10=1.0,
            p90=1.0,
            n_pairs=1,
            total_pairs=1,
            subsampled=False,
        )
        core_hamming = CoreHammingSummary(
            metric="hamming",
            nnd_k1=KnnBlock(top_candidates=top_knn, diversified_candidates=diversified_knn),
            nnd_k5=None,
            pairwise=PairwiseBlock(
                top_candidates=top_pairwise, diversified_candidates=diversified_pairwise, max_diversity_upper_bound=None
            ),
        )
        unweighted_knn = KnnBlock(top_candidates=top_knn, diversified_candidates=diversified_knn)
        entropy_block = EntropyBlock(
            top_candidates=EntropySummary(values=[0.0, 0.0], n=2),
            diversified_candidates=EntropySummary(values=[0.0, 0.0], n=2),
        )
        score_block = ScoreQuantilesBlock(
            top_candidates=ScoreQuantiles(p10=0.5, p50=1.0, p90=1.5, mean=1.0),
            diversified_candidates=ScoreQuantiles(p10=0.5, p50=1.0, p90=1.5, mean=1.0),
            top_candidates_global=None,
            max_diversity_upper_bound=None,
        )
        score_norm_summary = ScoreSummaryBlock(
            top_candidates=ScoreSummary(min=0.9, median=0.95, max=1.0),
            diversified_candidates=ScoreSummary(min=0.9, median=0.95, max=1.0),
        )
        diversity = DiversitySummary(
            candidate_pool_size=2,
            nnd_unweighted_k1=unweighted_knn,
            nnd_unweighted_median_top=1.0,
            nnd_unweighted_median_diversified=1.0,
            delta_nnd_unweighted_median=0.0,
            core_hamming=core_hamming,
            set_overlap_fraction=1.0,
            set_overlap_swaps=0,
            core_entropy=entropy_block,
            score_quantiles=score_block,
            score_norm_summary=score_norm_summary,
        )
        summary = PWMSamplingSummary(
            input_name="demo_pwm",
            regulator="regA",
            backend="fimo",
            pwm_consensus="AAAA",
            pwm_consensus_iupac="AAAA",
            pwm_consensus_score=1.0,
            pwm_theoretical_max_score=1.0,
            uniqueness_key="core",
            collapsed_by_core_identity=0,
            generated=10,
            target=10,
            target_sites=2,
            candidates_with_hit=9,
            eligible_raw=3,
            eligible_unique=2,
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
            tier_fractions=[0.001, 0.01, 0.09],
            tier_fractions_source="default",
            eligible_score_hist_edges=[0.0, 1.0, 2.0],
            eligible_score_hist_counts=[1, 1],
            tier_target_fraction=0.001,
            tier_target_required_unique=2000,
            tier_target_met=True,
            selection_policy="top_score",
            selection_relevance_norm=None,
            selection_pool_size_final=2,
            selection_pool_rung_fraction_used=1.0,
            selection_pool_min_score_norm_used=None,
            selection_pool_capped=False,
            selection_pool_cap_value=None,
            diversity=diversity,
            eligible_score_norm_by_tier={"tier0": {"min": 0.9, "median": 0.9, "max": 0.9}},
            mining_audit=None,
            motif_width=20,
            trim_window_length=16,
            trim_window_strategy="max_info",
            trim_window_start=2,
            trim_window_score=1.23,
            trimmed_width=16,
            trim_window_applied=True,
        )
        return [("regA", "AAAA", "dummy")], df, [summary]


def test_pool_manifest_includes_stage_a_sampling(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.9",
                    "run": {"id": "demo", "root": "."},
                    "inputs": [
                        {
                            "name": "demo_pwm",
                            "type": "pwm_meme",
                            "path": str(tmp_path / "motifs.meme"),
                            "sampling": {
                                "n_sites": 2,
                                "mining": {
                                    "batch_size": 1000,
                                    "budget": {
                                        "mode": "fixed_candidates",
                                        "candidates": 4,
                                    },
                                },
                                "bgfile": "inputs/bg.txt",
                            },
                        }
                    ],
                    "output": {
                        "targets": ["parquet"],
                        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                        "parquet": {"path": "outputs/tables/records.parquet"},
                    },
                    "generation": {
                        "sequence_length": 10,
                        "plan": [
                            {
                                "name": "default",
                                "quota": 1,
                                "sampling": {"include_inputs": ["demo_pwm"]},
                                "regulator_constraints": {"groups": []},
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
    assert stage_a_sampling["tier_fractions"] == [0.001, 0.01, 0.09]
    assert stage_a_sampling["tier_fractions_source"] == "default"
    assert stage_a_sampling["eligibility_rule"].startswith("best_hit_score")
    assert stage_a_sampling["retention_rule"] == "top_n_sites_by_best_hit_score"
    assert stage_a_sampling["fimo_thresh"] == 1.0
    assert stage_a_sampling["uniqueness_key"] == "core"
    assert stage_a_sampling["bgfile"] == "inputs/bg.txt"
    assert stage_a_sampling["background_source"] == "bgfile"
    hist = stage_a_sampling["eligible_score_hist"]
    assert hist[0]["regulator"] == "regA"
    assert hist[0]["pwm_consensus"] == "AAAA"
    assert hist[0]["pwm_consensus_iupac"] == "AAAA"
    assert hist[0]["pwm_consensus_score"] == 1.0
    assert hist[0]["pwm_theoretical_max_score"] == 1.0
    assert hist[0]["edges"] == [0.0, 1.0, 2.0]
    assert hist[0]["counts"] == [1, 1]
    assert hist[0]["tier0_score"] == 2.0
    assert hist[0]["tier1_score"] == 1.5
    assert hist[0]["tier2_score"] == 1.0
    assert hist[0]["tier_fractions"] == [0.001, 0.01, 0.09]
    assert hist[0]["tier_fractions_source"] == "default"
    assert hist[0]["eligible_score_norm_by_tier"] == {"tier0": {"min": 0.9, "median": 0.9, "max": 0.9}}
    assert hist[0]["selection_score_norm_max_raw"] is None
    assert hist[0]["selection_score_norm_clipped"] is None
    assert hist[0]["max_observed_score"] is None
    assert hist[0]["generated"] == 10
    assert hist[0]["candidates_with_hit"] == 9
    assert hist[0]["eligible_raw"] == 3
    assert hist[0]["eligible_unique"] == 2
    assert hist[0]["retained"] == 1
    assert hist[0]["mining_audit"] is None
    assert hist[0]["motif_width"] == 20
    assert hist[0]["trim_window_length"] == 16
    assert hist[0]["trim_window_strategy"] == "max_info"
    assert hist[0]["trim_window_start"] == 2
    assert hist[0]["trim_window_score"] == 1.23
    assert hist[0]["trimmed_width"] == 16
    assert hist[0]["trim_window_applied"] is True
