"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_end_to_end_score_sampling.py

End-to-end Stage-A score-only sampling across two PWM artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.densegen.src.adapters.sources import pwm_sampling
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact
from dnadesign.densegen.src.core.pipeline import default_deps
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.src.viz.plotting import plot_stage_a_summary

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _artifact_paths() -> list[str]:
    base = Path(__file__).resolve().parents[1] / "workspaces" / "demo_meme_two_tf" / "inputs" / "motif_artifacts"
    return [
        str(base / "lexA__meme_suite_meme__lexA_CTGTATAWAWWHACA.json"),
        str(base / "cpxR__meme_suite_meme__cpxR_MANWWHTTTAM.json"),
    ]


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "densegen": {
                    "schema_version": "2.7",
                    "run": {"id": "stage_a_e2e", "root": "."},
                    "inputs": [
                        {
                            "name": "demo_pwm",
                            "type": "pwm_artifact_set",
                            "paths": _artifact_paths(),
                            "sampling": {
                                "strategy": "stochastic",
                                "n_sites": 5,
                                "mining": {
                                    "batch_size": 200,
                                    "budget": {"mode": "fixed_candidates", "candidates": 2000},
                                    "log_every_batches": 1,
                                },
                                "length": {"policy": "range", "range": [15, 20]},
                                "keep_all_candidates_debug": True,
                            },
                        }
                    ],
                    "output": {
                        "targets": ["parquet"],
                        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                        "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
                    },
                    "generation": {
                        "sequence_length": 60,
                        "quota": 1,
                        "plan": [{"name": "default", "quota": 1}],
                    },
                    "solver": {"backend": "CBC", "strategy": "iterate"},
                    "runtime": {
                        "round_robin": False,
                        "arrays_generated_before_resample": 5,
                        "min_count_per_tf": 0,
                        "max_duplicate_solutions": 1,
                        "stall_seconds_before_resample": 1,
                        "stall_warning_every_seconds": 1,
                        "max_resample_attempts": 1,
                        "max_total_resamples": 1,
                        "max_seconds_per_plan": 1,
                        "max_failed_solutions": 0,
                        "checkpoint_every": 0,
                        "leaderboard_every": 1,
                    },
                    "logging": {"log_dir": "outputs/logs", "level": "INFO"},
                    "postprocess": {"pad": {"mode": "off"}},
                }
            }
        )
    )
    return cfg_path


def _top_scoring_sequences(candidates: pd.DataFrame, count: int) -> list[str]:
    ranked_rows = candidates.sort_values(["best_hit_score", "sequence"], ascending=[False, True])
    ranked: list[pwm_sampling.FimoCandidate] = []
    for row in ranked_rows.itertuples():
        ranked.append(
            pwm_sampling.FimoCandidate(
                seq=str(row.sequence),
                score=float(row.best_hit_score),
                start=int(row.start),
                stop=int(row.stop),
                strand=str(row.strand or "+"),
                matched_sequence=str(row.matched_sequence) if row.matched_sequence else None,
            )
        )
    deduped, _collapsed = pwm_sampling._collapse_by_core_identity(ranked)
    return [cand.seq for cand in deduped[:count]]


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_stage_a_end_to_end_score_sampling(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    outputs_root = tmp_path / "outputs"
    out_dir = outputs_root / "pools"
    deps = default_deps()
    artifact, pool_data = build_pool_artifact(
        cfg=cfg,
        cfg_path=cfg_path,
        deps=deps,
        rng=np.random.default_rng(0),
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    pool = pool_data["demo_pwm"]
    assert pool.df is not None
    df = pool.df
    for col in ("best_hit_score", "tier", "rank_within_regulator", "tfbs_sequence", "tier_target_eligible_unique"):
        assert col in df.columns

    regulators = sorted({str(tf) for tf in df["tf"].tolist()})
    assert len(regulators) == 2
    for reg in regulators:
        sub = df[df["tf"].astype(str) == reg].copy()
        assert not sub.empty
        assert (sub["best_hit_score"] > 0).all()
        assert int(sub["rank_within_regulator"].min()) >= 1

    candidates_dir = outputs_root / "pools" / "candidates" / "demo_pwm"
    candidate_files = sorted(candidates_dir.glob("candidates__*.parquet"))
    assert candidate_files
    candidates = pd.concat([pd.read_parquet(path) for path in candidate_files], ignore_index=True)
    accepted = candidates[candidates["accepted"]].copy()
    assert not accepted.empty
    for reg in regulators:
        sub = df[df["tf"].astype(str) == reg].copy()
        expected = _top_scoring_sequences(
            accepted[accepted["motif_label"].astype(str) == reg],
            len(sub),
        )
        retained_sorted = sub.sort_values(["best_hit_score", "tfbs_sequence"], ascending=[False, True])[
            "tfbs_sequence"
        ].tolist()
        assert retained_sorted == expected

    entry = artifact.entry_for("demo_pwm")
    sampling = entry.stage_a_sampling
    assert sampling is not None
    for row in sampling.get("eligible_score_hist") or []:
        assert "tier0_score" in row
        assert "tier1_score" in row
        assert "tier2_score" in row
        assert "generated" in row
        assert "candidates_with_hit" in row
        assert "eligible" in row
        assert "eligible_unique" in row
        assert "retained" in row

    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "stage_a_summary.png"
    paths = plot_stage_a_summary(
        df=pd.DataFrame(),
        out_path=out_path,
        pools={"demo_pwm": df},
        pool_manifest=artifact,
    )
    assert len(paths) == 3
    assert paths[0].exists()
