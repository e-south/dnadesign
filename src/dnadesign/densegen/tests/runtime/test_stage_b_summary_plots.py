"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_stage_b_summary_plots.py

Focused coverage for contract-grade Stage-B summary plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.densegen.src.core.artifacts.pool import TFBSPoolArtifact
from dnadesign.densegen.src.viz.plot_stage_b_summary import (
    _deployed_tfbs_frame,
    _unique_deployed_length_summary,
    plot_accepted_arrays_by_plan,
    plot_plan_by_regulator_heatmap,
    plot_retained_vs_deployed_length_shift,
    plot_retained_vs_deployed_tier_mix,
    plot_score_strata_and_deployed_length_by_regulator,
    plot_upstream_evidence_quality_summary,
    plot_used_unique_vs_retained,
)


def _pool_manifest(tmp_path: Path) -> TFBSPoolArtifact:
    manifest_path = tmp_path / "pool_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.6",
                "run_id": "demo",
                "run_root": ".",
                "config_path": "config.yaml",
                "inputs": [
                    {
                        "name": "lexA_pwm",
                        "type": "binding_sites",
                        "pool_path": "lexA_pwm__pool.parquet",
                        "rows": 3,
                        "columns": ["input_name", "tf", "tfbs_sequence", "tier"],
                        "pool_mode": "tfbs",
                        "stage_a_sampling": {
                            "eligible_score_hist": [
                                {
                                    "regulator": "lexA",
                                    "pwm_consensus_score": 3.0,
                                    "pwm_theoretical_max_score": 3.0,
                                    "candidates_with_hit": 18,
                                    "eligible_unique": 8,
                                    "retained": 3,
                                    "edges": [0.0, 1.0, 2.0, 3.0],
                                    "counts": [1, 3, 2],
                                }
                            ]
                        },
                    },
                    {
                        "name": "cpxR_pwm",
                        "type": "binding_sites",
                        "pool_path": "cpxR_pwm__pool.parquet",
                        "rows": 3,
                        "columns": ["input_name", "tf", "tfbs_sequence", "tier"],
                        "pool_mode": "tfbs",
                        "stage_a_sampling": {
                            "eligible_score_hist": [
                                {
                                    "regulator": "cpxR",
                                    "pwm_consensus_score": 2.5,
                                    "pwm_theoretical_max_score": 2.5,
                                    "candidates_with_hit": 12,
                                    "eligible_unique": 6,
                                    "retained": 3,
                                    "edges": [0.0, 0.8, 1.6, 2.4],
                                    "counts": [2, 2, 1],
                                }
                            ]
                        },
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TFBSPoolArtifact.load(manifest_path)


def _pools() -> dict[str, pd.DataFrame]:
    return {
        "lexA_pwm": pd.DataFrame(
            {
                "input_name": ["lexA_pwm"] * 3,
                "tf": ["lexA"] * 3,
                "tfbs_sequence": ["AAAA", "AAAAT", "AAAACC"],
                "tfbs_core": ["AAAA", "AAAT", "AACC"],
                "best_hit_score": [2.8, 2.3, 1.9],
                "tier": [0, 1, 2],
            }
        ),
        "cpxR_pwm": pd.DataFrame(
            {
                "input_name": ["cpxR_pwm"] * 3,
                "tf": ["cpxR"] * 3,
                "tfbs_sequence": ["CCCCCCCC", "CCCCCCCCC", "CCGCCC"],
                "tfbs_core": ["CCCCCC", "CCCCCG", "CCGCCC"],
                "best_hit_score": [2.4, 2.0, 1.6],
                "tier": [0, 1, 2],
            }
        ),
    }


def _output_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["row-1", "row-2", "row-3", "row-4"],
            "densegen__plan": [
                "background_only",
                "ethanol",
                "ethanol",
                "ciprofloxacin",
            ],
            "densegen__used_tfbs_detail": [
                [
                    {"part_kind": "tfbs", "regulator": "lexA", "sequence": "AAAA", "length": 4},
                    {"part_kind": "tfbs", "regulator": "cpxR", "sequence": "CCCCCCCC", "length": 8},
                ],
                [
                    {"part_kind": "tfbs", "regulator": "lexA", "sequence": "AAAAT", "length": 5},
                ],
                [
                    {"part_kind": "tfbs", "regulator": "cpxR", "sequence": "CCGCCC", "length": 6},
                ],
                [
                    {"part_kind": "tfbs", "regulator": "lexA", "sequence": "AAAA", "length": 4},
                    {"part_kind": "tfbs", "regulator": "background", "sequence": "TTTT", "length": 4},
                ],
            ],
        }
    )


def test_stage_b_summary_plots_generate_expected_artifacts(tmp_path: Path) -> None:
    output_records = _output_records()
    pools = _pools()
    manifest = _pool_manifest(tmp_path)

    paths = []
    paths.extend(plot_accepted_arrays_by_plan(output_records, tmp_path / "accepted_arrays_by_plan.png", style={}))
    paths.extend(plot_plan_by_regulator_heatmap(output_records, tmp_path / "plan_by_regulator_heatmap.png", style={}))
    paths.extend(
        plot_retained_vs_deployed_length_shift(
            output_records,
            tmp_path / "retained_vs_deployed_length_shift.png",
            pools=pools,
            style={},
        )
    )
    paths.extend(
        plot_used_unique_vs_retained(
            output_records,
            tmp_path / "used_unique_vs_retained.png",
            pools=pools,
            style={},
        )
    )
    paths.extend(
        plot_retained_vs_deployed_tier_mix(
            output_records,
            tmp_path / "retained_vs_deployed_tier_mix.png",
            pools=pools,
            style={},
        )
    )
    paths.extend(
        plot_upstream_evidence_quality_summary(
            output_records,
            tmp_path / "upstream_evidence_quality_summary.png",
            pool_manifest=manifest,
            style={},
        )
    )
    paths.extend(
        plot_score_strata_and_deployed_length_by_regulator(
            output_records,
            tmp_path / "score_strata_and_deployed_length_by_regulator.png",
            pools=pools,
            pool_manifest=manifest,
            style={},
        )
    )

    relative_paths = {str(path.relative_to(tmp_path)) for path in paths}
    assert relative_paths == {
        "stage_b_summary/accepted_arrays_by_plan.png",
        "stage_b_summary/plan_by_regulator_heatmap.png",
        "stage_b_summary/retained_vs_deployed_length_shift.png",
        "stage_b_summary/retained_vs_deployed_tier_mix.png",
        "stage_b_summary/score_strata_and_deployed_length_by_regulator.png",
        "stage_b_summary/upstream_evidence_quality_summary.png",
        "stage_b_summary/used_unique_vs_retained.png",
    }
    assert all(path.exists() for path in paths)
    plt.close("all")


def test_bridge_plot_counts_unique_deployed_sequences_by_length(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_save(fig, out: Path, *, style=None) -> None:
        del style
        captured["fig"] = fig
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"plot")

    monkeypatch.setattr("dnadesign.densegen.src.viz.plot_stage_b_summary._save_figure", _capture_save)
    output_records = _output_records()
    pools = _pools()
    manifest = _pool_manifest(tmp_path)

    plot_score_strata_and_deployed_length_by_regulator(
        output_records,
        tmp_path / "bridge.png",
        pools=pools,
        pool_manifest=manifest,
        style={},
    )

    fig = captured["fig"]
    assert len(fig.axes) == 2
    ax_score = fig.axes[0]
    ax_length = fig.axes[-1]
    x_max = ax_length.get_xlim()[1]
    assert x_max <= 3.0
    assert ax_length.get_xlabel() == "Unique deployed TFBS count"
    assert ax_length.get_legend().get_title().get_text() == "Length"
    assert "Background" in [tick.get_text() for tick in ax_length.get_yticklabels()]
    assert [tick.get_text() for tick in ax_score.get_yticklabels()] == ["LexA", "CpxR"]
    bridge_text = "\n".join(text.get_text() for ax in fig.axes[:1] for text in ax.texts if text.get_text())
    assert "Used: 2" in bridge_text
    assert "AVG. pairwise" in bridge_text
    assert "hamming 1.0" in bridge_text
    assert "Seq pairwise" not in bridge_text
    assert "hamming n/a" not in bridge_text
    plt.close(fig)


def test_unique_deployed_length_summary_deduplicates_sequences_and_orders_lengths_descending() -> None:
    deployed = _deployed_tfbs_frame(_output_records())
    (
        unique_deployed,
        counts_by_regulator,
        counts_by_regulator_and_length,
        length_values,
    ) = _unique_deployed_length_summary(deployed)

    assert unique_deployed.shape[0] == 5
    assert counts_by_regulator == {"background": 1, "cpxR": 2, "lexA": 2}
    assert length_values == [8, 6, 5, 4]
    assert counts_by_regulator_and_length[("lexA", 5)] == 1
    assert counts_by_regulator_and_length[("lexA", 4)] == 1
    assert counts_by_regulator_and_length[("cpxR", 8)] == 1
    assert counts_by_regulator_and_length[("cpxR", 6)] == 1
    assert counts_by_regulator_and_length[("background", 4)] == 1
    assert (
        sum(count for (regulator, _length), count in counts_by_regulator_and_length.items() if regulator == "lexA")
        == counts_by_regulator["lexA"]
    )
    assert (
        sum(count for (regulator, _length), count in counts_by_regulator_and_length.items() if regulator == "cpxR")
        == counts_by_regulator["cpxR"]
    )
    assert (
        sum(
            count for (regulator, _length), count in counts_by_regulator_and_length.items() if regulator == "background"
        )
        == counts_by_regulator["background"]
    )
