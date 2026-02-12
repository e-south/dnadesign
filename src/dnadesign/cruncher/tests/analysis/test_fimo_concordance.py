"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_fimo_concordance.py

Validates optimizer-vs-FIMO concordance diagnostics and plotting behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.fimo_concordance import build_fimo_concordance_table
from dnadesign.cruncher.analysis.plots.fimo_concordance import plot_optimizer_vs_fimo
from dnadesign.cruncher.core.pwm import PWM


def _toy_pwm(name: str, width: int = 4) -> PWM:
    matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(width)]
    return PWM(name=name, matrix=matrix)


def test_fimo_concordance_keeps_rows_and_assigns_zero_for_missing_hits(tmp_path, monkeypatch) -> None:
    points_df = pd.DataFrame(
        {
            "objective_scalar": [0.8, 0.3],
            "sequence": ["ACGTACGT", "TTTTTTTT"],
            "phase": ["draw", "draw"],
            "sweep": [1, 2],
        }
    )
    tf_names = ["lexA", "cpxR"]
    pwms = {tf: _toy_pwm(tf) for tf in tf_names}

    def _fake_run_fimo(*, motif_path, fasta_path, bidirectional, threshold, tool_path):
        if motif_path.stem == "lexA":
            return [
                {
                    "sequence_name": "seq_000000",
                    "start": 1,
                    "stop": 4,
                    "strand": "+",
                    "score": 3.0,
                    "p_value": 1e-4,
                }
            ]
        return []

    monkeypatch.setattr("dnadesign.cruncher.analysis.fimo_concordance._run_fimo", _fake_run_fimo)

    concordance_df, summary = build_fimo_concordance_table(
        points_df=points_df,
        tf_names=tf_names,
        pwms=pwms,
        bidirectional=True,
        threshold=1.0,
        work_dir=tmp_path,
        tool_path=None,
    )

    assert len(concordance_df) == 2
    assert "objective_scalar" in concordance_df.columns
    assert "fimo_joint_weakest_score" in concordance_df.columns
    assert float(concordance_df["fimo_joint_weakest_score"].iloc[0]) == 0.0
    assert float(concordance_df["fimo_joint_weakest_score"].iloc[1]) == 0.0
    assert summary["n_rows"] == 2


def test_plot_optimizer_vs_fimo_smoke(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "objective_scalar": [0.1, 0.4, 0.9],
            "fimo_joint_weakest_score": [0.0, 1.2, 2.4],
        }
    )
    out_path = tmp_path / "plot__optimizer_vs_fimo.png"
    metadata = plot_optimizer_vs_fimo(df, out_path, dpi=150, png_compress_level=9)

    assert out_path.exists()
    assert metadata["n_points"] == 3
    assert "Cruncher" in str(metadata["title"])
    assert float(metadata["low_score_fraction"]) == 1.0 / 3.0
    assert int(metadata["trend_points"]) >= 0


def test_fimo_concordance_forwards_tool_path(tmp_path, monkeypatch) -> None:
    points_df = pd.DataFrame(
        {
            "objective_scalar": [0.7],
            "sequence": ["ACGTACGT"],
            "phase": ["draw"],
            "sweep": [1],
        }
    )
    tf_names = ["lexA"]
    pwms = {"lexA": _toy_pwm("lexA")}
    configured_tool_path = tmp_path / "meme_bin"
    configured_tool_path.mkdir()
    seen: dict[str, object] = {}

    def _fake_run_fimo(*, motif_path, fasta_path, bidirectional, threshold, tool_path):
        seen["tool_path"] = tool_path
        return []

    monkeypatch.setattr("dnadesign.cruncher.analysis.fimo_concordance._run_fimo", _fake_run_fimo)

    build_fimo_concordance_table(
        points_df=points_df,
        tf_names=tf_names,
        pwms=pwms,
        bidirectional=True,
        threshold=1.0,
        work_dir=tmp_path,
        tool_path=configured_tool_path,
    )

    assert seen["tool_path"] == configured_tool_path
