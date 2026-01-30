"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_stage_a_summary.py

CLI coverage for Stage-A build-pool summaries.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ABOUTME: CLI coverage for Stage-A build-pool length summaries.
# ABOUTME: Ensures pooled TFBS length stats are surfaced in stdout.
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.densegen.src.adapters.sources.stage_a_metrics import (
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
)
from dnadesign.densegen.src.adapters.sources.stage_a_summary import PWMSamplingSummary
from dnadesign.densegen.src.cli import _format_tier_counts, _stage_a_sampling_rows, app
from dnadesign.densegen.src.core.artifacts.pool import PoolData
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _dummy_diversity_summary() -> DiversitySummary:
    baseline_knn = KnnSummary(
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
    actual_knn = KnnSummary(
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
    baseline_pairwise = PairwiseSummary(
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
    actual_pairwise = PairwiseSummary(
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
        nnd_k1=KnnBlock(baseline=baseline_knn, actual=actual_knn),
        nnd_k5=None,
        pairwise=PairwiseBlock(baseline=baseline_pairwise, actual=actual_pairwise, upper_bound=None),
    )
    entropy_block = EntropyBlock(
        baseline=EntropySummary(values=[0.1, 0.2, 0.3], n=2),
        actual=EntropySummary(values=[0.1, 0.2, 0.3], n=2),
    )
    score_block = ScoreQuantilesBlock(
        baseline=ScoreQuantiles(p10=0.5, p50=1.0, p90=1.5, mean=1.0),
        actual=ScoreQuantiles(p10=0.4, p50=0.9, p90=1.4, mean=0.9),
        baseline_global=None,
        upper_bound=None,
    )
    return DiversitySummary(
        candidate_pool_size=50,
        shortlist_target=250,
        core_hamming=core_hamming,
        set_overlap_fraction=0.5,
        set_overlap_swaps=1,
        core_entropy=entropy_block,
        score_quantiles=score_block,
    )


def _write_stage_a_config(tmp_path: Path) -> Path:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "sites.csv").write_text(
        textwrap.dedent(
            """
            tf,tfbs
            TF1,AAAAAAAAAA
            TF2,CCCCCCCCCCCC
            TF3,GGGGGGGGGGGGGG
            """
        ).strip()
        + "\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.7"
              run:
                id: demo
                root: "."
              inputs:
                - name: toy_sites
                  type: binding_sites
                  path: {inputs_dir / "sites.csv"}
                  format: csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 30
                quota: 1
                plan:
                  - name: default
                    quota: 1
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    return cfg_path


def _write_pwm_stage_a_config(tmp_path: Path) -> Path:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    meme_path = inputs_dir / "motifs.meme"
    meme_path.write_text(
        textwrap.dedent(
            """
            MEME version 4

            ALPHABET= ACGT

            Background letter frequencies
            A 0.25 C 0.25 G 0.25 T 0.25

            MOTIF M1
            letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
            0.8 0.1 0.05 0.05
            0.1 0.7 0.1 0.1
            0.1 0.1 0.7 0.1

            MOTIF M2
            letter-probability matrix: alength= 4 w= 2 nsites= 10 E= 0
            0.6 0.2 0.1 0.1
            0.2 0.6 0.1 0.1
            """
        ).strip()
        + "\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.7"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_pwm
                  type: pwm_meme
                  path: {meme_path}
                  sampling:
                    strategy: consensus
                    n_sites: 1
                    mining:
                      batch_size: 1
                      budget:
                        mode: fixed_candidates
                        candidates: 1
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 30
                quota: 1
                plan:
                  - name: default
                    quota: 1
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    return cfg_path


def test_stage_a_build_pool_reports_sampling_recap(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Stage-A sampling recap" in result.output
    assert "Input: toy_sites" in result.output
    assert "generated" in result.output
    assert "tier fill" in result.output
    assert "score" in result.output
    assert "retained" in result.output


def test_stage_a_build_pool_accepts_fresh_flag(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "--fresh", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output


def test_stage_a_build_pool_logs_initialized(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Logging initialized" in result.output


def test_stage_a_build_pool_reports_plan(tmp_path: Path) -> None:
    cfg_path = _write_pwm_stage_a_config(tmp_path)
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Stage-A plan" in result.output
    assert "M1" in result.output
    assert "M2" in result.output


def test_tier_rows_include_zero_counts() -> None:
    label = _format_tier_counts([2, 0, 1, 0], [1, 0, 0, 0])
    assert label == "t0 2/1 | t1 0/0 | t2 1/0 | t3 0/0"


def test_stage_a_sampling_rows_include_pool_headroom() -> None:
    summary = PWMSamplingSummary(
        input_name="demo",
        regulator="regA",
        backend="fimo",
        pwm_consensus="AAAA",
        uniqueness_key="core",
        collapsed_by_core_identity=0,
        generated=10,
        target=10,
        target_sites=2,
        candidates_with_hit=9,
        eligible_raw=8,
        eligible_unique=5,
        retained=2,
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
        eligible_score_hist_edges=[0.0, 1.0],
        eligible_score_hist_counts=[1],
        selection_policy="mmr",
        selection_alpha=0.9,
        selection_shortlist_k=50,
        selection_shortlist_min=10,
        selection_shortlist_factor=5,
        selection_shortlist_target=250,
        selection_pool_source="shortlist_k",
        diversity=_dummy_diversity_summary(),
        mining_audit=None,
    )
    pool = PoolData(
        name="demo",
        input_type="pwm_meme",
        pool_mode="tfbs",
        df=None,
        sequences=[],
        pool_path=Path("demo.parquet"),
        summaries=[summary],
    )
    rows = _stage_a_sampling_rows({"demo": pool})
    assert rows[0]["diversity_pool"] == "50/250 (shortlist_k)"


def test_stage_a_sampling_rows_tier_target_omits_required_unique() -> None:
    summary = PWMSamplingSummary(
        input_name="demo",
        regulator="regA",
        backend="fimo",
        pwm_consensus="AAAA",
        uniqueness_key="core",
        collapsed_by_core_identity=0,
        generated=10,
        target=10,
        target_sites=2,
        candidates_with_hit=9,
        eligible_raw=8,
        eligible_unique=5,
        retained=2,
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
        eligible_score_hist_edges=[0.0, 1.0],
        eligible_score_hist_counts=[1],
        tier_target_fraction=0.001,
        tier_target_required_unique=5000,
        tier_target_met=False,
        selection_policy="mmr",
        selection_alpha=0.9,
        selection_shortlist_k=50,
        selection_shortlist_min=10,
        selection_shortlist_factor=5,
        selection_shortlist_target=250,
        selection_pool_source="shortlist_k",
        diversity=_dummy_diversity_summary(),
        mining_audit=None,
    )
    pool = PoolData(
        name="demo",
        input_type="pwm_meme",
        pool_mode="tfbs",
        df=None,
        sequences=[],
        pool_path=Path("demo.parquet"),
        summaries=[summary],
    )
    rows = _stage_a_sampling_rows({"demo": pool})
    assert "need" not in rows[0]["tier_target"]
