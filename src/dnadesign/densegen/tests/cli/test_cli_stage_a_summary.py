"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_stage_a_summary.py

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

from dnadesign.densegen.src.cli.main import _format_tier_counts, _stage_a_sampling_rows, app
from dnadesign.densegen.src.cli.render import stage_a_recap_tables
from dnadesign.densegen.src.core.artifacts.pool import PoolData
from dnadesign.densegen.src.core.stage_a.stage_a_metrics import (
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
from dnadesign.densegen.src.core.stage_a.stage_a_summary import PWMSamplingSummary
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable

pytestmark = pytest.mark.fimo

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _dummy_diversity_summary() -> DiversitySummary:
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
        top_candidates=EntropySummary(values=[0.1, 0.2, 0.3], n=2),
        diversified_candidates=EntropySummary(values=[0.1, 0.2, 0.3], n=2),
    )
    score_block = ScoreQuantilesBlock(
        top_candidates=ScoreQuantiles(p10=0.5, p50=1.0, p90=1.5, mean=1.0),
        diversified_candidates=ScoreQuantiles(p10=0.4, p50=0.9, p90=1.4, mean=0.9),
        top_candidates_global=None,
        max_diversity_upper_bound=None,
    )
    score_norm_summary = ScoreSummaryBlock(
        top_candidates=ScoreSummary(min=0.9, median=0.95, max=1.0),
        diversified_candidates=ScoreSummary(min=0.88, median=0.94, max=0.99),
    )
    return DiversitySummary(
        candidate_pool_size=50,
        nnd_unweighted_k1=unweighted_knn,
        nnd_unweighted_median_top=1.0,
        nnd_unweighted_median_diversified=1.0,
        delta_nnd_unweighted_median=0.0,
        core_hamming=core_hamming,
        set_overlap_fraction=0.5,
        set_overlap_swaps=1,
        core_entropy=entropy_block,
        score_quantiles=score_block,
        score_norm_summary=score_norm_summary,
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
              schema_version: "2.9"
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
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 30
                plan:
                  - name: default
                    sequences: 1
                    sampling:
                      include_inputs: [toy_sites]
                    regulator_constraints:
                      groups:
                        - name: all
                          members: [TF1, TF2, TF3]
                          min_required: 1
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
              schema_version: "2.9"
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
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 30
                plan:
                  - name: default
                    sequences: 1
                    sampling:
                      include_inputs: [demo_pwm]
                    regulator_constraints:
                      groups: []
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


def test_stage_a_build_pool_compact_recap_omits_verbose_columns(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "has_hit" not in result.output
    assert "eligible_raw" not in result.output
    assert "Δscore_norm" not in result.output
    assert "Δdiv" not in result.output


def test_stage_a_build_pool_verbose_recap_includes_verbose_columns(tmp_path: Path) -> None:
    cfg_path = _write_stage_a_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["stage-a", "build-pool", "--verbose", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "has_hit" in result.output
    assert "eligible_raw" in result.output


def test_stage_a_recap_tables_include_verbose_headers() -> None:
    summary = PWMSamplingSummary(
        input_name="demo",
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
        selection_relevance_norm="minmax_raw_score",
        selection_pool_size_final=50,
        selection_pool_rung_fraction_used=0.001,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
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
    tables = stage_a_recap_tables(rows, display_map_by_input={}, show_motif_ids=True, verbose=True)
    headers = [col.header for col in tables[0][1].columns]
    assert "score_norm top (min/med/max)" in headers
    assert "score_norm div (min/med/max)" in headers
    assert "pairwise top" in headers
    assert "pairwise div" in headers


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
        pwm_consensus_iupac="AAAA",
        pwm_consensus_score=1.0,
        pwm_theoretical_max_score=1.0,
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
        selection_relevance_norm="minmax_raw_score",
        selection_pool_size_final=50,
        selection_pool_rung_fraction_used=0.001,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
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
    assert rows[0]["diversity_pool"] == "50"


def test_stage_a_sampling_rows_tier_target_omits_required_unique() -> None:
    summary = PWMSamplingSummary(
        input_name="demo",
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
        selection_relevance_norm="minmax_raw_score",
        selection_pool_size_final=50,
        selection_pool_rung_fraction_used=0.001,
        selection_pool_min_score_norm_used=None,
        selection_pool_capped=False,
        selection_pool_cap_value=None,
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
