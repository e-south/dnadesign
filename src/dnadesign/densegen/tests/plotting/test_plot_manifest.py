"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_manifest.py

Plot manifest coverage for plot generation outputs.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.viz import plotting as plotting_module
from dnadesign.densegen.src.viz.plotting import run_plots_from_config


def _diversity_block(core_len: int) -> dict:
    bins = [0, 1, 2]
    counts = [0, 2, 0]
    return {
        "candidate_pool_size": 2,
        "nnd_unweighted_k1": {
            "top_candidates": {
                "bins": bins,
                "counts": counts,
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
                "frac_le_1": 1.0,
                "n": 2,
                "subsampled": False,
                "k": 1,
            },
            "diversified_candidates": {
                "bins": bins,
                "counts": counts,
                "median": 1.0,
                "p05": 1.0,
                "p95": 1.0,
                "frac_le_1": 1.0,
                "n": 2,
                "subsampled": False,
                "k": 1,
            },
        },
        "nnd_unweighted_median_top": 1.0,
        "nnd_unweighted_median_diversified": 1.0,
        "delta_nnd_unweighted_median": 0.0,
        "core_hamming": {
            "metric": "hamming",
            "nnd_k1": {
                "k": 1,
                "top_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
                "diversified_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "frac_le_1": 1.0,
                    "n": 2,
                    "subsampled": False,
                },
            },
            "nnd_k5": None,
            "pairwise": {
                "top_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "diversified_candidates": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
                "max_diversity_upper_bound": {
                    "bins": bins,
                    "counts": counts,
                    "median": 1.0,
                    "mean": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                    "n_pairs": 1,
                    "total_pairs": 1,
                },
            },
        },
        "set_overlap_fraction": 1.0,
        "set_overlap_swaps": 0,
        "core_entropy": {
            "top_candidates": {"values": [0.0] * core_len, "n": 2},
            "diversified_candidates": {"values": [0.0] * core_len, "n": 2},
        },
        "score_quantiles": {
            "top_candidates": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "diversified_candidates": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "top_candidates_global": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
            "max_diversity_upper_bound": {"p10": 1.0, "p50": 1.5, "p90": 2.0, "mean": 1.5},
        },
    }


def _write_config(
    path: Path,
    *,
    plots_default: list[str],
    plots_options: dict[str, dict[str, object]] | None = None,
) -> None:
    options = plots_options or {}
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              format: png
              default: PLACEHOLDER_DEFAULT
              options: PLACEHOLDER_OPTIONS
            """
        )
        .strip()
        .replace("PLACEHOLDER_DEFAULT", json.dumps(plots_default))
        .replace("PLACEHOLDER_OPTIONS", json.dumps(options))
        + "\n"
    )


def _write_pool_manifest(run_root: Path) -> None:
    pools_dir = run_root / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["tfA", "tfA", "tfB"],
            "tfbs": ["AAAA", "AAAAT", "AAAAAA"],
            "tfbs_core": ["AAAA", "AAAT", "AAAAAA"],
            "best_hit_score": [7.0, 9.0, 5.5],
            "tier": [1, 0, 2],
            "rank_within_regulator": [2, 1, 1],
            "selection_rank": [2, 1, 1],
            "nearest_selected_similarity": [0.5, 0.0, 0.0],
            "selection_score_norm": [0.25, 1.0, 1.0],
            "nearest_selected_distance_norm": [0.5, None, None],
            "motif_id": ["m1", "m1", "m2"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
                "stage_a_sampling": {
                    "backend": "fimo",
                    "tier_scheme": "pct_0.1_1_9",
                    "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
                    "retention_rule": "top_n_sites_by_best_hit_score",
                    "fimo_thresh": 1.0,
                    "bgfile": None,
                    "background_source": "motif_background",
                    "eligible_score_hist": [
                        {
                            "regulator": "tfA",
                            "pwm_consensus": "AAAA",
                            "pwm_consensus_iupac": "AAAA",
                            "pwm_consensus_score": 10.0,
                            "pwm_theoretical_max_score": 10.0,
                            "edges": [4.0, 6.0, 8.0, 10.0],
                            "counts": [0, 1, 1],
                            "tier0_score": 9.0,
                            "tier1_score": 7.0,
                            "tier2_score": 6.0,
                            "tier_fractions": [0.001, 0.01, 0.09],
                            "tier_fractions_source": "default",
                            "generated": 10,
                            "candidates_with_hit": 9,
                            "eligible_raw": 8,
                            "eligible_unique": 3,
                            "retained": 2,
                            "selection_policy": "mmr",
                            "selection_alpha": 0.9,
                            "selection_similarity": "weighted_hamming_tolerant",
                            "selection_relevance_norm": "minmax_raw_score",
                            "selection_pool_size_final": 50,
                            "selection_pool_rung_fraction_used": 0.001,
                            "selection_pool_min_score_norm_used": None,
                            "selection_pool_capped": False,
                            "selection_pool_cap_value": None,
                            "diversity": _diversity_block(core_len=4),
                            "mining_audit": None,
                            "padding_audit": None,
                        },
                        {
                            "regulator": "tfB",
                            "pwm_consensus": "AAAAAA",
                            "pwm_consensus_iupac": "AAAAAA",
                            "pwm_consensus_score": 6.0,
                            "pwm_theoretical_max_score": 6.0,
                            "edges": [4.0, 6.0, 8.0],
                            "counts": [1, 0],
                            "tier0_score": 5.5,
                            "tier1_score": None,
                            "tier2_score": None,
                            "tier_fractions": [0.001, 0.01, 0.09],
                            "tier_fractions_source": "default",
                            "generated": 5,
                            "candidates_with_hit": 4,
                            "eligible_raw": 3,
                            "eligible_unique": 2,
                            "retained": 1,
                            "selection_policy": "mmr",
                            "selection_alpha": 0.9,
                            "selection_similarity": "weighted_hamming_tolerant",
                            "selection_relevance_norm": "minmax_raw_score",
                            "selection_pool_size_final": 10,
                            "selection_pool_rung_fraction_used": 0.001,
                            "selection_pool_min_score_norm_used": None,
                            "selection_pool_capped": False,
                            "selection_pool_cap_value": None,
                            "diversity": _diversity_block(core_len=6),
                            "mining_audit": None,
                            "padding_audit": None,
                        },
                    ],
                },
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_pool_manifest_without_sampling(run_root: Path) -> None:
    pools_dir = run_root / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["TF_A", "TF_B", "TF_C"],
            "tfbs": ["AAAA", "CCCC", "GGGG"],
            "tfbs_core": ["AAAA", "CCCC", "GGGG"],
            "best_hit_score": [1.0, 1.0, 1.0],
            "tier": [0, 0, 0],
            "rank_within_regulator": [1, 1, 1],
            "selection_rank": [1, 1, 1],
            "nearest_selected_similarity": [0.0, 0.0, 0.0],
            "selection_score_norm": [1.0, 1.0, 1.0],
            "nearest_selected_distance_norm": [None, None, None],
            "motif_id": ["m1", "m2", "m3"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
    manifest = {
        "schema_version": "1.6",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
                "stage_a_sampling": None,
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def test_plot_manifest_written(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    names = {item["name"] for item in payload.get("plots", [])}
    assert "stage_a_summary" in names
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_a/pool_tiers.png" in paths
    assert "stage_a/diversity.png" in paths
    stage_a_entries = [item for item in payload.get("plots", []) if item.get("name") == "stage_a_summary"]
    assert stage_a_entries
    assert all(str(item.get("plan_name") or "") == "stage_a" for item in stage_a_entries)
    for item in payload.get("plots", []):
        assert "plot_id" in item
        assert "group" in item
        assert "family" in item
        assert "variant" in item


def test_stage_a_plots_without_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    plots_dir = run_root / "outputs" / "plots"
    overview_plot = plots_dir / "stage_a" / "pool_tiers.png"
    assert overview_plot.exists()
    diversity_plot = plots_dir / "stage_a" / "diversity.png"
    assert diversity_plot.exists()


def test_stage_a_plots_skip_missing_sampling_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest_without_sampling(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    paths = {item.get("path") for item in payload.get("plots", [])}
    assert "stage_a/no_stage_a_panels.png" in paths
    stage_a_entries = [item for item in payload.get("plots", []) if item.get("name") == "stage_a_summary"]
    assert stage_a_entries
    assert all(str(item.get("plan_name") or "") == "stage_a" for item in stage_a_entries)
    assert (run_root / "outputs" / "plots" / "stage_a" / "no_stage_a_panels.png").exists()


def test_plot_run_removes_legacy_flat_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    legacy = run_root / "outputs" / "plots" / "stage_a_summary__pool_tiers.png"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy")
    assert legacy.exists()

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    assert not legacy.exists()
    assert (run_root / "outputs" / "plots" / "stage_a" / "pool_tiers.png").exists()


def test_plot_runner_rejects_unknown_plot_before_loading_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)
    calls = {"load_records": 0}

    def _fail_if_called(*_args, **_kwargs):
        calls["load_records"] += 1
        raise AssertionError("records loader should not run for invalid --only plot names")

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        _fail_if_called,
    )

    with pytest.raises(ValueError, match="Unknown plot name requested: definitely_missing"):
        run_plots_from_config(loaded.root, cfg_path, only="definitely_missing")
    assert calls["load_records"] == 0


def test_run_health_plot_requests_projected_output_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["run_health"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    loaded = load_config(cfg_path)
    captured: dict[str, list[str] | None] = {"columns": None}

    def _fake_load_records_from_config(*_args, **kwargs):
        cols = kwargs.get("columns")
        captured["columns"] = list(cols) if cols is not None else None
        return (
            pd.DataFrame(
                {
                    "densegen__compression_ratio": [1.0],
                    "densegen__plan": ["demo_plan"],
                }
            ),
            "parquet:outputs/tables/records.parquet",
        )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        _fake_load_records_from_config,
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_attempts",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"status": ["ok"], "reason": ["ok"], "plan_name": ["demo_plan"], "run_order": [1]}
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "solution_id": ["sol-1"],
                "input_name": ["demo_input"],
                "plan_name": ["demo_plan"],
                "tf": ["TF_A"],
                "tfbs": ["AAAA"],
                "offset": [1],
                "length": [4],
            }
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"plan": [{"name": "demo_plan", "quota": 1}]}}},
    )

    def _fake_run_health(df: pd.DataFrame, out_path: Path, **_kwargs) -> list[Path]:
        target = out_path.parent / "run_health" / "run_health.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        assert "densegen__compression_ratio" in df.columns
        assert "densegen__plan" in df.columns
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["run_health"], "fn", _fake_run_health)
    run_plots_from_config(loaded.root, cfg_path, only="run_health")
    assert captured["columns"] is not None
    assert "densegen__compression_ratio" in (captured["columns"] or [])
    assert "densegen__plan" in (captured["columns"] or [])


def test_placement_map_uses_selected_output_source_for_solutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  root: outputs/usr_datasets
                  dataset: densegen/demo
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: usr
              out_dir: outputs/plots
              format: png
              default: ["placement_map"]
            """
        ).strip()
        + "\n"
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\nTF_A,AAAA\n")
    loaded = load_config(cfg_path)

    dense_arrays_df = pd.DataFrame(
        {
            "id": ["sol-1"],
            "sequence": ["ACGTACGTAA"],
            "densegen__input_name": ["demo_input"],
            "densegen__plan": ["demo_plan"],
        }
    )
    composition_df = pd.DataFrame(
        {
            "solution_id": ["sol-1"],
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
            "offset": [1],
            "length": [4],
        }
    )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:densegen/demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: composition_df.copy(),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {"densegen": {"generation": {"sequence_length": 10}}},
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_dense_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("placement_map must use selected output source rows, not records.parquet")
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_libraries",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"library_index": [1], "library_hash": ["hash1"]}),
            pd.DataFrame(
                {
                    "input_name": ["demo_input"],
                    "plan_name": ["demo_plan"],
                    "library_index": [1],
                    "library_hash": ["hash1"],
                    "tf": ["TF_A"],
                    "tfbs": ["AAAA"],
                }
            ),
        ),
    )

    def _fake_placement_map(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        cfg: dict,
        **_kwargs,
    ) -> list[Path]:
        assert list(dense_arrays_df["id"]) == ["sol-1"]
        assert list(composition_df["solution_id"]) == ["sol-1"]
        assert int(cfg["densegen"]["generation"]["sequence_length"]) == 10
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "occupancy.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["placement_map"], "fn", _fake_placement_map)

    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    payload = json.loads(manifest_path.read_text())
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "stage_b/demo_plan/demo_input/occupancy.png" in paths


def test_tfbs_usage_does_not_eager_load_stage_a_pools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["tfbs_usage"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "tf": ["TF_A"], "tfbs": ["AAAA"]}
    ).to_parquet(tables_dir / "composition.parquet", index=False)
    libs_dir = run_root / "outputs" / "libraries"
    libs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "library_index": [1], "library_hash": ["h1"]}
    ).to_parquet(libs_dir / "library_builds.parquet", index=False)
    pd.DataFrame(
        {
            "input_name": ["demo_input"],
            "plan_name": ["demo_plan"],
            "library_index": [1],
            "library_hash": ["h1"],
            "tf": ["TF_A"],
            "tfbs": ["AAAA"],
        }
    ).to_parquet(libs_dir / "library_members.parquet", index=False)

    loaded = load_config(cfg_path)
    expected_error = AssertionError("stage-a pools should not be loaded for tfbs_usage")
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._maybe_load_stage_a_pools",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(expected_error),
    )

    def _fake_tfbs_usage(
        _df: pd.DataFrame, out_path: Path, *, pools=None, composition_df=None, **_kwargs
    ) -> list[Path]:
        assert pools is None
        assert isinstance(composition_df, pd.DataFrame)
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "tfbs_usage.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["tfbs_usage"], "fn", _fake_tfbs_usage)
    run_plots_from_config(loaded.root, cfg_path, only="tfbs_usage")


def test_stage_a_summary_reads_projected_pool_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_summary"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    original = plotting_module.pd.read_parquet
    captured: dict[str, list[str] | None] = {"columns": None}

    def _spy_read_parquet(path: Path, *args, **kwargs):
        if Path(path).name.endswith("__pool.parquet"):
            cols = kwargs.get("columns")
            captured["columns"] = list(cols) if cols is not None else None
        return original(path, *args, **kwargs)

    monkeypatch.setattr(plotting_module.pd, "read_parquet", _spy_read_parquet)
    run_plots_from_config(loaded.root, cfg_path, only="stage_a_summary")
    assert captured["columns"] is not None


def test_stage_b_plot_options_reject_unknown_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["placement_map"],
        plots_options={"placement_map": {"scope": "auto", "max_plans": 2, "unknown_key": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    loaded = load_config(cfg_path)

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "id": ["sol-1"],
                    "sequence": ["ACGTACGTAA"],
                    "densegen__input_name": ["demo_input"],
                    "densegen__plan": ["demo_plan"],
                }
            ),
            "parquet:records",
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "solution_id": ["sol-1"],
                "input_name": ["demo_input"],
                "plan_name": ["demo_plan"],
                "tf": ["TF_A"],
                "tfbs": ["AAAA"],
                "offset": [1],
                "length": [4],
            }
        ),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {
            "densegen": {
                "generation": {
                    "sequence_length": 10,
                    "plan": [{"name": "demo_plan", "quota": 1}],
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Unknown options for plot 'placement_map'"):
        run_plots_from_config(loaded.root, cfg_path, only="placement_map")


def test_tfbs_usage_plot_options_reject_unknown_keys(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["tfbs_usage"],
        plots_options={"tfbs_usage": {"scope": "auto", "max_plans": 2, "unknown_key": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "tf": ["TF_A"], "tfbs": ["AAAA"]}
    ).to_parquet(tables_dir / "composition.parquet", index=False)

    loaded = load_config(cfg_path)
    with pytest.raises(ValueError, match="Unknown options for plot 'tfbs_usage'"):
        run_plots_from_config(loaded.root, cfg_path, only="tfbs_usage")


def test_stage_b_outputs_are_cleaned_before_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["tfbs_usage"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = run_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"input_name": ["demo_input"], "plan_name": ["demo_plan"], "tf": ["TF_A"], "tfbs": ["AAAA"]}
    ).to_parquet(tables_dir / "composition.parquet", index=False)

    stale = run_root / "outputs" / "plots" / "stage_b" / "stale_plan" / "stale_input" / "stale.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")
    assert stale.exists()

    loaded = load_config(cfg_path)

    def _fake_tfbs_usage(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        **_kwargs,
    ) -> list[Path]:
        assert not composition_df.empty
        target = out_path.parent / "stage_b" / "demo_plan" / "demo_input" / "tfbs_usage.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["tfbs_usage"], "fn", _fake_tfbs_usage)

    run_plots_from_config(loaded.root, cfg_path, only="tfbs_usage")

    assert not stale.exists()
    assert (run_root / "outputs" / "plots" / "stage_b" / "demo_plan" / "demo_input" / "tfbs_usage.png").exists()


def test_stage_b_scope_auto_groups_and_drills_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(
        cfg_path,
        plots_default=["placement_map"],
        plots_options={"placement_map": {"scope": "auto", "max_plans": 2, "drilldown_plans": 1}},
    )
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    loaded = load_config(cfg_path)

    dense_arrays_df = pd.DataFrame(
        {
            "id": [f"sol-{idx}" for idx in range(1, 7)],
            "sequence": ["ACGTACGTAA"] * 6,
            "densegen__input_name": [
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_b__sig10_B",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
            ],
            "densegen__plan": [
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=b__sig10=B",
                "sigma70_topup__sig35=f__sig10=H",
                "sigma70_topup__sig35=f__sig10=H",
            ],
        }
    )
    composition_df = pd.DataFrame(
        {
            "solution_id": [f"sol-{idx}" for idx in range(1, 7)],
            "input_name": [
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_a__sig10_A",
                "plan_pool__sigma70_panel__sig35_b__sig10_B",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
                "plan_pool__sigma70_topup__sig35_f__sig10_H",
            ],
            "plan_name": [
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=a__sig10=A",
                "sigma70_panel__sig35=b__sig10=B",
                "sigma70_topup__sig35=f__sig10=H",
                "sigma70_topup__sig35=f__sig10=H",
            ],
            "tf": ["TF_A"] * 6,
            "tfbs": ["AAAA"] * 6,
            "offset": [1, 1, 1, 1, 1, 1],
            "length": [4, 4, 4, 4, 4, 4],
        }
    )

    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting.load_records_from_config",
        lambda *_args, **_kwargs: (dense_arrays_df.copy(), "usr:demo"),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_composition",
        lambda *_args, **_kwargs: composition_df.copy(),
    )
    monkeypatch.setattr(
        "dnadesign.densegen.src.viz.plotting._load_effective_config",
        lambda *_args, **_kwargs: {
            "densegen": {
                "generation": {
                    "sequence_length": 10,
                    "plan": [{"name": "sigma70_panel", "quota": 1}, {"name": "sigma70_topup", "quota": 1}],
                }
            }
        },
    )

    seen_plan_sets: list[tuple[str, ...]] = []
    seen_input_sets: list[tuple[str, ...]] = []

    def _fake_placement_map(
        _df: pd.DataFrame,
        out_path: Path,
        *,
        composition_df: pd.DataFrame,
        dense_arrays_df: pd.DataFrame,
        **_kwargs,
    ) -> list[Path]:
        dense_plans = tuple(sorted(set(dense_arrays_df["densegen__plan"].astype(str))))
        comp_plans = tuple(sorted(set(composition_df["plan_name"].astype(str))))
        dense_inputs = tuple(sorted(set(dense_arrays_df["densegen__input_name"].astype(str))))
        comp_inputs = tuple(sorted(set(composition_df["input_name"].astype(str))))
        assert dense_plans == comp_plans
        assert dense_inputs == comp_inputs
        seen_plan_sets.append(dense_plans)
        seen_input_sets.append(dense_inputs)
        target = out_path.parent / "stage_b" / dense_plans[0] / dense_inputs[0] / f"occupancy_{len(seen_plan_sets)}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plot")
        return [target]

    monkeypatch.setitem(plotting_module.AVAILABLE_PLOTS["placement_map"], "fn", _fake_placement_map)

    run_plots_from_config(loaded.root, cfg_path, only="placement_map")

    assert len(seen_plan_sets) == 2
    assert seen_plan_sets[0] == ("sigma70_panel", "sigma70_topup")
    assert seen_plan_sets[1] == ("sigma70_panel__sig35=a__sig10=A",)
    assert seen_input_sets[0] == ("plan_pool__sigma70_panel", "plan_pool__sigma70_topup")
    assert seen_input_sets[1] == ("plan_pool__sigma70_panel__sig35_a__sig10_A",)
