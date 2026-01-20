"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_promoter_eda_utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import pytest

from dnadesign.opal.src.analysis import promoter_eda_utils as utils


def test_find_repo_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested_dir = root / "a" / "b"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "file.py"
    nested_file.write_text("print('hi')\n")

    found = utils.find_repo_root(nested_file)
    assert found == root


def test_resolve_usr_root_override(tmp_path: Path) -> None:
    override = tmp_path / "usr_root"
    override.mkdir()
    resolved = utils.resolve_usr_root(None, str(override))
    assert resolved == override

    with pytest.raises(ValueError):
        utils.resolve_usr_root(None, str(tmp_path / "missing"))


def test_resolve_dataset_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    usr_root = repo_root / "src" / "dnadesign" / "usr" / "datasets"
    usr_root.mkdir(parents=True)
    dataset = usr_root / "demo"
    dataset.mkdir()
    (dataset / "records.parquet").write_text("dummy")

    path, mode = utils.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name="demo",
        custom_path=None,
    )
    assert mode == "usr"
    assert path == dataset / "records.parquet"

    custom_rel, mode = utils.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name=None,
        custom_path="data/custom.parquet",
    )
    assert mode == "custom"
    assert custom_rel == repo_root / "data" / "custom.parquet"

    custom_abs = tmp_path / "abs.parquet"
    custom_path, mode = utils.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name=None,
        custom_path=str(custom_abs),
    )
    assert mode == "custom"
    assert custom_path == custom_abs


def test_namespace_summary() -> None:
    cols = ["id", "densegen__plan", "opal__x__label_hist", "cluster__ldn_v1"]
    summary = utils.namespace_summary(cols, max_examples=2)
    summary_dict = {row["namespace"]: row for row in summary.to_dicts()}
    assert summary_dict["core"]["count"] == 1
    assert summary_dict["densegen"]["count"] == 1
    assert summary_dict["opal"]["count"] == 1
    assert summary_dict["cluster"]["count"] == 1


def test_opal_labeled_mask() -> None:
    df = pl.DataFrame(
        {
            "opal__a__label_hist": [[], None, [{"r": 1}]],
            "opal__b__label_hist": [None, [], []],
        }
    )
    mask = utils.opal_labeled_mask(df, ["opal__a__label_hist", "opal__b__label_hist"])
    assert mask.to_list() == [False, False, True]


def test_numeric_rule_builder() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": [5.0, 6.0, 7.0]})
    rules = [
        utils.NumericRule(enabled=True, column="a", op=">=", value=2.0),
        utils.NumericRule(enabled=True, column="b", op="<=", value=6.0),
    ]
    filtered = utils.apply_numeric_rules(df, rules)
    assert filtered.height == 1
    assert filtered["a"].to_list() == [2.0]

    rules_null = [utils.NumericRule(enabled=True, column="a", op="is null")]
    filtered_null = utils.apply_numeric_rules(df, rules_null)
    assert filtered_null.height == 1
    assert filtered_null["a"].to_list() == [None]


def test_sfxi_metrics_deterministic() -> None:
    df = pl.DataFrame({"sfxi_8_vector_y_label": [[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]]})
    params = utils.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        fallback_p=75.0,
        min_n=1,
        eps=1e-6,
    )
    result = utils.compute_sfxi_metrics(
        df=df,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=df,
    )
    assert result.pool_size == 1
    assert abs(result.denom - 2.0) < 1e-6
    row = result.df.row(0)
    row_dict = dict(zip(result.df.columns, row))
    assert abs(row_dict["logic_fidelity"] - 1.0) < 1e-6
    assert abs(row_dict["effect_raw"] - 2.0) < 1e-6
    assert abs(row_dict["effect_scaled"] - 1.0) < 1e-6
    assert abs(row_dict["score"] - 1.0) < 1e-6


def test_sfxi_metrics_edge_cases() -> None:
    df = pl.DataFrame(
        {
            "sfxi_8_vector_y_label": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                None,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            ]
        }
    )
    params = utils.compute_sfxi_params(
        setpoint=[0.0, 0.0, 0.0, 0.0],
        beta=1.0,
        gamma=1.0,
        delta=10.0,
        p=95.0,
        fallback_p=75.0,
        min_n=2,
        eps=1e-6,
    )
    empty_pool = pl.DataFrame({"sfxi_8_vector_y_label": []})
    result = utils.compute_sfxi_metrics(
        df=df,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=empty_pool,
    )
    assert result.df.height == 1
    assert result.weights == (0.0, 0.0, 0.0, 0.0)
    assert result.denom == 1.0
    assert result.denom_source == "disabled"
    row = result.df.row(0)
    row_dict = dict(zip(result.df.columns, row))
    assert abs(row_dict["effect_scaled"] - 1.0) < 1e-6
    assert abs(row_dict["score"] - 1.0) < 1e-6

    pool = pl.DataFrame({"sfxi_8_vector_y_label": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]})
    result_fallback = utils.compute_sfxi_metrics(
        df=pool,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=pool,
    )
    assert result_fallback.denom == 1.0
    assert result_fallback.denom_source == "disabled"

    params_strict = utils.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=10.0,
        p=95.0,
        fallback_p=75.0,
        min_n=2,
        eps=1e-6,
    )
    with pytest.raises(ValueError, match="min_n"):
        utils.compute_sfxi_metrics(
            df=pool,
            vec_col="sfxi_8_vector_y_label",
            params=params_strict,
            denom_pool_df=pool,
        )


def test_integration_smoke(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "densegen__plan": ["plan1", "plan2", "plan1"],
            "cluster__ldn_v1": ["c1", "c2", "c1"],
            "cluster__ldn_v1__umap_x": [0.1, 0.2, 0.3],
            "cluster__ldn_v1__umap_y": [0.0, 0.5, 0.7],
            "sfxi_8_vector_y_label": [
                [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0],
            ],
            "opal__demo__label_hist": [[], [{"r": 1}], []],
            "densegen__used_tfbs_detail": [
                [{"offset": 1, "orientation": "fwd", "tf": "x", "tfbs": "AAAA"}],
                [],
                [],
            ],
            "cluster__ldn_v1__meta": [
                {"algo": "ldn", "n": 2},
                {"algo": "ldn", "n": 2},
                {"algo": "ldn", "n": 2},
            ],
        }
    )
    path = tmp_path / "records.parquet"
    df.write_parquet(path)

    loaded = pl.read_parquet(path)
    rules = [utils.NumericRule(enabled=True, column="cluster__ldn_v1__umap_x", op=">=", value=0.2)]
    filtered = utils.apply_numeric_rules(loaded, rules)
    assert filtered.height == 2

    missing = utils.missingness_summary(filtered)
    assert "null_pct" in missing.columns

    params = utils.compute_sfxi_params(
        setpoint=[0.5, 0.5, 0.5, 0.5],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        fallback_p=75.0,
        min_n=1,
        eps=1e-6,
    )
    sfxi = utils.compute_sfxi_metrics(
        df=loaded,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=loaded,
    )
    assert "score" in sfxi.df.columns

    chart = alt.Chart(loaded).mark_point().encode(x="cluster__ldn_v1__umap_x", y="cluster__ldn_v1__umap_y")
    assert isinstance(chart, alt.Chart)


def test_dedupe_helpers() -> None:
    assert utils.dedupe_columns(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    exprs = utils.dedupe_exprs([pl.col("a"), pl.col("a"), pl.col("b")])
    names = [expr.meta.output_name() for expr in exprs]
    assert names == ["a", "b"]


def test_selection_coercion() -> None:
    class UndefinedType:
        pass

    assert utils.is_altair_undefined(UndefinedType())
    assert utils.coerce_selection_dataframe(None) is None
    df = pl.DataFrame({"a": [1]})
    assert utils.coerce_selection_dataframe(df) is df


def test_list_series_to_numpy() -> None:
    series = pl.Series("x", [[1.0, 2.0], [3.0, 4.0]])
    arr = utils.list_series_to_numpy(series, expected_len=2)
    assert arr is not None
    assert arr.shape == (2, 2)
    assert utils.list_series_to_numpy(series, expected_len=3) is None


def test_resolve_objective_mode_aliases() -> None:
    mode, warnings = utils.resolve_objective_mode({"objective_mode": "minimize"})
    assert mode == "minimize"
    assert warnings == []

    mode, warnings = utils.resolve_objective_mode({"objective": "minimize"})
    assert mode == "minimize"
    assert warnings

    mode, warnings = utils.resolve_objective_mode({"objective": "unknown"})
    assert mode == "maximize"
    assert warnings

    with pytest.raises(ValueError):
        utils.resolve_objective_mode({"objective_mode": "maximize", "objective": "minimize"})


def test_build_label_events_parsing_variants() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "x_vec": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "opal__demo__label_hist": pl.Series(
                "opal__demo__label_hist",
                [
                    [{"r": 1, "y": [0.1, 0.2], "src": "ingest_y"}],
                    ['{"r": 2, "y": [0.3, 0.4], "src": "manual"}'],
                    "not json",
                ],
                dtype=pl.Object,
            ),
        }
    )
    events, diag = utils.build_label_events(
        df=df,
        label_hist_col="opal__demo__label_hist",
        y_col_name="y_obs",
    )
    assert "x_vec" in events.columns
    assert events.height == 2
    assert diag["status"] in {"parse_warning", "ok"}
