# ABOUTME: Exercises dashboard utilities for parsing OPAL label history and plots.
# ABOUTME: Covers parsing, filters, and SFXI overlays for dashboard workflows.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_dashboard_utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import polars as pl
import pytest

from dnadesign.opal.src.analysis.dashboard import (
    datasets,
    diagnostics,
    filters,
    hues,
    labels,
    selection,
    transient,
    ui,
    util,
)
from dnadesign.opal.src.analysis.dashboard.charts import plots
from dnadesign.opal.src.analysis.dashboard.views import sfxi
from dnadesign.opal.src.analysis.sfxi.state_order import STATE_ORDER


def test_find_repo_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested_dir = root / "a" / "b"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "file.py"
    nested_file.write_text("print('hi')\n")

    found = datasets.find_repo_root(nested_file)
    assert found == root


def test_resolve_usr_root_override(tmp_path: Path) -> None:
    override = tmp_path / "usr_root"
    override.mkdir()
    resolved = datasets.resolve_usr_root(None, str(override))
    assert resolved == override

    with pytest.raises(ValueError):
        datasets.resolve_usr_root(None, str(tmp_path / "missing"))


def test_resolve_dataset_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    usr_root = repo_root / "src" / "dnadesign" / "usr" / "datasets"
    usr_root.mkdir(parents=True)
    dataset = usr_root / "demo"
    dataset.mkdir()
    (dataset / "records.parquet").write_text("dummy")

    path, mode = datasets.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name="demo",
        custom_path=None,
    )
    assert mode == "usr"
    assert path == dataset / "records.parquet"

    custom_rel, mode = datasets.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name=None,
        custom_path="data/custom.parquet",
    )
    assert mode == "custom"
    assert custom_rel == repo_root / "data" / "custom.parquet"

    custom_abs = tmp_path / "abs.parquet"
    custom_path, mode = datasets.resolve_dataset_path(
        repo_root=repo_root,
        usr_root=usr_root,
        dataset_name=None,
        custom_path=str(custom_abs),
    )
    assert mode == "custom"
    assert custom_path == custom_abs


def test_parse_campaign_info_dashboard_defaults(tmp_path: Path) -> None:
    raw = {
        "campaign": {"name": "demo", "slug": "demo", "workdir": "."},
        "data": {"x_column_name": "x", "y_column_name": "y", "y_expected_length": 8},
        "model": {"name": "random_forest", "params": {}},
        "objective": {"name": "sfxi_v1", "params": {}},
        "selection": {"name": "top_n", "params": {}},
        "training": {"policy": {}, "y_ops": []},
        "metadata": {"dashboard": {"explorer_defaults": {"x": "opal__view__score", "y": "opal__view__logic_fidelity"}}},
    }
    info = datasets.parse_campaign_info(raw=raw, path=tmp_path / "campaign.yaml", label="demo")
    assert info.dashboard is not None
    assert info.dashboard["explorer_defaults"]["x"] == "opal__view__score"


def test_namespace_summary() -> None:
    cols = ["id", "densegen__plan", "opal__x__label_hist", "cluster__ldn_v1"]
    summary = util.namespace_summary(cols, max_examples=2)
    summary_dict = {row["namespace"]: row for row in summary.to_dicts()}
    assert summary_dict["core"]["count"] == 1
    assert summary_dict["densegen"]["count"] == 1
    assert summary_dict["opal"]["count"] == 1
    assert summary_dict["cluster"]["count"] == 1


def test_default_view_hues_include_sfxi_diagnostics() -> None:
    keys = {option.key for option in hues.default_view_hues()}
    assert "opal__sfxi__nearest_gate_class" in keys
    assert "opal__sfxi__nearest_gate_dist" in keys
    assert "opal__sfxi__dist_to_labeled_logic" in keys
    assert "opal__sfxi__dist_to_labeled_x" in keys
    assert "opal__sfxi__uncertainty" in keys


def test_umap_explorer_chart_overlays_observed_labels() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "cluster__ldn_v1__umap_x": [0.0, 1.0],
            "cluster__ldn_v1__umap_y": [0.0, 1.0],
            "opal__view__observed": [True, False],
        }
    )
    result = plots.build_umap_explorer_chart(
        df=df,
        x_col="cluster__ldn_v1__umap_x",
        y_col="cluster__ldn_v1__umap_y",
        hue=None,
        point_size=20,
        opacity=0.5,
        plot_size=200,
        dataset_name="demo",
    )
    assert isinstance(result.chart, alt.LayerChart)


def test_choose_dropdown_value_prefers_current_then_preferred() -> None:
    options = ["a", "b"]
    assert util.choose_dropdown_value(options, current="b", preferred="a") == "b"
    assert util.choose_dropdown_value(options, current="c", preferred="a") == "a"
    assert util.choose_dropdown_value(options, current=None, preferred=None) == "a"
    assert util.choose_dropdown_value([], current="a", preferred="b") is None


def test_choose_axis_defaults_skips_row_id() -> None:
    numeric_cols = ["__row_id", "opal__view__score", "opal__view__effect_scaled"]
    x_default, y_default = util.choose_axis_defaults(
        numeric_cols=numeric_cols,
        preferred_x="opal__view__score",
        preferred_y="opal__view__effect_scaled",
    )
    assert x_default == "opal__view__score"
    assert y_default == "opal__view__effect_scaled"


def test_state_value_changed() -> None:
    assert util.state_value_changed("score", "score") is False
    assert util.state_value_changed("score", "logic_fidelity") is True
    assert util.state_value_changed(None, None) is False
    assert util.state_value_changed(None, "score") is True


def test_attach_namespace_columns_prefers_row_id() -> None:
    df_base = pl.DataFrame(
        {
            "__row_id": [0, 1],
            "id": ["a", "b"],
            "cluster__ldn_v1": ["0", "1"],
            "densegen__score": [0.1, 0.2],
            "other": ["x", "y"],
        }
    )
    df_target = pl.DataFrame(
        {
            "__row_id": [0, 1],
            "id": ["x", "y"],
            "logic_fidelity": [0.5, 0.6],
        }
    )
    df_out = util.attach_namespace_columns(
        df=df_target,
        df_base=df_base,
        prefixes=("cluster__", "densegen__"),
    )
    assert "cluster__ldn_v1" in df_out.columns
    assert "densegen__score" in df_out.columns
    assert "other" not in df_out.columns
    assert df_out["cluster__ldn_v1"].to_list() == ["0", "1"]
    assert df_out["densegen__score"].to_list() == [0.1, 0.2]


def test_hue_registry_filters_invalid_columns() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "valid_num": [0.1, 0.2],
            "valid_cat": ["x", "y"],
            "all_null": [None, None],
            "nested": [[1, 2], [3, 4]],
            "opal__cache__score": [0.1, 0.2],
            "opal__ledger__score": [0.2, 0.3],
        }
    )
    registry = hues.build_hue_registry(df, include_columns=True, denylist={"id"})
    labels = registry.labels()
    assert "valid_num" in labels
    assert "valid_cat" in labels
    assert "all_null" not in labels
    assert "nested" not in labels
    assert "opal__cache__score" not in labels
    assert "opal__ledger__score" not in labels


def test_explorer_hue_registry_allows_high_cardinality() -> None:
    values = [str(i) for i in range(67)]
    df = pl.DataFrame(
        {
            "id": [f"id_{i}" for i in range(len(values))],
            "cluster__ldn_v1": values,
            "opal__view__score": list(range(len(values))),
        }
    )
    default_labels = hues.build_hue_registry(df, include_columns=True).labels()
    assert "cluster__ldn_v1" not in default_labels

    explorer_labels = hues.build_explorer_hue_registry(df, include_columns=True).labels()
    assert "cluster__ldn_v1" in explorer_labels


def test_sfxi_hue_registry_allows_high_cardinality() -> None:
    values = [str(i) for i in range(67)]
    df = pl.DataFrame(
        {
            "id": [f"id_{i}" for i in range(len(values))],
            "cluster__ldn_v1": values,
            "score": list(range(len(values))),
        }
    )
    sfxi_labels = hues.build_sfxi_hue_registry(df, include_columns=True).labels()
    assert "cluster__ldn_v1" in sfxi_labels


def test_ensure_selection_columns() -> None:
    df = pl.DataFrame({"id": ["a", "b", "c"], "score": [0.3, 0.1, 0.2]})
    df_out, warnings, err = selection.ensure_selection_columns(
        df,
        id_col="id",
        score_col="score",
        selection_params={"top_k": 1, "objective_mode": "maximize"},
        rank_col="rank",
        top_k_col="top_k",
    )
    assert err is None
    assert "rank" in df_out.columns
    assert "top_k" in df_out.columns
    assert len(df_out["rank"]) == 3
    assert warnings == []


def test_build_umap_overlay_charts() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "cluster__ldn_v1__umap_x": [0.1, 0.2],
            "cluster__ldn_v1__umap_y": [0.3, 0.4],
            "cluster__ldn_v1": ["0", "1"],
            "opal__view__score": [0.2, 0.4],
        }
    )
    cluster_chart, score_chart = plots.build_umap_overlay_charts(
        df=df,
        dataset_name="demo",
    )
    assert isinstance(cluster_chart, alt.Chart)
    assert isinstance(score_chart, alt.Chart)


def test_build_umap_explorer_chart_cases() -> None:
    base = pl.DataFrame(
        {
            "id": ["a", "b"],
            "x": [0.1, 0.2],
            "y": [0.3, 0.4],
            "opal__overlay__observed_event": [True, False],
            "opal__score__top_k": [True, False],
        }
    )
    hue_observed = hues.HueOption(
        key="opal__overlay__observed_event",
        label="Observed",
        kind="categorical",
        dtype=pl.Boolean,
        category_labels=("Observed", "Unlabeled"),
    )
    ok = plots.build_umap_explorer_chart(
        df=base,
        x_col="x",
        y_col="y",
        hue=hue_observed,
        point_size=40,
        opacity=0.7,
        plot_size=420,
        dataset_name="demo",
    )
    assert ok.valid is True
    assert ok.note is not None and "Plotting full dataset" in ok.note
    assert isinstance(ok.chart, alt.Chart)

    missing_id = plots.build_umap_explorer_chart(
        df=base.drop("id"),
        x_col="x",
        y_col="y",
        hue=None,
        point_size=40,
        opacity=0.7,
        plot_size=420,
        dataset_name="demo",
    )
    assert missing_id.valid is False
    assert missing_id.note is not None and "required column `id`" in missing_id.note

    missing_xy = plots.build_umap_explorer_chart(
        df=base,
        x_col="",
        y_col="",
        hue=None,
        point_size=40,
        opacity=0.7,
        plot_size=420,
        dataset_name="demo",
    )
    assert missing_xy.valid is False
    assert missing_xy.note is not None and "provide x/y columns" in missing_xy.note

    non_numeric = plots.build_umap_explorer_chart(
        df=pl.DataFrame({"id": ["a"], "x": ["na"], "y": ["nb"]}),
        x_col="x",
        y_col="y",
        hue=None,
        point_size=40,
        opacity=0.7,
        plot_size=420,
        dataset_name="demo",
    )
    assert non_numeric.valid is False
    assert non_numeric.note is not None and "must be numeric" in non_numeric.note

    no_non_null = plots.build_umap_explorer_chart(
        df=base.with_columns(pl.lit(None).alias("all_null")),
        x_col="x",
        y_col="y",
        hue=hues.HueOption(
            key="all_null",
            label="All null",
            kind="numeric",
            dtype=pl.Float64,
        ),
        point_size=40,
        opacity=0.7,
        plot_size=420,
        dataset_name="demo",
    )
    assert no_non_null.valid is True
    assert no_non_null.note is not None and "no non-null values" in no_non_null.note


def test_score_histogram_lollipop_includes_id() -> None:
    df_pred_scored = pl.DataFrame(
        {
            "id": ["a", "b"],
            "pred_score": [0.1, 0.4],
        }
    )
    df_sfxi = pl.DataFrame({"id": ["obs-1"], "score": [0.2]})
    df_train = pl.DataFrame({"id": ["train-1"]})
    chart, note = transient.build_score_histogram(
        df_pred_scored=df_pred_scored,
        score_col="pred_score",
        df_sfxi=df_sfxi,
        df_train=df_train,
        dataset_name="demo",
    )
    assert note is None
    assert chart is not None
    spec = json.dumps(chart.to_dict())
    assert '"field": "id"' in spec


def test_build_umap_controls_uses_raw_column_names() -> None:
    class _DummyDropdown:
        def __init__(self, *, options, value=None, label=None, full_width=False):
            self.options = options
            self.value = value
            self.label = label
            self.full_width = full_width

    class _DummyText:
        def __init__(self, *, value=None, label=None, full_width=False):
            self.value = value
            self.label = label
            self.full_width = full_width

    class _DummyUI:
        def dropdown(self, *, options, value=None, label=None, full_width=False):
            return _DummyDropdown(options=options, value=value, label=label, full_width=full_width)

        def text(self, *, value=None, label=None, full_width=False):
            return _DummyText(value=value, label=label, full_width=full_width)

    class _DummyMo:
        ui = _DummyUI()

    df = pl.DataFrame(
        {
            "cluster__ldn_v1__umap_x": [0.1],
            "cluster__ldn_v1__umap_y": [0.2],
            "opal__view__score": [0.3],
        }
    )
    opt = hues.HueOption(key="opal__view__score", label="Score", kind="numeric", dtype=pl.Float64)
    registry = hues.HueRegistry(options=[opt], label_map={"Score": opt})
    controls = ui.build_umap_controls(
        mo=_DummyMo(),
        df_active=df,
        hue_registry=registry,
        default_hue_key="opal__view__score",
    )
    assert controls.umap_color_dropdown.options == ["(none)", "opal__view__score"]
    assert controls.umap_color_dropdown.value == "opal__view__score"


def test_apply_overlay_label_flags() -> None:
    df_overlay = pl.DataFrame({"id": ["a", "b"], "__row_id": [1, 2]})
    labels_view = pl.DataFrame({"id": ["a"], "label_src": ["ingest_y"]})
    df_sfxi = pl.DataFrame({"__row_id": [2]})
    out = sfxi.apply_overlay_label_flags(
        df_overlay=df_overlay,
        labels_view_df=labels_view,
        df_sfxi=df_sfxi,
        label_src="ingest_y",
        id_col="id",
    )
    assert out["opal__overlay__observed_event"].to_list() == [True, False]
    assert out["opal__overlay__sfxi_scored_label"].to_list() == [False, True]


def test_diagnostics_notes_merge() -> None:
    diag = diagnostics.Diagnostics().add_note("n1").add_warning("w1").add_error("e1")
    other = diagnostics.Diagnostics().add_note("n2").add_warning("w2")
    merged = diag.merge(other)
    assert merged.notes == ["n1", "n2"]
    assert merged.warnings == ["w1", "w2"]
    assert merged.errors == ["e1"]


def test_diagnostics_to_lines() -> None:
    diag = diagnostics.Diagnostics().add_note("note").add_warning("warn").add_error("err")
    lines = diagnostics.diagnostics_to_lines(diag)
    assert lines == ["note", "Warning: warn", "Error: err"]
    assert diagnostics.diagnostics_to_lines(None) == []


def test_opal_labeled_mask() -> None:
    df = pl.DataFrame(
        {
            "opal__a__label_hist": [
                [],
                None,
                [{"observed_round": 1, "y_obs": {"value": [0.1], "dtype": "vector"}}],
            ],
            "opal__b__label_hist": [None, [], []],
        }
    )
    mask = labels.opal_labeled_mask(df, ["opal__a__label_hist", "opal__b__label_hist"])
    assert mask.to_list() == [False, False, True]


def test_numeric_rule_builder() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": [5.0, 6.0, 7.0]})
    rules = [
        filters.NumericRule(enabled=True, column="a", op=">=", value=2.0),
        filters.NumericRule(enabled=True, column="b", op="<=", value=6.0),
    ]
    filtered = filters.apply_numeric_rules(df, rules)
    assert filtered.height == 1
    assert filtered["a"].to_list() == [2.0]

    rules_null = [filters.NumericRule(enabled=True, column="a", op="is null")]
    filtered_null = filters.apply_numeric_rules(df, rules_null)
    assert filtered_null.height == 1
    assert filtered_null["a"].to_list() == [None]


def test_compute_sfxi_params_requires_state_order() -> None:
    with pytest.raises(ValueError, match="state_order"):
        sfxi.compute_sfxi_params(
            setpoint=[0.25, 0.25, 0.25, 0.25],
            beta=1.0,
            gamma=1.0,
            delta=0.0,
            p=95.0,
            min_n=1,
            eps=1e-6,
            state_order=None,
        )


def test_compute_sfxi_params_rejects_invalid_setpoint() -> None:
    with pytest.raises(ValueError, match="setpoint_vector"):
        sfxi.compute_sfxi_params(
            setpoint=[-0.1, 0.25, 0.25, 1.1],
            beta=1.0,
            gamma=1.0,
            delta=0.0,
            p=95.0,
            min_n=1,
            eps=1e-6,
            state_order=STATE_ORDER,
        )


def test_sfxi_metrics_deterministic() -> None:
    df = pl.DataFrame({"sfxi_8_vector_y_label": [[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]]})
    params = sfxi.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        min_n=1,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    result = sfxi.compute_sfxi_metrics(
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


def test_compute_label_sfxi_view_preview() -> None:
    df = pl.DataFrame({"sfxi_8_vector_y_label": [[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]]})
    params = sfxi.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        min_n=1,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    view = sfxi.compute_label_sfxi_view(
        labels_view_df=df,
        labels_current_df=df,
        y_col="sfxi_8_vector_y_label",
        params=params,
    )
    assert view.notice is None
    assert not view.df.is_empty()
    assert any(col.endswith("_preview") for col in view.table_cols)


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
    params = sfxi.compute_sfxi_params(
        setpoint=[0.0, 0.0, 0.0, 0.0],
        beta=1.0,
        gamma=1.0,
        delta=10.0,
        p=95.0,
        min_n=2,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    empty_pool = pl.DataFrame({"sfxi_8_vector_y_label": []})
    result = sfxi.compute_sfxi_metrics(
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
    result_fallback = sfxi.compute_sfxi_metrics(
        df=pool,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=pool,
    )
    assert result_fallback.denom == 1.0
    assert result_fallback.denom_source == "disabled"

    params_strict = sfxi.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=10.0,
        p=95.0,
        min_n=2,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    with pytest.raises(ValueError, match="min_n"):
        sfxi.compute_sfxi_metrics(
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
            "opal__demo__label_hist": [
                [],
                [{"observed_round": 1, "y_obs": {"value": [0.1], "dtype": "vector"}}],
                [],
            ],
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
    rules = [filters.NumericRule(enabled=True, column="cluster__ldn_v1__umap_x", op=">=", value=0.2)]
    filtered = filters.apply_numeric_rules(loaded, rules)
    assert filtered.height == 2

    missing = util.missingness_summary(filtered)
    assert "null_pct" in missing.columns

    params = sfxi.compute_sfxi_params(
        setpoint=[0.5, 0.5, 0.5, 0.5],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        min_n=1,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    sfxi_result = sfxi.compute_sfxi_metrics(
        df=loaded,
        vec_col="sfxi_8_vector_y_label",
        params=params,
        denom_pool_df=loaded,
    )
    assert "score" in sfxi_result.df.columns

    chart = alt.Chart(loaded).mark_point().encode(x="cluster__ldn_v1__umap_x", y="cluster__ldn_v1__umap_y")
    assert isinstance(chart, alt.Chart)


def test_dedupe_helpers() -> None:
    assert util.dedupe_columns(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    exprs = util.dedupe_exprs([pl.col("a"), pl.col("a"), pl.col("b")])
    names = [expr.meta.output_name() for expr in exprs]
    assert names == ["a", "b"]


def test_selection_coercion() -> None:
    class UndefinedType:
        pass

    assert util.is_altair_undefined(UndefinedType())
    assert selection.coerce_selection_dataframe(None) is None
    df = pl.DataFrame({"a": [1]})
    assert selection.coerce_selection_dataframe(df) is df


def test_list_series_to_numpy() -> None:
    series = pl.Series("x", [[1.0, 2.0], [3.0, 4.0]])
    arr = util.list_series_to_numpy(series, expected_len=2)
    assert arr is not None
    assert arr.shape == (2, 2)
    assert util.list_series_to_numpy(series, expected_len=3) is None


def test_resolve_objective_mode_aliases() -> None:
    mode, warnings = selection.resolve_objective_mode({"objective_mode": "minimize"})
    assert mode == "minimize"
    assert warnings == []

    mode, warnings = selection.resolve_objective_mode({"objective": "minimize"})
    assert mode == "minimize"
    assert warnings

    mode, warnings = selection.resolve_objective_mode({"objective": "unknown"})
    assert mode == "maximize"
    assert warnings

    with pytest.raises(ValueError):
        selection.resolve_objective_mode({"objective_mode": "maximize", "objective": "minimize"})


def test_build_label_events_parsing_variants() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "x_vec": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "opal__demo__label_hist": pl.Series(
                "opal__demo__label_hist",
                [
                    [
                        {
                            "observed_round": 1,
                            "y_obs": {"value": [0.1, 0.2], "dtype": "vector"},
                            "src": "ingest_y",
                        }
                    ],
                    ['{"observed_round": 2, "y_obs": {"value": [0.3, 0.4], "dtype": "vector"}, "src": "manual"}'],
                    "not json",
                ],
                dtype=pl.Object,
            ),
        }
    )
    label_events = labels.build_label_events(
        df=df,
        label_hist_col="opal__demo__label_hist",
        y_col_name="y_obs",
    )
    assert "x_vec" in label_events.df.columns
    assert label_events.df.height == 2
    assert label_events.diag.status in {"parse_warning", "ok"}


def test_build_pred_events_parsing_wrappers() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "opal__demo__label_hist": pl.Series(
                "opal__demo__label_hist",
                [
                    [
                        {
                            "kind": "pred",
                            "as_of_round": 1,
                            "run_id": "run-1",
                            "y_pred": {"value": [0.2, 0.3], "dtype": "vector"},
                            "y_space": "objective",
                            "objective": {"name": "sfxi_v1", "params": {"setpoint_vector": [0, 0, 0, 1]}},
                            "metrics": {"score": 0.5},
                            "selection": {"rank": 1, "top_k": True},
                        }
                    ],
                    [
                        {
                            "kind": "pred",
                            "as_of_round": 1,
                            "run_id": "run-2",
                            "y_pred": {"value": {"note": "opaque"}, "dtype": "object"},
                            "y_space": "objective",
                            "objective": {"name": "sfxi_v1", "params": {"setpoint_vector": [0, 0, 0, 1]}},
                            "metrics": {"score": 0.2},
                            "selection": {"rank": 2, "top_k": False},
                        }
                    ],
                ],
                dtype=pl.Object,
            ),
        }
    )
    pred_events = labels.build_pred_events(df=df, label_hist_col="opal__demo__label_hist")
    assert pred_events.df.height == 2
    assert pred_events.df["pred_y_hat"].to_list()[0] == [0.2, 0.3]
    assert pred_events.df["pred_y_hat"].to_list()[1] is None


def test_observed_event_ids() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label_src": ["ingest_y", "manual", "ingest_y"],
        }
    )
    observed = labels.observed_event_ids(df, label_src="ingest_y")
    assert sorted(observed) == ["a", "c"]


def test_infer_round_from_labels() -> None:
    df = pl.DataFrame({"observed_round": [1, 3, 2, None]})
    assert labels.infer_round_from_labels(df) == 3
    assert labels.infer_round_from_labels(pl.DataFrame({"x": [1]})) is None


def test_overlay_provenance_on_early_exit() -> None:
    df_base = pl.DataFrame({"id": ["a"], "__row_id": [0], "x_vec": [[1.0, 2.0]]})
    params = sfxi.compute_sfxi_params(
        setpoint=[0.25, 0.25, 0.25, 0.25],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=95.0,
        min_n=1,
        eps=1e-6,
        state_order=STATE_ORDER,
    )
    result = transient.compute_transient_overlay(
        df_base=df_base,
        pred_df=pl.DataFrame(),
        labels_current_df=pl.DataFrame(),
        df_sfxi=pl.DataFrame(),
        y_col="y_vec",
        sfxi_params=params,
        selection_params={},
        dataset_name="demo",
        as_of_round=0,
        run_id="run-1",
    )
    assert "opal__overlay__round" in result.df_overlay.columns
    assert result.df_overlay["opal__overlay__round"].to_list() == [0]
    assert result.df_overlay["opal__overlay__run_id"].to_list() == ["run-1"]
    assert result.diagnostics.errors


def test_build_pred_sfxi_view_canonical() -> None:
    params = sfxi.compute_sfxi_params(
        setpoint=[0.0, 0.0, 0.0, 1.0],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=50.0,
        min_n=1,
        eps=1.0e-8,
        state_order=STATE_ORDER,
    )
    pred_df = pl.DataFrame(
        {
            "id": ["a"],
            "pred_score": [0.42],
            "pred_logic_fidelity": [0.7],
            "pred_effect_scaled": [0.6],
        }
    )
    view = sfxi.build_pred_sfxi_view(
        pred_df=pred_df,
        labels_current_df=pl.DataFrame(),
        y_col="y_obs",
        params=params,
        mode="Canonical",
    )
    assert view.notice is None
    assert view.df.select(pl.col("score")).item() == pytest.approx(0.42)
    assert view.df.select(pl.col("logic_fidelity")).item() == pytest.approx(0.7)
    assert view.df.select(pl.col("effect_scaled")).item() == pytest.approx(0.6)


def test_build_pred_sfxi_view_overlay() -> None:
    params = sfxi.compute_sfxi_params(
        setpoint=[0.0, 0.0, 0.0, 1.0],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=50.0,
        min_n=1,
        eps=1.0e-8,
        state_order=STATE_ORDER,
    )
    pred_df = pl.DataFrame(
        {
            "id": ["a"],
            "pred_y_hat": [[0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.2, 0.2]],
        }
    )
    labels_current_df = pl.DataFrame(
        {
            "y_obs": [[0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.2, 0.2]],
        }
    )
    view = sfxi.build_pred_sfxi_view(
        pred_df=pred_df,
        labels_current_df=labels_current_df,
        y_col="y_obs",
        params=params,
        mode="Overlay",
    )
    assert view.notice is None
    assert not view.df.is_empty()


def test_build_pred_sfxi_view_overlay_object_vectors() -> None:
    params = sfxi.compute_sfxi_params(
        setpoint=[0.0, 0.0, 0.0, 1.0],
        beta=1.0,
        gamma=1.0,
        delta=0.0,
        p=50.0,
        min_n=1,
        eps=1.0e-8,
        state_order=STATE_ORDER,
    )
    vec = [0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.2, 0.2]
    pred_df = pl.DataFrame(
        {
            "id": ["a"],
            "pred_y_hat": pl.Series("pred_y_hat", [vec], dtype=pl.Object),
        }
    )
    labels_current_df = pl.DataFrame(
        {
            "y_obs": pl.Series("y_obs", [vec], dtype=pl.Object),
        }
    )
    view = sfxi.build_pred_sfxi_view(
        pred_df=pred_df,
        labels_current_df=labels_current_df,
        y_col="y_obs",
        params=params,
        mode="Overlay",
    )
    assert view.notice is None
    assert not view.df.is_empty()
