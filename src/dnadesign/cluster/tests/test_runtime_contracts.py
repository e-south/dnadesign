"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_runtime_contracts.py

Runtime path and method-contract tests for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytest
import typer

from dnadesign.cluster import (
    AnalysisRequest,
    AnalysisRun,
    ClusteringMethod,
    ClusterRun,
    EmbeddingRun,
    FeatureSpec,
    FitRequest,
    InputSource,
    MethodConfig,
    MethodRegistry,
    RunCounts,
    WorkspaceConfig,
    builtin_workspaces_dir,
    init_workspace,
    load_workspace_config,
    register_method,
    run_fit,
)
from dnadesign.cluster.contracts import RunIndexEntry
from dnadesign.cluster.src.analysis.numeric_per_cluster import _coerce_numeric
from dnadesign.cluster.src.cli.resolution import resolve_cli_or_config_value
from dnadesign.cluster.src.execution import cluster_overlay_col, intra_sim_overlay_col
from dnadesign.cluster.src.io.read import extract_X
from dnadesign.cluster.src.io.write import _append_usr_event
from dnadesign.cluster.src.layout import builtin_cluster_dir, explicit_results_root
from dnadesign.cluster.src.methods import parse_method_param_assignments
from dnadesign.cluster.src.methods.kmeans import resolve_fit_params as resolve_kmeans_fit_params
from dnadesign.cluster.src.methods.leiden import LEIDEN_FIT_PARAM_NAMES, resolve_fit_params
from dnadesign.cluster.src.methods.registry import get_method
from dnadesign.cluster.src.runs.contracts import FIT_REUSE_REQUIRED_COLUMNS, SweepRun, fit_alias_from_cluster_col
from dnadesign.cluster.src.runs.index import add_or_update_index, compact_index, list_runs
from dnadesign.cluster.src.runs.reuse import find_equivalent_fit
from dnadesign.cluster.src.runs.signatures import InputSignature, MethodSignature, UmapSignature
from dnadesign.cluster.src.workspaces.paths import resolve_workspace_dir


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_builtin_workspace_configs_stay_portable_and_use_cwd_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo_root()
    monkeypatch.chdir(tmp_path)
    promoter_workspace = load_workspace_config("promoter_clusters_v1")
    perm_workspace = load_workspace_config("perm_v1")

    assert isinstance(promoter_workspace, WorkspaceConfig)
    assert promoter_workspace.source == "builtin"
    assert perm_workspace.source == "builtin"
    assert "usr_root" not in promoter_workspace.input
    assert "highlight" not in promoter_workspace.umap
    assert "file" not in perm_workspace.input
    assert promoter_workspace.results_root == (tmp_path / "workspaces" / "promoter_clusters_v1" / "outputs" / "cluster")
    assert (
        promoter_workspace.workspace_dir
        == (repo_root / "src/dnadesign/cluster/workspaces/promoter_clusters_v1").resolve()
    )


def test_local_workspace_config_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "local_ws"
    workspace_dir.mkdir()
    input_file = workspace_dir / "inputs" / "records.parquet"
    input_file.parent.mkdir()
    input_file.write_text("stub", encoding="utf-8")
    highlight_file = workspace_dir / "inputs" / "ids.parquet"
    highlight_file.write_text("stub", encoding="utf-8")
    (workspace_dir / "config.yaml").write_text(
        (
            "schema_version: 1\n"
            "input:\n"
            "  file: inputs/records.parquet\n"
            "fit:\n"
            "  name: local_ws\n"
            "  x_col: infer__x\n"
            "umap:\n"
            "  name: local_ws\n"
            "  x_col: infer__x\n"
            "  highlight: inputs/ids.parquet\n"
            "analyze:\n"
            "  cluster_col: cluster__local_ws\n"
        ),
        encoding="utf-8",
    )

    config = load_workspace_config(workspace_dir)

    assert config.source == "local"
    assert config.input["file"] == str(input_file.resolve())
    assert config.umap["highlight"] == str(highlight_file.resolve())
    assert config.results_root == workspace_dir / "outputs" / "cluster"


def test_local_workspace_dir_wins_over_packaged_workspace_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_workspace = tmp_path / "perm_v1"
    local_workspace.mkdir()
    (local_workspace / "config.yaml").write_text("input: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_workspace_dir("perm_v1") == local_workspace.resolve()


def test_explicit_results_root_is_required_and_rejects_builtin_package_tree() -> None:
    with pytest.raises(RuntimeError, match="artifact roots are explicit"):
        explicit_results_root(None)

    with pytest.raises(RuntimeError, match="cannot default under the built-in package tree"):
        explicit_results_root(builtin_cluster_dir() / "results")


def test_explicit_results_root_accepts_workspace_output_root_from_current_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    promoter_workspace = load_workspace_config("promoter_clusters_v1")

    assert explicit_results_root(promoter_workspace.results_root) == promoter_workspace.results_root


def test_list_runs_is_read_only_for_missing_results_root(tmp_path: Path) -> None:
    root = tmp_path / "results"

    df = list_runs(root=root)

    assert df.empty
    assert not root.exists()


def test_list_runs_supports_narrow_index_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "results"
    add_or_update_index(
        RunIndexEntry(
            kind="fit",
            run_slug="perm_v1",
            alias="perm_v1",
            created_utc="2026-03-16T12:00:00+00:00",
            source_kind="parquet",
            source_ref="/tmp/in.parquet",
            x_col="infer__x",
            n_rows=12,
            n_clusters=3,
            method_id="leiden",
            method_params={"neighbors": 5, "resolution": 0.3},
            method_sig_hash="method-a",
            input_sig_hash="input-a",
            labels_path="/tmp/labels.parquet",
            status="complete",
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=None,
            sweep_path=None,
        ),
        root=root,
    )

    original_read_parquet = pd.read_parquet
    captured: list[dict[str, object]] = []

    def counted_read_parquet(*args, **kwargs):
        if Path(args[0]) == root / "index.parquet":
            captured.append(dict(kwargs))
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)

    runs = list_runs(
        root=root,
        filters={"kind": "fit", "alias": "perm_v1"},
        columns=("alias", "labels_path"),
    )

    assert runs.to_dict(orient="records") == [{"alias": "perm_v1", "labels_path": "/tmp/labels.parquet"}]
    assert captured
    assert set(captured[-1]["columns"]) == {"alias", "labels_path", "kind", "created_utc", "run_slug"}
    assert captured[-1]["filters"] == [[("kind", "=", "fit"), ("alias", "=", "perm_v1")]]


def test_public_workspace_helpers_expose_source_root_and_init_local_workspace(tmp_path: Path) -> None:
    source_root = builtin_workspaces_dir()
    created = init_workspace(workspace_id="audit_ws", root=tmp_path)

    assert source_root.name == "workspaces"
    assert created == tmp_path / "audit_ws"
    assert (created / "config.yaml").is_file()
    assert (created / "outputs" / "cluster").is_dir()
    text = (created / "config.yaml").read_text(encoding="utf-8")
    assert 'name: "audit_ws"' in text
    assert 'cluster_col: "cluster__audit_ws"' in text


def test_cluster_overlay_helpers_enforce_one_namespace_contract() -> None:
    assert cluster_overlay_col("perm_v1") == "cluster__perm_v1"
    assert cluster_overlay_col("perm_v1", "epistasis") == "cluster__perm_v1__epistasis"
    assert intra_sim_overlay_col("cluster__perm_v1") == "cluster__perm_v1__intra_sim"

    with pytest.raises(typer.BadParameter, match="fit label column"):
        intra_sim_overlay_col("cluster__perm_v1__meta")


def test_method_registry_is_explicit_and_fail_fast() -> None:
    method = get_method("leiden")
    kmeans = get_method("kmeans")

    assert method.method_id == "leiden"
    assert method.default_run_prefix == "leiden"
    assert method.fit_param_names == LEIDEN_FIT_PARAM_NAMES
    assert method.get_operation("resolution_sweep") is not None
    assert kmeans.method_id == "kmeans"
    assert "n_clusters" in kmeans.fit_param_names

    with pytest.raises(ValueError, match="Unsupported clustering method"):
        get_method("hdbscan")


def test_method_registry_supports_explicit_registration() -> None:
    registry = MethodRegistry()
    custom_method = ClusteringMethod(
        method_id="toy",
        display_name="Toy method",
        default_run_prefix="toy",
        fit_param_names=frozenset(),
        resolve_fit_params=lambda preset=None, raw_params=None: {},
        fit=lambda X, **_: np.zeros(X.shape[0], dtype=int),
        slug_params=lambda params: {},
        operations={},
    )

    registry.register_method(custom_method)

    assert registry.get_method("toy").display_name == "Toy method"


def test_parse_method_param_assignments_is_explicit_and_last_write_wins() -> None:
    parsed = parse_method_param_assignments(["neighbors=15", "metric=cosine", "neighbors=30"])

    assert parsed == {"neighbors": "30", "metric": "cosine"}

    with pytest.raises(ValueError, match="Expected key=value"):
        parse_method_param_assignments(["neighbors"])


def test_leiden_method_params_are_resolved_from_generic_raw_mapping() -> None:
    params = resolve_fit_params(
        preset={"neighbors": 15, "metric": "euclidean"},
        raw_params={"resolution": "0.8", "scale": "true", "backend": "leidenalg"},
    )

    assert params == {
        "neighbors": 15,
        "resolution": 0.8,
        "scale": True,
        "metric": "euclidean",
        "random_state": 42,
        "backend": "leidenalg",
    }

    with pytest.raises(ValueError, match="Unsupported Leiden fit params"):
        resolve_fit_params(raw_params={"bogus": "x"})


def test_kmeans_method_params_are_resolved_from_generic_raw_mapping() -> None:
    params = resolve_kmeans_fit_params(
        preset={"batch_size": 512, "max_iter": 50},
        raw_params={"n_clusters": "6", "init": "random", "reassignment_ratio": "0.05"},
    )

    assert params == {
        "n_clusters": 6,
        "batch_size": 512,
        "max_iter": 50,
        "random_state": 42,
        "n_init": "auto",
        "reassignment_ratio": 0.05,
        "tol": 0.0,
        "init": "random",
    }

    with pytest.raises(ValueError, match="Unsupported K-Means fit params"):
        resolve_kmeans_fit_params(raw_params={"bogus": "x"})


def test_feature_spec_requires_exactly_one_feature_definition() -> None:
    assert FeatureSpec.from_inputs(x_col="infer__x", x_cols=None).columns == ("infer__x",)
    assert FeatureSpec.from_inputs(x_col=None, x_cols="a,b").columns == ("a", "b")

    with pytest.raises(ValueError, match="Provide exactly one of --x-col or --x-cols"):
        FeatureSpec.from_inputs(x_col="x", x_cols="a,b")


def test_input_source_source_clause_is_explicit() -> None:
    usr = InputSource(kind="usr", source_ref="dataset_a", file=Path("/tmp/usr.parquet"), dataset="dataset_a")
    parquet = InputSource(kind="parquet", source_ref="/tmp/in.parquet", file=Path("/tmp/in.parquet"))

    assert usr.source_clause() == {"kind": "usr", "dataset": "dataset_a"}
    assert parquet.source_clause() == {"kind": "parquet", "file": "/tmp/in.parquet"}


def test_cluster_run_contract_builds_meta_and_index_payloads() -> None:
    source = InputSource(kind="parquet", source_ref="/tmp/in.parquet", file=Path("/tmp/in.parquet"))
    feature = FeatureSpec.from_inputs(x_col="infer__x", x_cols=None)
    method = MethodConfig(method_id="leiden", params={"neighbors": 30, "resolution": 0.8})
    request = FitRequest(source=source, key_col="id", feature=feature, method=method)
    input_sig = InputSignature(
        **request.input_signature_payload(
            row_ids_hash="abc123",
            x_dim=512,
            fingerprint={"mtime": 1, "size": 2},
        )
    )
    method_sig = MethodSignature(method_id="leiden", params=method.params, libs={})
    run = ClusterRun(
        alias="perm_v1",
        slug="perm_v1",
        created_utc="2026-03-16T12:00:00+00:00",
        input_signature=input_sig,
        method_signature=method_sig,
        source=source,
        feature=feature,
        x_dim=512,
        counts=RunCounts(n_rows=1000, n_clusters=37),
        wrote_usr_columns=False,
        attached_columns=("cluster__perm_v1", "cluster__perm_v1__meta", "cluster__perm_v1__quality"),
    )

    meta = run.meta_payload()
    index_entry = run.index_entry(labels_path=Path("/tmp/labels.parquet"))

    assert meta["io"] == {"kind": "parquet", "file": "/tmp/in.parquet"}
    assert meta["counts"] == {"n_rows": 1000, "n_clusters": 37}
    assert meta["columns"][-1] == "cluster__perm_v1__quality"
    assert index_entry.payload()["method_id"] == "leiden"
    assert index_entry.payload()["input_sig_hash"] == input_sig.hash()
    assert index_entry.payload()["labels_path"] == "/tmp/labels.parquet"


def test_embedding_run_contract_builds_meta_and_index_payloads() -> None:
    source = InputSource(kind="usr", source_ref="dataset_a", file=Path("/tmp/usr.parquet"), dataset="dataset_a")
    feature = FeatureSpec.from_inputs(x_col=None, x_cols="x1,x2")
    signature = UmapSignature(params={"neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}, libs={})
    run = EmbeddingRun(
        alias="perm_v1",
        slug="perm_v1_20260316T120000Z_deadbeef",
        created_utc="2026-03-16T12:00:00+00:00",
        source=source,
        feature=feature,
        counts=RunCounts(n_rows=1000),
        params={"neighbors": 15, "min_dist": 0.1, "metric": "euclidean"},
        signature=signature,
    )

    meta = run.meta_payload()
    index_entry = run.index_entry(coords_path=Path("/tmp/coords.parquet"), plot_root=Path("/tmp/umap"))

    assert meta["embedding_kind"] == "umap"
    assert meta["slug"] == "perm_v1_20260316T120000Z_deadbeef"
    assert meta["source"] == {"kind": "usr", "dataset": "dataset_a"}
    assert meta["x"] == {"col": "<multi>"}
    assert index_entry.payload()["coords_path"] == "/tmp/coords.parquet"
    assert index_entry.payload()["plot_paths"] == "/tmp/umap"
    assert index_entry.payload()["umap_slug"] == "perm_v1_20260316T120000Z_deadbeef"
    assert run.index_entry(coords_path=Path("/tmp/coords.parquet"), plot_root=None).payload()["plot_paths"] is None


def test_run_umap_file_write_reads_source_table_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.umap.compute as umap_compute_mod
    import dnadesign.cluster.src.umap.plot as umap_plot_mod
    from dnadesign.cluster.src.execution_umap import run_umap

    records_path = tmp_path / "records.parquet"
    out_path = tmp_path / "records_with_umap.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x1": [0.0, 0.1, 10.0, 10.1],
            "x2": [0.1, 0.2, 10.2, 10.3],
        }
    ).to_parquet(records_path, index=False)

    original_read_parquet = pd.read_parquet
    source_reads = {"count": 0}

    def counted_read_parquet(*args, **kwargs):
        if Path(args[0]) == records_path:
            source_reads["count"] += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)
    monkeypatch.setattr(
        umap_compute_mod,
        "compute",
        lambda X, neighbors, min_dist, metric, seed: np.column_stack(
            (np.arange(X.shape[0], dtype=np.float32), np.arange(X.shape[0], dtype=np.float32) + 1.0)
        ),
    )
    monkeypatch.setattr(
        umap_plot_mod,
        "scatter",
        lambda *args, **kwargs: kwargs["out_path"].write_text("png", encoding="utf-8"),
    )

    result = run_umap(
        dataset=None,
        file=str(records_path),
        usr_root=None,
        name="bench",
        key_col="id",
        x_col=None,
        x_cols="x1,x2",
        neighbors=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        preset=None,
        color_by=["numeric:x1"],
        highlight=None,
        highlight_topn=None,
        highlight_topn_col=None,
        highlight_topn_asc=False,
        highlight_hue_col=None,
        alpha=None,
        size=None,
        dims=None,
        font_scale=None,
        opal_campaign=None,
        opal_run=None,
        opal_as_of_round=None,
        opal_fields=None,
        derive_ratio=[],
        attach_coords=True,
        write=True,
        allow_overwrite=True,
        inplace=False,
        out=str(out_path),
        root=tmp_path / "results",
        console=None,
    )

    assert source_reads["count"] == 1
    written = original_read_parquet(out_path)
    assert result.artifact_path.is_dir()
    assert "cluster__bench__umap_x" in written.columns
    assert "cluster__bench__umap_y" in written.columns


def test_run_fit_file_write_reads_source_table_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    method_id = "toy_fit_reads_once"
    register_method(
        ClusteringMethod(
            method_id=method_id,
            display_name="Toy fit reads once",
            default_run_prefix="toy_fit_reads_once",
            fit_param_names=frozenset(),
            resolve_fit_params=lambda preset=None, raw_params=None: {},
            fit=lambda X, **_: np.zeros(X.shape[0], dtype=int),
            slug_params=lambda params: {},
            operations={},
        ),
        replace=True,
    )

    records_path = tmp_path / "records.parquet"
    out_path = tmp_path / "records_with_fit.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x1": [0.0, 0.1, 10.0, 10.1],
            "x2": [0.1, 0.2, 10.2, 10.3],
            "extra": ["u", "v", "w", "x"],
        }
    ).to_parquet(records_path, index=False)

    original_read_parquet = pd.read_parquet
    source_reads = {"count": 0}

    def counted_read_parquet(*args, **kwargs):
        if Path(args[0]) == records_path:
            source_reads["count"] += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)

    result = run_fit(
        results_root=tmp_path / "results",
        file=records_path,
        name="bench_fit",
        key_col="id",
        x_cols=("x1", "x2"),
        method=method_id,
        method_params={},
        write=True,
        allow_overwrite=True,
        inplace=False,
        out=out_path,
    )

    assert source_reads["count"] == 1
    written = original_read_parquet(out_path)
    assert result.artifact_path.is_dir()
    assert "cluster__bench_fit" in written.columns
    assert "cluster__bench_fit__meta" in written.columns
    assert written["extra"].tolist() == ["u", "v", "w", "x"]


def test_apply_fit_attachment_parquet_bypasses_pandas_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.execution_fit_support as fit_support

    records_path = tmp_path / "records.parquet"
    out_path = tmp_path / "records_with_fit.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "x1": [0.0, 1.0],
            "extra": ["u", "v"],
        }
    ).to_parquet(records_path, index=False)

    monkeypatch.setattr(
        fit_support,
        "write_generic",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic pandas write path should not be used")),
    )

    fit_support.apply_fit_attachment(
        ctx={"kind": "parquet", "file": records_path, "dataset": None, "usr_root": None},
        attach_cols=pd.DataFrame(
            {
                "id": ["a", "b"],
                "cluster__bench": [0, 1],
                "cluster__bench__meta": ["{}", "{}"],
            }
        ),
        key_col="id",
        allow_overwrite=True,
        inplace=False,
        out=str(out_path),
        attach_base_df=None,
        console=None,
    )

    written = pd.read_parquet(out_path)
    assert written.columns.tolist() == ["id", "x1", "extra", "cluster__bench", "cluster__bench__meta"]


def test_write_umap_overlays_parquet_bypasses_pandas_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.umap.overlays as overlays_module

    records_path = tmp_path / "records.parquet"
    out_path = tmp_path / "records_with_umap.parquet"
    base_df = pd.DataFrame({"id": ["a", "b"], "x1": [0.0, 1.0], "x2": [0.5, 1.5]})
    base_df.to_parquet(records_path, index=False)

    monkeypatch.setattr(
        overlays_module,
        "write_generic",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic pandas write path should not be used")),
    )

    overlays_module.write_umap_overlays(
        ictx={"kind": "parquet", "file": records_path, "dataset": None, "usr_root": None},
        attach_base_df=base_df,
        df=base_df.copy(),
        name="bench",
        key_col="id",
        coords=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        derived_cols=[],
        attach_coords=True,
        write=True,
        allow_overwrite=True,
        inplace=False,
        out=str(out_path),
        console=None,
    )

    written = pd.read_parquet(out_path)
    assert written.columns.tolist() == ["id", "x1", "x2", "cluster__bench__umap_x", "cluster__bench__umap_y"]


def test_analysis_run_contract_builds_meta_payload() -> None:
    source = InputSource(kind="usr", source_ref="dataset_a", file=Path("/tmp/usr.parquet"), dataset="dataset_a")
    run = AnalysisRun(
        alias="perm_v1",
        slug="perm_v1_20260316T120000Z_cafefeed",
        cluster_col="cluster__perm_v1",
        created_utc="2026-03-16T12:00:00+00:00",
        source=source,
        group_by=("source", "round"),
        out_dir=Path("/tmp/analysis"),
        composition=True,
        diversity=True,
        difffeat=False,
        plots=True,
        numeric_cols=("metric_a", "permuter__mut_count"),
        numeric_plots=False,
        font_scale=1.2,
        fit_alias="perm_v1",
        opal_fields=("pred__score",),
        opal_campaign="demo-campaign",
        opal_as_of_round=2,
    )

    meta = run.meta_payload()

    assert meta["cluster"] == {"column": "cluster__perm_v1", "fit_alias": "perm_v1"}
    assert meta["alias"] == "perm_v1"
    assert meta["slug"] == "perm_v1_20260316T120000Z_cafefeed"
    assert meta["out_dir"] == "/tmp/analysis"
    assert meta["group_by"] == ["source", "round"]
    assert meta["steps"] == {
        "composition": True,
        "diversity": True,
        "difffeat": False,
        "numeric": True,
    }
    assert meta["plots"] == {"enabled": True, "numeric": False, "font_scale": 1.2}
    assert meta["opal_join"] == {
        "campaign": "demo-campaign",
        "as_of_round": 2,
        "fields": ["pred__score"],
    }
    index_entry = run.index_entry(analysis_path=Path("/tmp/analysis/analysis.json"))
    assert index_entry.payload()["analysis_path"] == "/tmp/analysis/analysis.json"
    assert index_entry.payload()["kind"] == "analysis"


def test_run_index_entry_columns_are_centralized() -> None:
    assert RunIndexEntry.columns()[0] == "kind"
    assert "method_id" in RunIndexEntry.columns()
    assert "method_sig_hash" in RunIndexEntry.columns()
    assert "plot_paths" in RunIndexEntry.columns()
    assert "analysis_path" in RunIndexEntry.columns()
    assert "sweep_path" in RunIndexEntry.columns()


def test_sweep_run_contract_builds_meta_and_index_payloads() -> None:
    source = InputSource(kind="parquet", source_ref="/tmp/in.parquet", file=Path("/tmp/in.parquet"))
    feature = FeatureSpec.from_inputs(x_col="infer__x", x_cols=None)
    method_sig = MethodSignature(method_id="leiden", params={"neighbors": 5, "resolution": 0.5}, libs={})
    run = SweepRun(
        alias="leiden-sweep",
        slug="leiden-sweep",
        created_utc="2026-03-16T12:00:00+00:00",
        source=source,
        feature=feature,
        method_signature=method_sig,
        res_min=0.1,
        res_max=0.5,
        step=0.1,
        seeds=(1, 2, 3),
    )

    meta = run.meta_payload()
    index_entry = run.index_entry(sweep_path=Path("/tmp/sweeps/leiden-sweep/sweep.json"))

    assert meta["resolution"] == {"min": 0.1, "max": 0.5, "step": 0.1, "seeds": [1, 2, 3]}
    assert index_entry.payload()["kind"] == "sweep"
    assert index_entry.payload()["method_sig_hash"] == method_sig.hash()
    assert index_entry.payload()["sweep_path"] == "/tmp/sweeps/leiden-sweep/sweep.json"


def test_fit_alias_from_cluster_col_requires_label_shape() -> None:
    assert fit_alias_from_cluster_col("cluster__perm_v1") == "perm_v1"
    assert fit_alias_from_cluster_col("cluster__perm_v1__meta") is None
    assert fit_alias_from_cluster_col("permuter__metric") is None


def test_analysis_request_resolves_output_root_numeric_cols_and_opal_fields(tmp_path: Path) -> None:
    source = InputSource(kind="usr", source_ref="dataset_a", file=Path("/tmp/usr.parquet"), dataset="dataset_a")
    request = AnalysisRequest.from_runtime(
        source=source,
        df_columns=["id", "metric_a", "permuter__mut_count"],
        cluster_col="cluster__perm_v1",
        group_by=("source", "round"),
        out_dir=None,
        results_root=tmp_path / "results",
        composition=True,
        diversity=False,
        difffeat=False,
        plots=True,
        numeric="metric_a,obj__logic_fidelity",
        numeric_missing_policy="drop_and_log",
        numeric_plots=False,
        font_scale=1.2,
        opal_campaign="demo-campaign",
        opal_as_of_round=3,
        opal_fields="pred__score,obj__logic_fidelity",
    )

    assert request.out_dir == tmp_path / "results" / "perm_v1" / "analysis"
    assert request.group_by == ("source", "round")
    assert request.numeric_cols == ("metric_a", "obj__logic_fidelity", "permuter__mut_count")
    assert request.required_opal_fields == ("pred__score", "obj__logic_fidelity")
    assert request.command_payload()["opal_campaign"] == "demo-campaign"
    assert request.to_run(
        alias="perm_v1",
        slug="perm_v1_20260316T120000Z_deadbeef",
        created_utc="2026-03-16T12:00:00+00:00",
    ).meta_payload()["opal_join"]["fields"] == [
        "pred__score",
        "obj__logic_fidelity",
    ]


def test_workspace_config_rejects_unknown_keys(tmp_path: Path) -> None:
    workspace_dir = init_workspace(workspace_id="bad_ws", root=tmp_path)
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: 1
input:
  file: "../records.csv"
  typo_key: "oops"

fit:
  name: "bad_ws"
  x_cols: "x1,x2"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="unsupported keys: typo_key"):
        load_workspace_config(workspace_dir)


def test_workspace_config_rejects_unknown_umap_plot_keys(tmp_path: Path) -> None:
    workspace_dir = init_workspace(workspace_id="bad_plot_ws", root=tmp_path)
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: 1
input:
  file: "../records.csv"

fit:
  name: "bad_plot_ws"
  x_cols: "x1,x2"

umap:
  name: "bad_plot_ws"
  x_cols: "x1,x2"
  plot:
    bogus: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="umap.plot"):
        load_workspace_config(workspace_dir)


def test_fit_run_ledger_uses_unique_artifact_dir_per_repeated_alias(tmp_path: Path) -> None:
    records_path = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "x1": [0.0, 0.1, 10.0, 10.1, 20.0, 20.2],
            "x2": [0.1, 0.2, 10.2, 10.3, 20.1, 20.3],
        }
    ).to_csv(records_path, index=False)
    results_root = tmp_path / "results"

    first = run_fit(
        results_root=results_root,
        file=records_path,
        name="same",
        key_col="id",
        x_cols=("x1", "x2"),
        method="leiden",
        method_params={"neighbors": 2, "resolution": 0.2},
    )
    second = run_fit(
        results_root=results_root,
        file=records_path,
        name="same",
        key_col="id",
        x_cols=("x1", "x2"),
        method="leiden",
        method_params={"neighbors": 2, "resolution": 0.4},
    )

    runs = list_runs(root=results_root)

    assert first.artifact_path != second.artifact_path
    assert first.artifact_path.parent.parent == results_root / "same"
    assert second.artifact_path.parent.parent == results_root / "same"
    labels_paths = runs.loc[runs["kind"] == "fit", "labels_path"].tolist()
    assert len(labels_paths) == len(set(labels_paths))


def test_run_index_appends_to_delta_log_before_compaction(tmp_path: Path) -> None:
    root = tmp_path / "results"
    first = RunIndexEntry(
        kind="fit",
        run_slug="fit-1",
        alias="fit-1",
        created_utc="2026-03-18T10:00:00+00:00",
        source_kind="parquet",
        source_ref="/tmp/in.parquet",
        x_col="infer__x",
        n_rows=12,
        n_clusters=3,
        method_id="kmeans",
        method_params={"n_clusters": 3},
        method_sig_hash="method-1",
        input_sig_hash="input-1",
        labels_path="/tmp/labels-1.parquet",
        status="complete",
        umap_slug=None,
        umap_params=None,
        coords_path=None,
        plot_paths=None,
        analysis_path=None,
        sweep_path=None,
    )
    second = RunIndexEntry(
        **{
            **first.payload(),
            "run_slug": "fit-2",
            "alias": "fit-2",
            "created_utc": "2026-03-18T10:01:00+00:00",
            "labels_path": "/tmp/labels-2.parquet",
        }
    )

    add_or_update_index(first, root=root)
    snapshot = root / "index.parquet"
    assert snapshot.exists()
    snapshot_size = snapshot.stat().st_size

    add_or_update_index(second, root=root)

    delta_dir = root / "index.delta"
    assert snapshot.stat().st_size == snapshot_size
    assert len(list(delta_dir.glob("*.parquet"))) == 2

    runs = list_runs(root=root)
    assert runs["alias"].tolist() == ["fit-2", "fit-1"]

    compact_index(root=root)
    assert not delta_dir.exists()
    compacted = list_runs(root=root)
    assert compacted["alias"].tolist() == ["fit-2", "fit-1"]


def test_run_fit_supports_builtin_kmeans(tmp_path: Path) -> None:
    pytest.importorskip("sklearn.cluster")

    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "x1": [0.0, 0.1, 10.0, 10.1, 20.0, 20.2],
            "x2": [0.1, 0.2, 10.2, 10.3, 20.1, 20.3],
        }
    ).to_parquet(records_path, index=False)

    result = run_fit(
        results_root=tmp_path / "results",
        file=records_path,
        name="kmeans_fast",
        key_col="id",
        x_cols=("x1", "x2"),
        method="kmeans",
        method_params={"n_clusters": 3, "batch_size": 16},
    )

    runs = list_runs(root=tmp_path / "results")

    assert result.artifact_path.is_dir()
    fit_runs = runs.loc[runs["alias"] == "kmeans_fast"]
    assert len(fit_runs) == 1
    assert fit_runs.iloc[0]["method_id"] == "kmeans"


def test_run_umap_skips_plot_module_when_rendering_is_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cluster.src.umap.compute as umap_compute_mod
    from dnadesign.cluster.src.execution_umap import run_umap

    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x1": [0.0, 0.1, 10.0, 10.1],
            "x2": [0.1, 0.2, 10.2, 10.3],
        }
    ).to_parquet(records_path, index=False)

    monkeypatch.setattr(
        umap_compute_mod,
        "compute",
        lambda X, neighbors, min_dist, metric, seed: np.column_stack(
            (np.arange(X.shape[0], dtype=np.float32), np.arange(X.shape[0], dtype=np.float32) + 1.0)
        ),
    )
    sys.modules.pop("dnadesign.cluster.src.umap.plot", None)

    result = run_umap(
        dataset=None,
        file=str(records_path),
        usr_root=None,
        name="bench_noplot",
        key_col="id",
        x_col=None,
        x_cols="x1,x2",
        neighbors=2,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        preset=None,
        color_by=["numeric:x1"],
        highlight=None,
        highlight_topn=None,
        highlight_topn_col=None,
        highlight_topn_asc=False,
        highlight_hue_col=None,
        alpha=None,
        size=None,
        dims=None,
        font_scale=None,
        opal_campaign=None,
        opal_run=None,
        opal_as_of_round=None,
        opal_fields=None,
        derive_ratio=[],
        attach_coords=False,
        write=False,
        allow_overwrite=False,
        inplace=False,
        out=None,
        root=tmp_path / "results",
        console=None,
        render_plots=False,
    )

    runs = list_runs(root=tmp_path / "results")

    assert result.artifact_path.is_dir()
    assert "dnadesign.cluster.src.umap.plot" not in sys.modules
    umap_runs = runs.loc[runs["alias"] == "bench_noplot"]
    assert len(umap_runs) == 1
    assert pd.isna(umap_runs.iloc[0]["plot_paths"])


def test_usr_event_logging_fails_fast_when_event_log_is_unwritable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="event logging failed"):
        _append_usr_event(tmp_path / "missing" / ".events.log", {"event": "cluster_attach"})


def test_analysis_request_fails_fast_on_empty_work_and_bad_missing_policy(tmp_path: Path) -> None:
    source = InputSource(kind="parquet", source_ref="/tmp/in.parquet", file=Path("/tmp/in.parquet"))

    with pytest.raises(ValueError, match="Select at least one analysis step"):
        AnalysisRequest.from_runtime(
            source=source,
            df_columns=["id", "cluster__perm_v1"],
            cluster_col="cluster__perm_v1",
            group_by="source",
            out_dir=tmp_path / "analysis",
            results_root=None,
            composition=False,
            diversity=False,
            difffeat=False,
            plots=False,
            numeric=None,
            numeric_missing_policy="error",
            numeric_plots=True,
            font_scale=1.2,
            opal_campaign=None,
            opal_as_of_round=None,
            opal_fields=None,
        )

    with pytest.raises(ValueError, match="Unsupported numeric missing policy"):
        AnalysisRequest.from_runtime(
            source=source,
            df_columns=["id", "cluster__perm_v1", "metric_a"],
            cluster_col="cluster__perm_v1",
            group_by="source",
            out_dir=tmp_path / "analysis",
            results_root=None,
            composition=False,
            diversity=False,
            difffeat=False,
            plots=False,
            numeric="metric_a",
            numeric_missing_policy="warn",
            numeric_plots=True,
            font_scale=1.2,
            opal_campaign=None,
            opal_as_of_round=None,
            opal_fields=None,
        )


def test_analysis_request_requires_fit_label_when_out_dir_omitted(tmp_path: Path) -> None:
    source = InputSource(kind="parquet", source_ref="/tmp/in.parquet", file=Path("/tmp/in.parquet"))

    with pytest.raises(ValueError, match="must be a fit label column"):
        AnalysisRequest.from_runtime(
            source=source,
            df_columns=["id", "metric_a"],
            cluster_col="cluster__perm_v1__meta",
            group_by="source",
            out_dir=None,
            results_root=tmp_path / "results",
            composition=False,
            diversity=False,
            difffeat=False,
            plots=False,
            numeric="metric_a",
            numeric_missing_policy="error",
            numeric_plots=True,
            font_scale=1.2,
            opal_campaign=None,
            opal_as_of_round=None,
            opal_fields=None,
        )


def test_cli_config_resolution_uses_config_only_when_cli_value_is_default() -> None:
    assert (
        resolve_cli_or_config_value(
            parameter_source=click.core.ParameterSource.DEFAULT,
            cli_value="source",
            config_value="permuter__mut_count",
        )
        == "permuter__mut_count"
    )
    assert (
        resolve_cli_or_config_value(
            parameter_source=click.core.ParameterSource.COMMANDLINE,
            cli_value=False,
            config_value=True,
        )
        is False
    )


def test_coerce_numeric_preserves_input_and_fails_fast_on_bad_missing_policy() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "cluster__perm_v1": [0, 1],
            "metric_a": ["1.5", "2.5"],
            "extra": ["keep", "same"],
        }
    )

    out = _coerce_numeric(df, ["metric_a"], missing_policy="drop_and_log")

    assert df["metric_a"].tolist() == ["1.5", "2.5"]
    assert out["metric_a"].tolist() == [1.5, 2.5]
    assert out["extra"].tolist() == ["keep", "same"]

    with pytest.raises(ValueError, match="missing_policy must be 'error' or 'drop_and_log'"):
        _coerce_numeric(df, ["metric_a"], missing_policy="warn")


def test_find_equivalent_fit_requires_method_signature_hash(tmp_path: Path) -> None:
    root = tmp_path / "results"
    add_or_update_index(
        RunIndexEntry(
            kind="fit",
            run_slug="perm_v1",
            alias="perm_v1",
            created_utc="2026-03-16T12:00:00+00:00",
            source_kind="parquet",
            source_ref="/tmp/in.parquet",
            x_col="infer__x",
            n_rows=12,
            n_clusters=3,
            method_id="leiden",
            method_params={"neighbors": 5, "resolution": 0.3},
            method_sig_hash="method-a",
            input_sig_hash="input-a",
            labels_path="/tmp/labels.parquet",
            status="complete",
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=None,
            sweep_path=None,
        ),
        root=root,
    )

    assert find_equivalent_fit("input-a", "method-b", root=root) is None
    assert find_equivalent_fit("input-a", "method-a", root=root)["alias"] == "perm_v1"


def test_find_equivalent_fit_reads_only_required_index_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "results"
    add_or_update_index(
        RunIndexEntry(
            kind="fit",
            run_slug="perm_v1",
            alias="perm_v1",
            created_utc="2026-03-16T12:00:00+00:00",
            source_kind="parquet",
            source_ref="/tmp/in.parquet",
            x_col="infer__x",
            n_rows=12,
            n_clusters=3,
            method_id="leiden",
            method_params={"neighbors": 5, "resolution": 0.3},
            method_sig_hash="method-a",
            input_sig_hash="input-a",
            labels_path="/tmp/labels.parquet",
            status="complete",
            umap_slug=None,
            umap_params=None,
            coords_path=None,
            plot_paths=None,
            analysis_path=None,
            sweep_path=None,
        ),
        root=root,
    )

    original_read_parquet = pd.read_parquet
    captured: list[dict[str, object]] = []

    def counted_read_parquet(*args, **kwargs):
        if Path(args[0]) == root / "index.parquet":
            captured.append(dict(kwargs))
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counted_read_parquet)

    hit = find_equivalent_fit("input-a", "method-a", root=root)

    assert hit is not None
    assert hit["labels_path"] == "/tmp/labels.parquet"
    assert captured
    assert set(captured[-1]["columns"]) == FIT_REUSE_REQUIRED_COLUMNS.union(
        {"alias", "run_slug", "labels_path", "created_utc"}
    )
    assert captured[-1]["filters"] == [
        [("kind", "=", "fit"), ("input_sig_hash", "=", "input-a"), ("method_sig_hash", "=", "method-a")]
    ]


def test_extract_x_fails_fast_on_null_rows_in_json_array_column() -> None:
    df = pd.DataFrame({"id": ["a", "b"], "infer__x": [None, "[1.0, 2.0]"]})

    with pytest.raises(ValueError, match="contains null values"):
        extract_X(df, x_col="infer__x")
