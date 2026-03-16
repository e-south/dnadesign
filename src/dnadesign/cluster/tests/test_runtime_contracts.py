"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_runtime_contracts.py

Runtime path and method-contract tests for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import pytest

from dnadesign.cluster import (
    AnalysisRequest,
    AnalysisRun,
    ClusterRun,
    EmbeddingRun,
    FeatureSpec,
    FitRequest,
    InputSource,
    MethodConfig,
    RunCounts,
)
from dnadesign.cluster.contracts import RunIndexEntry
from dnadesign.cluster.src.analysis.numeric_per_cluster import _coerce_numeric
from dnadesign.cluster.src.cli.app import _resolve_cli_or_job_value
from dnadesign.cluster.src.jobs.loader import load_job_file
from dnadesign.cluster.src.layout import builtin_cluster_dir, configured_workspace_cluster_dir, default_results_root
from dnadesign.cluster.src.methods import parse_method_param_assignments
from dnadesign.cluster.src.methods.leiden import LEIDEN_FIT_PARAM_NAMES, resolve_fit_params
from dnadesign.cluster.src.methods.registry import get_method
from dnadesign.cluster.src.runs.contracts import fit_alias_from_cluster_col
from dnadesign.cluster.src.runs.signatures import InputSignature, MethodSignature, UmapSignature


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_load_job_file_resolves_paths_relative_to_job_file() -> None:
    repo_root = _repo_root()
    promoter_fit = load_job_file("src/dnadesign/cluster/jobs/promoter_clusters_v1/fit.yaml")
    perm_fit = load_job_file("src/dnadesign/cluster/jobs/perm_v1/fit.yaml")

    assert promoter_fit["params"]["usr_root"] == str((repo_root / "src/dnadesign/usr").resolve())
    assert perm_fit["params"]["file"] == str(
        (repo_root / "src/dnadesign/permuter/results/rt_combine_from_dms/records.parquet").resolve()
    )


def test_default_results_root_prefers_nearest_project_cluster_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_jobs = tmp_path / "workspace" / "cluster" / "jobs"
    cluster_jobs.mkdir(parents=True)
    monkeypatch.chdir(cluster_jobs)

    assert default_results_root() == (tmp_path / "workspace" / "cluster" / "results").resolve()


def test_default_results_root_falls_back_to_cwd_when_no_project_cluster_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert default_results_root() == (tmp_path / "results").resolve()


def test_default_results_root_rejects_builtin_package_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(builtin_cluster_dir() / "src")

    with pytest.raises(RuntimeError, match="cannot default under the built-in package tree"):
        default_results_root()


def test_configured_workspace_cluster_dir_rejects_builtin_package_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_CLUSTER_ROOT", str(builtin_cluster_dir()))

    with pytest.raises(RuntimeError, match="must point to a writable workspace 'cluster/' directory"):
        configured_workspace_cluster_dir()


def test_method_registry_is_explicit_and_fail_fast() -> None:
    method = get_method("leiden")

    assert method.method_id == "leiden"
    assert method.default_run_prefix == "leiden"
    assert method.fit_param_names == LEIDEN_FIT_PARAM_NAMES
    assert method.resolution_sweep is not None

    with pytest.raises(ValueError, match="Unsupported clustering method"):
        get_method("hdbscan")


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
    assert meta["source"] == {"kind": "usr", "dataset": "dataset_a"}
    assert meta["x"] == {"col": "<multi>"}
    assert index_entry.payload()["coords_path"] == "/tmp/coords.parquet"
    assert index_entry.payload()["plot_paths"] == "/tmp/umap"
    assert index_entry.payload()["umap_slug"] == "flat"


def test_analysis_run_contract_builds_meta_payload() -> None:
    source = InputSource(kind="usr", source_ref="dataset_a", file=Path("/tmp/usr.parquet"), dataset="dataset_a")
    run = AnalysisRun(
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


def test_run_index_entry_columns_are_centralized() -> None:
    assert RunIndexEntry.columns()[0] == "kind"
    assert "method_id" in RunIndexEntry.columns()
    assert "plot_paths" in RunIndexEntry.columns()


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
    assert request.to_run(created_utc="2026-03-16T12:00:00+00:00").meta_payload()["opal_join"]["fields"] == [
        "pred__score",
        "obj__logic_fidelity",
    ]


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


def test_cli_job_resolution_uses_job_only_when_cli_value_is_default() -> None:
    assert (
        _resolve_cli_or_job_value(
            parameter_source=click.core.ParameterSource.DEFAULT,
            cli_value="source",
            job_value="permuter__mut_count",
        )
        == "permuter__mut_count"
    )
    assert (
        _resolve_cli_or_job_value(
            parameter_source=click.core.ParameterSource.COMMANDLINE,
            cli_value=False,
            job_value=True,
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
