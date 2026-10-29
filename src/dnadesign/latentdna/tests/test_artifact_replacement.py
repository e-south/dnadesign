"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_artifact_replacement.py

Failed computation must not replace an existing successful artifact.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import importlib
from types import SimpleNamespace

import pytest

CASES = [
    ("alignment", "build_alignment", "build_alignment_artifact", "alignments", {}),
    ("agreement", "compare_agreement", "compare_agreement_artifact", "agreements", {}),
    (
        "distance",
        "score_distance",
        "score_distance_artifact",
        "distances",
        {"view_id": "v", "landmark_ids": [], "metric": "cosine"},
    ),
    (
        "enrichment",
        "score_enrichment",
        "score_enrichment_artifact",
        "enrichments",
        {"neighbors_id": "n", "cohort_id": "c", "landmark_ids": []},
    ),
    ("snapshot", "build_snapshot", "build_snapshot_artifact", "snapshots", {"source_id": "s"}),
    (
        "sample",
        "build_sample",
        "build_sample_artifact",
        "samples",
        {"view_id": "v", "strategy": "random", "n": 2, "group_column": None, "seed": 1},
    ),
    ("scalar", "derive_scalar", "derive_scalar_artifact", "scalars", {}),
    ("scalar", "build_scalar", "build_scalar_artifact", "scalars", {"builder_kind": "representation_scorecard"}),
    (
        "neighbors",
        "fit_neighbors",
        "fit_neighbor_artifact",
        "neighbors",
        {
            "view_id": "v",
            "reduced_view_id": None,
            "k": 2,
            "metric": "cosine",
            "backend": "exact",
            "sample_id": None,
            "alignment_id": None,
            "seed": 1,
        },
    ),
    (
        "cluster",
        "fit_cluster",
        "fit_cluster_artifact",
        "clusters",
        {
            "view_id": "v",
            "reduced_view_id": None,
            "method": "kmeans",
            "n_clusters": 2,
            "seed": 1,
            "max_iter": 2,
            "sample_id": None,
            "alignment_id": None,
            "neighbor_set_id": None,
            "metric": "cosine",
            "k": 2,
            "resolution": 1.0,
        },
    ),
    ("export", "export_matrix", "build_export_matrix_artifact", "exports", {}),
    ("export", "export_table", "build_export_table_artifact", "exports", {}),
    ("export", "export_anndata", "build_export_anndata_artifact", "exports", {}),
]


@pytest.mark.parametrize("owner,operation,builder,directory,kwargs", CASES, ids=[case[1] for case in CASES])
def test_failed_forced_computation_preserves_previous_artifact(
    tmp_path, monkeypatch, owner, operation, builder, directory, kwargs
):
    service = importlib.import_module(f"dnadesign.latentdna.src.services.{owner}_service")
    context = SimpleNamespace(output_root=tmp_path, workspace_id="test")
    for name in ("alignment", "scalar", "export"):
        setattr(context, f"require_{name}", lambda _id: SimpleNamespace())
    monkeypatch.setattr(service, "load_workspace_config", lambda _: context)
    for name in ("evaluate_neighbors_preflight", "evaluate_cluster_preflight", "evaluate_export_preflight"):
        if hasattr(service, name):
            monkeypatch.setattr(service, name, lambda *args, **kwargs: None)
    if hasattr(service, "apply_memory_preflight"):
        monkeypatch.setattr(service, "apply_memory_preflight", lambda *args, **kwargs: ("ok", []))
    destination = tmp_path / directory / "result"
    destination.mkdir(parents=True)
    marker = destination / "valid-data"
    marker.write_bytes(b"prior successful output")

    def fail(*args, **kwargs):
        raise RuntimeError("injected computation failure")

    monkeypatch.setattr(service, builder, fail)
    with pytest.raises(RuntimeError, match="injected computation failure"):
        getattr(service, operation)(tmp_path, "result", force=True, **kwargs)
    assert marker.read_bytes() == b"prior successful output"


@pytest.mark.parametrize("owner,operation,builder,directory,kwargs", CASES, ids=[case[1] for case in CASES])
@pytest.mark.parametrize("invalid_id", ["absolute", "../escaped_artifact", "bad\n"])
def test_services_reject_unsafe_ids_before_creating_artifacts(
    tmp_path, monkeypatch, owner, operation, builder, directory, kwargs, invalid_id
):
    from dnadesign.latentdna.src.contracts.errors import WorkspaceValidationError

    service = importlib.import_module(f"dnadesign.latentdna.src.services.{owner}_service")
    root = tmp_path / "outputs"
    context = SimpleNamespace(output_root=root)
    for name in ("alignment", "scalar", "export"):
        setattr(context, f"require_{name}", lambda _id: SimpleNamespace())
    monkeypatch.setattr(service, "load_workspace_config", lambda _: context)
    if invalid_id == "absolute":
        invalid_id = str(tmp_path / "outside" / "escaped")
    with pytest.raises(WorkspaceValidationError):
        getattr(service, operation)(tmp_path, invalid_id, **kwargs)
    assert not root.exists()


@pytest.mark.parametrize("owner,operation,builder,directory,kwargs", CASES, ids=[case[1] for case in CASES])
def test_services_reject_symlinked_artifact_parents(
    tmp_path, monkeypatch, owner, operation, builder, directory, kwargs
):
    from dnadesign.latentdna.src.contracts.errors import WorkspaceValidationError

    service = importlib.import_module(f"dnadesign.latentdna.src.services.{owner}_service")
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / directory).symlink_to(outside, target_is_directory=True)
    context = SimpleNamespace(output_root=root)
    for name in ("alignment", "scalar", "export"):
        setattr(context, f"require_{name}", lambda _id: SimpleNamespace())
    monkeypatch.setattr(service, "load_workspace_config", lambda _: context)
    with pytest.raises(WorkspaceValidationError):
        getattr(service, operation)(tmp_path, "safe_id", **kwargs)
    assert not list(outside.iterdir())
