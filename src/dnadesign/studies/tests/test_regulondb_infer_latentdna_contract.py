from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.latentdna.workspace_snapshot import decision_ladder
from dnadesign.latentdna.workspaces import load_workspace_config


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_regulondb_infer_lanes_request_vector_and_scalar_bundle() -> None:
    repo_root = _repo_root()
    cases = [
        (
            "src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/"
            "config.sequence_views.native_full.evo2_7b.yaml",
            "usr_regulondb_native_promoters",
            "source_record",
            "unknown",
            "seq_mean",
        ),
        (
            "src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/"
            "config.sequence_views.core60_tss_upstream.evo2_7b.yaml",
            "usr_regulondb_native_promoter_core60",
            "analysis_window",
            "forward",
            "core60_mean",
        ),
    ]

    for config_ref, dataset_id, product_kind, orientation, pooling_operation in cases:
        config = yaml.safe_load((repo_root / config_ref).read_text(encoding="utf-8"))
        bundle = config["jobs"][0]["feature_bundle"]
        sequence_input = bundle["sequence_view_inputs"][0]

        assert bundle["collect_intermediate_embedding"] is True
        assert bundle["collect_output_layer_mean"] is True
        assert bundle["collect_log_likelihood"] is True
        assert sequence_input["dataset"] == dataset_id
        assert sequence_input["view_selector"] == {
            "product_kind": product_kind,
            "orientation": orientation,
        }
        assert sequence_input["pooling"]["operation"] == pooling_operation


def test_regulondb_latentdna_binding_declares_native_and_core60_bundle_sources() -> None:
    repo_root = _repo_root()
    binding = yaml.safe_load(
        (repo_root / "docs/studies/regulondb_native_promoter_panel/latentdna_binding.yaml").read_text(encoding="utf-8")
    )

    expected_sources = {
        "native_records",
        "native_7b_seq_mean_features",
        "native_7b_seq_mean_output_layer_features",
        "native_7b_seq_mean_log_likelihood_mean",
        "native_7b_seq_mean_log_likelihood_total",
        "core60_tss_upstream",
        "core60_tss_upstream_7b_core60_mean_features",
        "core60_tss_upstream_7b_core60_mean_output_layer_features",
        "core60_tss_upstream_7b_core60_mean_log_likelihood_mean",
        "core60_tss_upstream_7b_core60_mean_log_likelihood_total",
    }

    assert expected_sources <= set(binding["source_datasets"])
    assert (
        "intermediate_embedding_7b_core60_tss_upstream"
        in binding["default_geometry_inventory"]["default_review_geometries"]
    )
    assert "log_likelihood_total_7b_core60_tss_upstream" in binding["default_geometry_inventory"]["scalar_diagnostics"]


def test_regulondb_latentdna_binding_decision_deliverables_match_workspace_ladder() -> None:
    repo_root = _repo_root()
    binding = yaml.safe_load(
        (repo_root / "docs/studies/regulondb_native_promoter_panel/latentdna_binding.yaml").read_text(encoding="utf-8")
    )
    context = load_workspace_config(repo_root / "src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel")

    assert binding["decision_deliverables"] == decision_ladder(context)
    assert set(binding["decision_deliverables"]) <= set(context.config.deliverables)
