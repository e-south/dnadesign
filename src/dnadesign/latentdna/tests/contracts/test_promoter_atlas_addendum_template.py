"""
Promoter-atlas addendum template contracts for latentdna.
"""

from __future__ import annotations

import yaml

from dnadesign.latentdna.src.workspaces.paths import builtin_templates_dir


def _template_config() -> dict[str, object]:
    config_path = builtin_templates_dir() / "landmark_atlas_committee" / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_landmark_atlas_committee_template_tracks_addendum_cohorts_and_views() -> None:
    payload = _template_config()

    cohorts = payload["cohorts"]
    assert set(cohorts) >= {
        "design_family",
        "design_regulator_composition",
        "sigma70_variant",
        "campaign_prior",
        "is_control",
        "source_class",
    }
    assert cohorts["design_family"]["kind"] == "promoter_metadata"
    assert cohorts["design_family"]["derive"] == "design_family"
    assert cohorts["design_regulator_composition"]["derive"] == "design_regulator_composition"

    views = payload["views"]
    assert {"z20_60", "z20_1k_anchor", "z7_60", "z7_1k_anchor"} <= set(views)
    assert {
        "logits7_60",
        "logits20_60",
        "logits7_1k_anchor",
        "logits20_1k_anchor",
        "z20_1k_seq",
        "z7_1k_seq",
        "drag20",
        "drag7",
    } <= set(views)

    pooled_logits_views = [
        view_payload
        for view_payload in views.values()
        if isinstance(view_payload, dict)
        and "vector" in view_payload
        and "output_layer_mean" in str(((view_payload["vector"] or {}).get("name") or ""))
    ]
    assert pooled_logits_views
    assert all(view_payload["tags"]["family"] == "pooled_logits" for view_payload in pooled_logits_views)


def test_landmark_atlas_committee_template_uses_canonical_landmarks_and_single_browser_notebook() -> None:
    payload = _template_config()

    assert True not in payload["alignments"]["anchor_seq_20b"]
    assert True not in payload["alignments"]["anchor_seq_7b"]
    assert "on" in payload["alignments"]["anchor_seq_20b"]
    assert "on" in payload["alignments"]["anchor_seq_7b"]
    assert True not in payload["scalars"]["context_audit_20b"]["derive"]
    assert "on" in payload["scalars"]["context_audit_20b"]["derive"]

    assert set(payload["landmarks"]) >= {"spyp", "sulap", "soxsp", "j23105"}

    notebooks = payload["notebooks"]
    assert list(notebooks) == ["browser"]
    assert notebooks["browser"]["kind"] == "workspace"

    plots = payload["plots"]
    assert {
        "atlas_2x2_intermediate_main",
        "atlas_2x3_model_family",
        "control_pca_explained_variance_curve",
        "cluster_correspondence_primary",
        "drag_qc_distribution",
        "context_shift_vs_drag_primary",
    } <= set(plots)

    deliverables = payload["deliverables"]
    assert {
        "atlas_2x2_intermediate_main",
        "geometry_switchboard_20b",
        "context_audit_primary_20b",
        "control_pca_explained_variance_curve",
        "cluster_correspondence_primary",
        "x2_primary_20b",
    } <= set(deliverables)
    assert all(
        isinstance(deliverables[deliverable_id].get("question"), str)
        and str(deliverables[deliverable_id]["question"]).strip()
        for deliverable_id in deliverables
    )
    assert all(
        isinstance(deliverables[deliverable_id].get("title"), str)
        and isinstance(deliverables[deliverable_id].get("summary"), str)
        and "docs_refs" in deliverables[deliverable_id]
        and "acceptance_checks" in deliverables[deliverable_id]
        for deliverable_id in deliverables
    )
