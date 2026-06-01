from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .probe_modules import probe_module

build_tfbs_stage_b_realized_label_review = probe_module("tfbs.stage_b.review").build_tfbs_stage_b_realized_label_review
main = probe_module("cli").main


def test_stage_b_realized_review_reports_true_label_lift_and_pair_deltas(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_fixture(tmp_path)

    result = build_tfbs_stage_b_realized_label_review(manifest_path)

    trajectories = pd.read_csv(result.trajectory_csv_path)
    pairs = pd.read_csv(result.pair_summary_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    positive_final = trajectories.query("label_name == 'lexA_present' and oracle_role == 'positive' and round == 1")
    null_final = trajectories.query("label_name == 'lexA_present' and oracle_role == 'matched_null' and round == 1")
    assert positive_final["selected_true_mean"].iloc[0] == pytest.approx(1.0)
    assert positive_final["pool_baseline"].iloc[0] == pytest.approx(0.5)
    assert positive_final["selected_true_lift_ratio"].iloc[0] == pytest.approx(2.0)
    assert positive_final["seed_label_count"].iloc[0] == 2
    assert positive_final["seed_true_mean"].iloc[0] == pytest.approx(0.5)
    assert positive_final["round_zero_semantics"].iloc[0] == "first_model_selected_batch_after_seed_labels"
    assert null_final["selected_true_mean"].iloc[0] == pytest.approx(0.5)
    assert null_final["selected_true_lift_ratio"].iloc[0] == pytest.approx(1.0)

    pair = pairs.iloc[0].to_dict()
    assert pair["label_name"] == "lexA_present"
    assert pair["positive_final_lift_ratio"] == pytest.approx(2.0)
    assert pair["null_final_lift_ratio"] == pytest.approx(1.0)
    assert pair["final_positive_minus_null_lift_ratio"] == pytest.approx(1.0)
    assert pair["positive_final_selected_true_sum"] == pytest.approx(2.0)
    assert pair["positive_final_selected_count"] == 2
    assert pair["positive_mean_round_lift_ratio"] > pair["null_mean_round_lift_ratio"]
    assert pair["positive_trapezoid_auc_lift_ratio"] == pytest.approx(1.5)
    assert pair["trapezoid_auc_positive_minus_null_lift_ratio"] == pytest.approx(1.0)
    assert pair["peer_review_claim_status"] == "positive_exceeds_null"
    assert summary["status"] == "PASS"
    assert summary["campaign_count"] == 2
    assert summary["pair_count"] == 1
    assert Path(summary["plot_manifest_json_path"]).exists()
    assert Path(summary["claim_assessment_csv_path"]).exists()
    claims = pd.read_csv(summary["claim_assessment_csv_path"])
    assert claims["claim_readiness_status"].tolist() == ["READY_AS_VALID_NULL_LEARNABILITY_SIGNAL"]
    assert summary["claim_readiness"]["ready_claim_count"] == 1
    assert summary["claim_readiness"]["blocked_or_limited_claim_count"] == 0
    plot_manifest = json.loads(Path(summary["plot_manifest_json_path"]).read_text(encoding="utf-8"))
    assert plot_manifest["plot_count"] == 2


def test_stage_b_realized_review_fails_fast_on_selected_ids_missing_from_label_table(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_fixture(tmp_path, include_missing_selection_id=True)

    with pytest.raises(ValueError, match="selected id"):
        build_tfbs_stage_b_realized_label_review(manifest_path)


def test_stage_b_review_cli_writes_realized_label_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_stage_b_fixture(tmp_path)

    assert main(["tfbs-stage-b-review", "--config-manifest", str(manifest_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert Path(payload["trajectory_csv_path"]).exists()
    assert Path(payload["pair_summary_csv_path"]).exists()
    assert Path(payload["plot_manifest_json_path"]).exists()
    assert payload["notebook_visual_registration"]["realized_label_review"]["status"] == "SKIPPED_INDEX_NOT_FOUND"
    assert payload["notebook_visual_registration"]["slot_count_diagnostics"]["status"] == "SKIPPED_NO_SLOT_LABELS"


def test_stage_b_realized_review_registers_notebook_collection_visuals(tmp_path: Path) -> None:
    stage_b_root = tmp_path / "stage_b"
    manifest_path = _write_stage_b_fixture(stage_b_root / "manifests")
    visual_index_path = stage_b_root / "notebooks" / "collection_visuals" / "collection_visual_manifest.json"
    visual_index_path.parent.mkdir(parents=True)
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v1",
                "generated_at": "2026-06-01T00:00:00+00:00",
                "collection_id": "fixture",
                "output_dir": str(visual_index_path.parent),
                "comparison_set_count": 0,
                "comparison_sets": [],
                "visual_count": 0,
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )

    result = build_tfbs_stage_b_realized_label_review(manifest_path)

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    refreshed = json.loads(visual_index_path.read_text(encoding="utf-8"))
    assert summary["notebook_visual_registration"]["realized_label_review"]["status"] == "REGISTERED"
    assert summary["notebook_visual_registration"]["slot_count_diagnostics"]["status"] == "SKIPPED_NO_SLOT_LABELS"
    assert refreshed["comparison_set_count"] == 1
    assert refreshed["visual_count"] == 2
    assert refreshed["comparison_sets"][0]["key"] == "stage_b_realized_label_review__lexA_present"
    assert refreshed["comparison_sets"][0]["label"] == "lexA_present positive/null pair"
    assert {visual["surface_kind"] for visual in refreshed["visuals"]} == {"study_realized_label_review"}
    assert {visual["metric"] for visual in refreshed["visuals"]} == {
        "selected_true_lift_ratio",
        "positive_minus_null_lift_ratio",
    }
    assert all(Path(visual["path"]).exists() for visual in refreshed["visuals"])

    build_tfbs_stage_b_realized_label_review(manifest_path)
    rerun = json.loads(visual_index_path.read_text(encoding="utf-8"))
    assert rerun["visual_count"] == 2
    assert len(rerun["visuals"]) == 2


def test_stage_b_realized_review_fails_on_invalid_notebook_visual_index_schema(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_fixture(tmp_path)
    visual_index_path = tmp_path / "bad_collection_visual_manifest.json"
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v0",
                "comparison_sets": [],
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported OPAL collection visual index schema"):
        build_tfbs_stage_b_realized_label_review(
            manifest_path,
            collection_visual_index_path=visual_index_path,
        )


def _write_stage_b_fixture(tmp_path: Path, *, include_missing_selection_id: bool = False) -> Path:
    campaigns = []
    pairs: dict[str, str] = {}
    for role in ("positive", "matched_null"):
        workdir = tmp_path / "campaigns" / f"lexA_present_{role}"
        config_path = workdir / "configs" / "campaign.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("campaign:\n  workdir: placeholder\n", encoding="utf-8")
        label_path = workdir / "labels.parquet"
        initial_label_path = workdir / "inputs" / "r0" / "labels-b0.parquet"
        values = [0, 0, 1, 1] if role == "positive" else [1, 0, 1, 0]
        frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "lexA_present": values})
        if role == "matched_null":
            frame["null_version"] = "densegen_tfbs_learnability_family_content_matched_null_v1"
        frame.to_parquet(label_path, index=False)
        initial_label_path.parent.mkdir(parents=True)
        frame.loc[frame["id"].isin(["a", "c"]), ["id", "lexA_present"]].to_parquet(initial_label_path, index=False)
        round_0_ids = ["c", "a"] if role == "positive" else ["b", "d"]
        _write_selection(workdir, 0, round_0_ids, scores=[0.1, 0.2])
        round_1_ids = ["c", "missing"] if include_missing_selection_id and role == "positive" else ["c", "d"]
        _write_selection(workdir, 1, round_1_ids, scores=[0.8, 0.7])
        campaign_key = f"lexA_present_{role}"
        pairs[role] = campaign_key
        campaigns.append(
            {
                "campaign_key": campaign_key,
                "label_name": "lexA_present",
                "label_family_id": "tf_family_presence",
                "oracle_role": role,
                "split_id": "random_id",
                "seed": 7,
                "selection_k": 2,
                "config_path": str(config_path),
                "label_table_path": str(label_path),
                "initial_label_input_path": str(initial_label_path),
            }
        )
    manifest = {
        "schema_version": "fixture.stage_b",
        "status": "PASS",
        "stage": "B",
        "scope": "sentinel",
        "rounds": 2,
        "selection_k": 2,
        "campaign_count": 2,
        "campaigns": campaigns,
        "pairs": [
            {
                "label_name": "lexA_present",
                "split_id": "random_id",
                "seed": 7,
                "positive_campaign_key": pairs["positive"],
                "null_campaign_key": pairs["matched_null"],
            }
        ],
    }
    manifest_path = tmp_path / "stage_b_sentinel_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _write_selection(workdir: Path, round_index: int, ids: list[str], *, scores: list[float]) -> None:
    path = workdir / "outputs" / "rounds" / f"round_{round_index}" / "selection" / "selection_top_k.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "pred__score_selected": scores}).to_csv(path, index=False)
