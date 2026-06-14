from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .probe_modules import probe_module

build_tfbs_stage_b_replicated_realized_label_review = probe_module(
    "tfbs.stage_b.review.replicates.materialization"
).build_tfbs_stage_b_replicated_realized_label_review
CANONICAL_COUNT_FRACTION_PROFILE_ID = probe_module("tfbs.profiles").CANONICAL_COUNT_FRACTION_PROFILE_ID
SLOT_POSITION_PROFILE_ID = probe_module("tfbs.profiles").SLOT_POSITION_PROFILE_ID
main = probe_module("cli").main


def test_stage_b_replicated_review_aggregates_seed_pair_rows_before_claims(tmp_path: Path) -> None:
    manifests = _write_three_replicate_fixtures(tmp_path)

    result = build_tfbs_stage_b_replicated_realized_label_review(
        manifests,
        out_dir=tmp_path / "replicated_review",
    )

    pair_summary = pd.read_csv(result.replicate_pair_summary_csv_path)
    endpoints = pd.read_csv(result.endpoint_summary_csv_path)
    claims = pd.read_csv(result.claim_assessment_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    plot_manifest = json.loads(result.plot_manifest_json_path.read_text(encoding="utf-8"))

    assert pair_summary["seed"].tolist() == [7, 17, 29]
    assert pair_summary["final_positive_minus_null_lift_ratio"].tolist() == pytest.approx([1.5, 1.0, 0.5])
    endpoint = endpoints.iloc[0].to_dict()
    assert endpoint["replicate_count"] == 3
    assert endpoint["replicate_seeds"] == "7,17,29"
    assert endpoint["final_positive_minus_null_lift_ratio_mean"] == pytest.approx(1.0)
    assert endpoint["final_positive_minus_null_lift_ratio_median"] == pytest.approx(1.0)
    assert endpoint["final_positive_minus_null_lift_ratio_q25"] == pytest.approx(0.75)
    assert endpoint["final_positive_minus_null_lift_ratio_q75"] == pytest.approx(1.25)
    assert endpoint["trapezoid_auc_positive_minus_null_lift_ratio_mean"] == pytest.approx(1.0)

    claim = claims.iloc[0].to_dict()
    assert claim["claim_readiness_status"] == "READY_AS_REPLICATED_VALID_NULL_LEARNABILITY_SIGNAL"
    assert claim["ready_replicate_count"] == 3
    assert bool(claim["claim_readiness_bool"]) is True
    assert summary["status"] == "PASS"
    assert summary["replicate_count"] == 3
    assert summary["replicate_seeds"] == [7, 17, 29]
    assert summary["endpoint_summary_csv_path"] == str(result.endpoint_summary_csv_path)
    assert plot_manifest["plot_count"] == 2
    assert {plot["interval"]["replicate_count"] for plot in plot_manifest["plots"]} == {3}
    assert {plot["interval_kind"] for plot in plot_manifest["plots"]} == {"sample_sd"}
    assert plot_manifest["plots"][0]["title"] == "LexA motif-count enrichment from promoter embeddings"
    assert plot_manifest["text_contract"]["interval"] == (
        "mean plus/minus sample SD across seed runs; n is recorded; not an inferential CI"
    )
    assert plot_manifest["text_contract"]["legend_layout"] == "single row below the plot"
    assert plot_manifest["text_contract"]["subtitle_layout"] == "centered single-line subtitle"
    assert plot_manifest["text_contract"]["title_alignment"] == "centered title; title may wrap, subtitle must not wrap"
    assert "selected-batch enrichment versus the candidate pool" in plot_manifest["plots"][0]["alt_text"]
    assert "Round 0 is the first acquired batch after the shared start" in plot_manifest["plots"][0]["alt_text"]
    assert len(plot_manifest["plots"][0]["alt_text"]) < 220
    visible_text = " ".join(str(plot[field]) for plot in plot_manifest["plots"] for field in ("title", "alt_text"))
    assert "oracle" not in visible_text.lower()


def test_stage_b_replicated_review_accepts_slot_position_profile_as_limited_boundary(
    tmp_path: Path,
) -> None:
    manifests = [
        _write_replicate_fixture(
            tmp_path,
            seed=7,
            label_name="lexA_in_slot0",
            label_family_id="tf_slot_family_presence",
            null_version="densegen_tfbs_learnability_slot_geometry_count_matched_null_v1",
            null_values=[0.75, 0.75, 0.25, 0.25],
            target_profile_id=SLOT_POSITION_PROFILE_ID,
        ),
        _write_replicate_fixture(
            tmp_path,
            seed=17,
            label_name="lexA_in_slot0",
            label_family_id="tf_slot_family_presence",
            null_version="densegen_tfbs_learnability_slot_geometry_count_matched_null_v1",
            null_values=[0.50, 0.50, 0.50, 0.50],
            target_profile_id=SLOT_POSITION_PROFILE_ID,
        ),
        _write_replicate_fixture(
            tmp_path,
            seed=29,
            label_name="lexA_in_slot0",
            label_family_id="tf_slot_family_presence",
            null_version="densegen_tfbs_learnability_slot_geometry_count_matched_null_v1",
            null_values=[0.25, 0.25, 0.75, 0.75],
            target_profile_id=SLOT_POSITION_PROFILE_ID,
        ),
    ]

    result = build_tfbs_stage_b_replicated_realized_label_review(
        manifests,
        out_dir=tmp_path / "slot_replicated_review",
    )

    claims = pd.read_csv(result.claim_assessment_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["target_profile"]["profile_id"] == SLOT_POSITION_PROFILE_ID
    assert summary["interpretation_boundary"] == summary["target_profile"]["interpretation_boundary"]
    assert summary["interval_boundary"] == (
        "Replicate bands use mean plus/minus sample standard deviation across deterministic seed pairs; "
        "they are descriptive spread, not inferential confidence intervals."
    )
    assert claims["claim_readiness_status"].tolist() == ["LIMITED_INVALID_NEGATIVE_CONTROL_REPLICATE"]
    assert claims["manuscript_claim_boundary"].iloc[0] == (
        "Do not claim valid-null learnability separation; at least one replicate lacks a valid matched null."
    )


def test_stage_b_replicated_review_rejects_incomplete_seed_set(tmp_path: Path) -> None:
    manifests = _write_three_replicate_fixtures(tmp_path)

    with pytest.raises(ValueError, match="requires replicate seeds \\[7, 17, 29\\]"):
        build_tfbs_stage_b_replicated_realized_label_review(
            manifests[:2],
            out_dir=tmp_path / "replicated_review",
        )


def test_stage_b_replicated_review_rejects_unpaired_initial_ids(tmp_path: Path) -> None:
    manifests = _write_three_replicate_fixtures(tmp_path, break_shared_start_seed=17)

    with pytest.raises(ValueError, match="positive/null initial label IDs differ"):
        build_tfbs_stage_b_replicated_realized_label_review(
            manifests,
            out_dir=tmp_path / "replicated_review",
        )


def test_stage_b_replicated_review_cli_writes_aggregate_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifests = _write_three_replicate_fixtures(tmp_path)
    args = ["tfbs-stage-b-replicate-review"]
    for manifest in manifests:
        args.extend(["--config-manifest", str(manifest)])
    args.extend(["--out-dir", str(tmp_path / "replicated_review"), "--json"])

    assert main(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["replicate_count"] == 3
    assert payload["replicate_seeds"] == [7, 17, 29]
    assert Path(payload["endpoint_summary_csv_path"]).exists()
    assert Path(payload["plot_manifest_json_path"]).exists()


def _write_three_replicate_fixtures(
    tmp_path: Path,
    *,
    break_shared_start_seed: int | None = None,
) -> list[Path]:
    return [
        _write_replicate_fixture(tmp_path, seed=7, null_values=[0.75, 0.75, 0.25, 0.25]),
        _write_replicate_fixture(
            tmp_path,
            seed=17,
            null_values=[0.50, 0.50, 0.50, 0.50],
            break_shared_start=break_shared_start_seed == 17,
        ),
        _write_replicate_fixture(tmp_path, seed=29, null_values=[0.25, 0.25, 0.75, 0.75]),
    ]


def _write_replicate_fixture(
    tmp_path: Path,
    *,
    seed: int,
    null_values: list[float],
    label_name: str = "lexA_count_fraction",
    label_family_id: str = "tf_family_count_fraction",
    null_version: str = "densegen_tfbs_learnability_family_content_matched_null_v1",
    target_profile_id: str = CANONICAL_COUNT_FRACTION_PROFILE_ID,
    break_shared_start: bool = False,
) -> Path:
    root = tmp_path / f"seed{seed}"
    campaigns = []
    campaign_keys: dict[str, str] = {}
    for role, values in {
        "positive": [0.0, 0.0, 1.0, 1.0],
        "matched_null": null_values,
    }.items():
        workdir = root / "campaigns" / f"{label_name}_{role}_seed{seed}"
        config_path = workdir / "configs" / "campaign.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("campaign:\n  workdir: placeholder\n", encoding="utf-8")
        label_path = workdir / "labels.parquet"
        label_frame = pd.DataFrame({"id": ["a", "b", "c", "d"], label_name: values})
        if role == "matched_null":
            label_frame["null_version"] = null_version
        label_frame.to_parquet(label_path, index=False)
        initial_ids = ["a", "d"] if break_shared_start and role == "matched_null" else ["a", "c"]
        initial_label_path = workdir / "inputs" / "r0" / "labels-b0.parquet"
        initial_label_path.parent.mkdir(parents=True)
        label_frame.loc[label_frame["id"].isin(initial_ids), ["id", label_name]].to_parquet(
            initial_label_path,
            index=False,
        )
        _write_selection(workdir, 0)
        _write_selection(workdir, 1)
        campaign_key = f"{label_name}_{role}_seed{seed}"
        campaign_keys[role] = campaign_key
        campaigns.append(
            {
                "campaign_key": campaign_key,
                "label_name": label_name,
                "label_family_id": label_family_id,
                "oracle_role": role,
                "split_id": "random_id",
                "seed": seed,
                "rounds": 2,
                "selection_k": 2,
                "initial_seed_policy": "label_value_stratified_random",
                "initial_seed_context": _seed_context(label_name=label_name, seed=seed),
                "initial_seed_source_role": "positive",
                "initial_label_ids_hash": "shared-start" if not break_shared_start else f"{role}-start",
                "config_path": str(config_path),
                "label_table_path": str(label_path),
                "initial_label_input_path": str(initial_label_path),
                "records_hash": "records-v1",
                "candidate_scope_hash": "candidate-scope-v1",
            }
        )
    manifest = {
        "schema_version": "fixture.stage_b",
        "status": "PASS",
        "stage": "B",
        "scope": "sentinel",
        "seed": seed,
        "rounds": 2,
        "selection_k": 2,
        "initial_label_count": 2,
        "campaign_count": 2,
        "records_hash": "records-v1",
        "candidate_scope_hash": "candidate-scope-v1",
        "sentinel_labels": [label_name],
        "target_profile": {"profile_id": target_profile_id, "canonical": False, "label_names": [label_name]},
        "campaigns": campaigns,
        "pairs": [
            {
                "label_name": label_name,
                "split_id": "random_id",
                "seed": seed,
                "null_permutation_seed": seed,
                "initial_seed_policy": "label_value_stratified_random",
                "initial_seed_context": _seed_context(label_name=label_name, seed=seed),
                "initial_seed_source_role": "positive",
                "initial_seed_pairing": "shared_positive_null_starting_ids",
                "initial_label_ids_hash": "shared-start",
                "positive_campaign_key": campaign_keys["positive"],
                "null_campaign_key": campaign_keys["matched_null"],
            }
        ],
    }
    manifest_path = root / "manifests" / "stage_b_sentinel_config_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _write_selection(workdir: Path, round_index: int) -> None:
    path = workdir / "outputs" / "rounds" / f"round_{round_index}" / "selection" / "selection_top_k.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["c", "d"], "pred__score_selected": [0.8, 0.7]}).to_csv(path, index=False)


def _seed_context(*, label_name: str, seed: int) -> str:
    return f"tfbs_stage_b_shared_initial_seed_v1:label={label_name}:split=random_id:seed={seed}"
