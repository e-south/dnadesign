from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .probe_modules import probe_module

build_tfbs_stage_b_slot_diagnostics = probe_module("tfbs.stage_b.slot_diagnostics").build_tfbs_stage_b_slot_diagnostics


def test_stage_b_slot_diagnostics_report_count_confound_and_restricted_lift(tmp_path: Path) -> None:
    manifest_path = _write_slot_fixture(tmp_path)

    result = build_tfbs_stage_b_slot_diagnostics(manifest_path)

    trajectory = pd.read_csv(result.trajectory_csv_path)
    pair_summary = pd.read_csv(result.pair_summary_csv_path)
    count_distribution = pd.read_csv(result.count_distribution_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    final_pos = trajectory.query("campaign_key == 'lexA_in_slot0_positive' and round == 1").iloc[0]
    final_null = trajectory.query("campaign_key == 'lexA_in_slot0_matched_null' and round == 1").iloc[0]
    assert final_pos["selected_target_count_mean"] == pytest.approx(1.5)
    assert final_pos["pool_target_count_mean"] == pytest.approx(1.5)
    assert final_pos["selected_nondeterministic_count"] == 2
    assert final_pos["count_stratified_expected_baseline"] == pytest.approx(0.5)
    assert final_pos["count_stratified_lift_ratio"] == pytest.approx(2.0)
    assert final_null["count_stratified_lift_ratio"] == pytest.approx(0.0)

    pair = pair_summary.iloc[0]
    assert pair["label_name"] == "lexA_in_slot0"
    assert pair["final_positive_minus_null_count_stratified_lift_ratio"] == pytest.approx(2.0)
    assert pair["slot_diagnostic_status"] == "position_signal_after_count_restriction"
    assert summary["slot_label_count"] == 1
    assert summary["resolved_position_signal_labels"] == ["lexA_in_slot0"]
    assert Path(summary["plot_manifest_json_path"]).exists()
    plot_manifest = json.loads(Path(summary["plot_manifest_json_path"]).read_text(encoding="utf-8"))
    assert [plot["kind"] for plot in plot_manifest["plots"]] == [
        "slot_target_count_mean_trajectory",
        "slot_count_stratified_lift_trajectory",
        "slot_count_stratified_lift_summary",
    ]
    assert plot_manifest["plots"][2]["title"] == "Count-stratified positive-minus-null slot lift"
    assert plot_manifest["plots"][2]["alt_text"].startswith("Bar plot comparing final")

    selected_dist = count_distribution.query(
        "campaign_key == 'lexA_in_slot0_positive' and round == 1 and target_count == 1"
    ).iloc[0]
    assert selected_dist["selected_count"] == 1
    assert selected_dist["pool_count"] == 2


def test_stage_b_slot_diagnostics_fail_fast_when_count_column_is_missing(tmp_path: Path) -> None:
    manifest_path = _write_slot_fixture(tmp_path, drop_count_column=True)

    with pytest.raises(ValueError, match="target-family count column"):
        build_tfbs_stage_b_slot_diagnostics(manifest_path)


def _write_slot_fixture(tmp_path: Path, *, drop_count_column: bool = False) -> Path:
    ids = ["a", "b", "c", "d", "e", "f"]
    counts = [0, 1, 1, 2, 2, 3]
    positive_labels = [0, 1, 0, 1, 0, 1]
    null_labels = [0, 0, 1, 0, 1, 1]
    campaigns = []
    pairs: dict[str, str] = {}
    for role, values in (("positive", positive_labels), ("matched_null", null_labels)):
        workdir = tmp_path / "campaigns" / f"lexA_in_slot0_{role}"
        config_path = workdir / "configs" / "campaign.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("campaign:\n  workdir: placeholder\n", encoding="utf-8")
        label_path = workdir / "labels.parquet"
        frame = pd.DataFrame({"id": ids, "lexA_count": counts, "lexA_in_slot0": values})
        if drop_count_column and role == "positive":
            frame = frame.drop(columns=["lexA_count"])
        frame.to_parquet(label_path, index=False)
        _write_selection(workdir, 0, ["b", "d"])
        _write_selection(workdir, 1, ["b", "d"])
        campaign_key = f"lexA_in_slot0_{role}"
        pairs[role] = campaign_key
        campaigns.append(
            {
                "campaign_key": campaign_key,
                "label_name": "lexA_in_slot0",
                "label_family_id": "tf_slot_family_presence",
                "oracle_role": role,
                "split_id": "random_id",
                "seed": 7,
                "selection_k": 2,
                "config_path": str(config_path),
                "label_table_path": str(label_path),
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
                "label_name": "lexA_in_slot0",
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


def _write_selection(workdir: Path, round_index: int, ids: list[str]) -> None:
    path = workdir / "outputs" / "rounds" / f"round_{round_index}" / "selection" / "selection_top_k.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "pred__score_selected": [0.9, 0.8]}).to_csv(path, index=False)
