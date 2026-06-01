from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_stage_b_source_fixture

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

_stage_b_configs = probe_module("tfbs.stage_b.configs")
TfbsStageBConfig = _stage_b_configs.TfbsStageBConfig
materialize_tfbs_stage_b_sentinel_configs = _stage_b_configs.materialize_tfbs_stage_b_sentinel_configs
main = probe_module("cli").main


def test_tfbs_stage_b_cli_generates_configs_from_stage_a_run_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_sentinel_configs"),
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["campaign_count"] == 10
    assert payload["validation_status"] == "SKIPPED"
    assert Path(payload["config_manifest_path"]).exists()
    assert Path(payload["collection_manifest_path"]).exists()


def test_tfbs_stage_b_cli_can_generate_restricted_label_set(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli-labels"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_sentinel_configs"),
                "--label-name",
                "lexA_present",
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["config_manifest_path"]).read_text(encoding="utf-8"))
    assert payload["campaign_count"] == 2
    assert manifest["sentinel_labels"] == ["lexA_present"]
    assert {row["label_name"] for row in manifest["campaigns"]} == {"lexA_present"}


def test_tfbs_stage_b_exact_budget_requires_seed_batch_matching_selection_k(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-budget"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )

    with pytest.raises(ValueError, match="exact-budget acquisition requires initial_label_count == selection_k"):
        materialize_tfbs_stage_b_sentinel_configs(
            TfbsStageBConfig(
                stage_a_run_root=stage_a_root,
                out_dir=stage_a_root / "stage_b_sentinel_configs",
                initial_label_count=3,
                selection_k=6,
                validate_configs=False,
            )
        )
