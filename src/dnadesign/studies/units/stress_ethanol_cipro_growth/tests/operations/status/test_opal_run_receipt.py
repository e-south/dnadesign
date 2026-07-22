"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/operations/status/test_opal_run_receipt.py

Focused tests for the study-owned OPAL round-0 receipt verifier.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.opal_run_receipt import (
    inspect_opal_round0_run_receipt,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.opal_surface import (
    inspect_opal_surface,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, object], Path, Path]:
    round_context_path = tmp_path / "outputs" / "rounds" / "round_0" / "metadata" / "round_ctx.json"
    selection_batch_path = tmp_path / "outputs" / "rounds" / "round_0" / "selection" / "selection_batch.parquet"
    round_context_path.parent.mkdir(parents=True)
    selection_batch_path.parent.mkdir(parents=True)
    run_id = "r0-test-run"
    campaign_slug = "secg_msrb_greedy"
    round_context_path.write_text(
        json.dumps(
            {
                "core/run_id": run_id,
                "core/round_index": 0,
                "core/campaign_slug": campaign_slug,
            }
        ),
        encoding="utf-8",
    )
    selection_batch_path.write_bytes(b"PAR1-test-selection-batch")
    round_context_ref = "repo:outputs/rounds/round_0/metadata/round_ctx.json"
    selection_batch_ref = "repo:outputs/rounds/round_0/selection/selection_batch.parquet"
    round_context_sha256 = _sha256(round_context_path)
    selection_batch_sha256 = _sha256(selection_batch_path)
    opal_config: dict[str, object] = {
        "round0": {
            "run_id": run_id,
            "campaign_slug": campaign_slug,
            "round_index": 0,
            "round_context": round_context_ref,
            "round_context_sha256": round_context_sha256,
            "selection_batch": selection_batch_ref,
            "selection_batch_sha256": selection_batch_sha256,
        }
    }
    artifacts = {
        "opal_round0_run_context": {
            "role": "opal_round_context",
            "ref": round_context_ref,
            "campaign_slug": campaign_slug,
            "round_index": 0,
            "run_id": run_id,
            "sha256": round_context_sha256,
        },
        "opal_round0_selection_batch": {
            "role": "opal_selection_batch",
            "ref": selection_batch_ref,
            "campaign_slug": campaign_slug,
            "round_index": 0,
            "run_id": run_id,
            "sha256": selection_batch_sha256,
        },
    }
    study_context = SimpleNamespace(
        study_repo_root=tmp_path,
        ops_contract=SimpleNamespace(artifacts=artifacts),
    )
    return study_context, opal_config, round_context_path, selection_batch_path


def test_opal_round0_receipt_verifies_declared_and_materialized_identity(tmp_path: Path) -> None:
    study_context, opal_config, round_context_path, selection_batch_path = _receipt_fixture(tmp_path)

    receipt = inspect_opal_round0_run_receipt(study_context=study_context, opal_config=opal_config)

    assert receipt["state"] == "ok"
    assert receipt["drives_top_level_attention"] is False
    assert receipt["mismatches"] == []
    assert receipt["run_id"] == "r0-test-run"
    assert receipt["campaign_slug"] == "secg_msrb_greedy"
    assert receipt["round_index"] == 0
    assert receipt["artifacts"] == {
        "round_context": {
            "ref": "repo:outputs/rounds/round_0/metadata/round_ctx.json",
            "resolved_ref": str(round_context_path),
            "declared_sha256": _sha256(round_context_path),
            "actual_sha256": _sha256(round_context_path),
            "verified": True,
        },
        "selection_batch": {
            "ref": "repo:outputs/rounds/round_0/selection/selection_batch.parquet",
            "resolved_ref": str(selection_batch_path),
            "declared_sha256": _sha256(selection_batch_path),
            "actual_sha256": _sha256(selection_batch_path),
            "verified": True,
        },
    }


def test_opal_round0_receipt_reports_declared_and_materialized_drift(tmp_path: Path) -> None:
    study_context, opal_config, round_context_path, selection_batch_path = _receipt_fixture(tmp_path)
    opal_config["round0"]["run_id"] = "r0-stale"  # type: ignore[index]
    selection_batch_path.write_bytes(b"PAR1-tampered")
    round_context_path.write_text(
        json.dumps(
            {
                "core/run_id": "r0-wrong-context",
                "core/round_index": 1,
                "core/campaign_slug": "wrong_campaign",
            }
        ),
        encoding="utf-8",
    )

    receipt = inspect_opal_round0_run_receipt(study_context=study_context, opal_config=opal_config)

    assert receipt["state"] == "attention"
    assert receipt["drives_top_level_attention"] is True
    mismatch_fields = {item["field"] for item in receipt["mismatches"]}
    assert mismatch_fields >= {
        "artifacts.round_context.actual_sha256",
        "artifacts.selection_batch.actual_sha256",
        "contract.round_context.run_id",
        "contract.selection_batch.run_id",
        "round_context.core/run_id",
        "round_context.core/round_index",
        "round_context.core/campaign_slug",
    }


def test_opal_round0_receipt_rejects_cross_record_ref_and_digest_drift(tmp_path: Path) -> None:
    study_context, opal_config, _, _ = _receipt_fixture(tmp_path)
    artifacts = study_context.ops_contract.artifacts
    artifacts["opal_round0_run_context"]["ref"] = "repo:outputs/wrong/round_ctx.json"
    artifacts["opal_round0_selection_batch"]["sha256"] = "0" * 64

    receipt = inspect_opal_round0_run_receipt(study_context=study_context, opal_config=opal_config)

    assert receipt["state"] == "attention"
    mismatch_fields = {item["field"] for item in receipt["mismatches"]}
    assert mismatch_fields >= {
        "contract.round_context.resolved_ref",
        "contract.selection_batch.sha256",
        "artifacts.round_context",
        "artifacts.selection_batch.actual_sha256",
    }


def test_opal_round0_receipt_is_not_configured_without_round0_declarations(tmp_path: Path) -> None:
    receipt = inspect_opal_round0_run_receipt(
        study_context=SimpleNamespace(study_repo_root=tmp_path, ops_contract=SimpleNamespace(artifacts={})),
        opal_config={},
    )

    assert receipt == {
        "configured": False,
        "state": "not_configured",
        "drives_top_level_attention": False,
        "summary": "OPAL round-0 run receipt is not configured",
        "mismatches": [],
    }


def test_opal_status_surface_exposes_round_receipt_integrity(tmp_path: Path) -> None:
    study_context, opal_config, _, _ = _receipt_fixture(tmp_path)
    study_context.study_pipeline = {
        "opal": {
            **opal_config,
            "config": "repo:campaigns/secg_msrb_greedy/configs/campaign.yaml",
            "candidate_feature_table": {
                "dataset": "candidate_table",
                "role": "opal_candidate_feature_table",
                "x_column": "x",
            },
        }
    }

    surface = inspect_opal_surface(study_context=study_context, default_doc="opal.md")

    assert surface["integrity_state"] == "ok"
    assert surface["run_receipt"]["state"] == "ok"
