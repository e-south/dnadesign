"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/operations/status/test_synthesis_handoff_surface.py

Focused tests for the active OPAL synthesis handoff status surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.opal_surface import (
    inspect_opal_surface,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.synthesis_handoff_surface import (
    inspect_synthesis_handoff_surface,
)

_HANDOFF_ID = "stress-opal-assay-b1-r0-msrb-v1"
_CAMPAIGN_SLUG = "secg_msrb_greedy"
_RUN_ID = "r0-test-run"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_handoff_record(study_root: Path, *, lifecycle_status: str = "accepted_for_order") -> Path:
    record_path = study_root / "record" / "synthesis_handoffs.yaml"
    record_path.parent.mkdir(parents=True)
    digest = "a" * 64
    record_path.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "study_id": "stress_ethanol_cipro_growth",
                "record_kind": "synthesis_handoff_lifecycle",
                "handoffs": [
                    {
                        "handoff_id": _HANDOFF_ID,
                        "lifecycle_status": lifecycle_status,
                        "source_authority": "opal_selection_batch",
                        "selection_epoch": "opal_model_round",
                        "assay_batch_index": 1,
                        "model_as_of_round": 0,
                        "run_id": _RUN_ID,
                        "campaign_slug": _CAMPAIGN_SLUG,
                        "strategy_id": "stress_promoter_insert:v1",
                        "expected_selection_views": [
                            {"selection_view_id": "ethanol", "expected_rows": 1},
                        ],
                        "materialization_contract": {
                            "campaign_config": {"path": "inputs/campaign.yaml", "sha256": digest},
                            "selection_batch": {"path": "inputs/selection.parquet", "sha256": digest},
                            "candidate_records": {"path": "inputs/records.parquet", "sha256": digest},
                            "promoter_alias_registry": {"path": "inputs/aliases.yaml", "sha256": digest},
                            "cloning_strategy": {"path": "inputs/strategy.yaml", "sha256": digest},
                            "expected_candidates": [
                                {
                                    "study_alias": "SECG-019",
                                    "candidate_id": "candidate-019",
                                    "core_sha256": digest,
                                }
                            ],
                        },
                        "expected_artifact": {
                            "campaign_slug": _CAMPAIGN_SLUG,
                            "expected_rows": 1,
                            "manifest_path": "generated/manifest.csv",
                            "vendor_workbook_path": "generated/vendor.xlsx",
                            "genbank_dir_path": "generated/genbank",
                            "genbank_feature_table_path": "generated/features.csv",
                            "manifest_sha256": digest,
                            "vendor_workbook_sha256": digest,
                            "genbank_dir_sha256": digest,
                            "genbank_feature_table_sha256": digest,
                            "workbook_readback_status": "pass",
                            "genbank_readback_status": "pass",
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return record_path


def _fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, object], Path]:
    study_root = tmp_path / "docs" / "studies" / "stress_ethanol_cipro_growth"
    record_path = _write_handoff_record(study_root)
    study_context = SimpleNamespace(
        study_repo_root=tmp_path,
        resolved_study_dir=study_root,
        ops_contract=SimpleNamespace(
            record_sources={"synthesis_handoffs_ref": "manifest:record/synthesis_handoffs.yaml"},
            artifacts={},
        ),
    )
    opal_config: dict[str, object] = {
        "round0": {
            "run_id": _RUN_ID,
            "campaign_slug": _CAMPAIGN_SLUG,
            "round_index": 0,
        },
        "synthesis_handoff": {
            "handoff_id": _HANDOFF_ID,
            "required_lifecycle_status": "accepted_for_order",
        },
    }
    return study_context, opal_config, record_path


def test_synthesis_handoff_surface_rejects_accepted_record_without_order_bundle(tmp_path: Path) -> None:
    study_context, opal_config, record_path = _fixture(tmp_path)

    surface = inspect_synthesis_handoff_surface(study_context=study_context, opal_config=opal_config)

    assert surface["state"] == "attention"
    assert surface["drives_top_level_attention"] is True
    assert surface["resolved_record_ref"] == str(record_path)
    assert surface["vendor_submission"] == "unknown"
    assert surface["record"]["lifecycle_status"] == "accepted_for_order"
    assert surface["record"]["expected_study_aliases"] == ["SECG-019"]
    assert surface["record"]["artifact_status"]["summary"]["current_contract_ready"] is False
    assert surface["mismatches"] == [
        {
            "field": "synthesis_handoff.artifact_status.current_contract_ready",
            "expected": True,
            "actual": False,
        }
    ]


def test_synthesis_handoff_surface_reports_lifecycle_and_round_identity_drift(tmp_path: Path) -> None:
    study_context, opal_config, _ = _fixture(tmp_path)
    opal_config["round0"] = {
        "run_id": "wrong-run",
        "campaign_slug": "wrong-campaign",
        "round_index": 2,
    }
    opal_config["synthesis_handoff"]["required_lifecycle_status"] = "ordered"  # type: ignore[index]

    surface = inspect_synthesis_handoff_surface(study_context=study_context, opal_config=opal_config)

    assert surface["state"] == "attention"
    assert surface["drives_top_level_attention"] is True
    assert {item["field"] for item in surface["mismatches"]} >= {
        "synthesis_handoff.lifecycle_status",
        "synthesis_handoff.campaign_slug",
        "synthesis_handoff.run_id",
        "synthesis_handoff.model_as_of_round",
    }


def test_synthesis_handoff_surface_reports_malformed_record(tmp_path: Path) -> None:
    study_context, opal_config, record_path = _fixture(tmp_path)
    record_path.write_text("version: 3\nhandoffs: not-a-list\n", encoding="utf-8")

    surface = inspect_synthesis_handoff_surface(study_context=study_context, opal_config=opal_config)

    assert surface["state"] == "attention"
    assert surface["drives_top_level_attention"] is True
    assert {item["field"] for item in surface["mismatches"]} == {"synthesis_handoff.record"}


def test_opal_surface_integrity_includes_synthesis_handoff_mismatch(tmp_path: Path) -> None:
    study_context, opal_config, _ = _fixture(tmp_path)
    round_context_path = tmp_path / "round_ctx.json"
    selection_batch_path = tmp_path / "selection_batch.parquet"
    round_context_path.write_text(
        json.dumps(
            {
                "core/run_id": _RUN_ID,
                "core/round_index": 0,
                "core/campaign_slug": _CAMPAIGN_SLUG,
            }
        ),
        encoding="utf-8",
    )
    selection_batch_path.write_bytes(b"PAR1-test-selection-batch")
    round_context_ref = "repo:round_ctx.json"
    selection_batch_ref = "repo:selection_batch.parquet"
    opal_config["round0"].update(  # type: ignore[union-attr]
        {
            "round_context": round_context_ref,
            "round_context_sha256": _sha256(round_context_path),
            "selection_batch": selection_batch_ref,
            "selection_batch_sha256": _sha256(selection_batch_path),
        }
    )
    study_context.ops_contract.artifacts = {
        "opal_round0_run_context": {
            "role": "opal_round_context",
            "ref": round_context_ref,
            "campaign_slug": _CAMPAIGN_SLUG,
            "round_index": 0,
            "run_id": _RUN_ID,
            "sha256": _sha256(round_context_path),
        },
        "opal_round0_selection_batch": {
            "role": "opal_selection_batch",
            "ref": selection_batch_ref,
            "campaign_slug": _CAMPAIGN_SLUG,
            "round_index": 0,
            "run_id": _RUN_ID,
            "sha256": _sha256(selection_batch_path),
        },
    }
    opal_config["synthesis_handoff"]["required_lifecycle_status"] = "ordered"  # type: ignore[index]
    study_context.study_pipeline = {
        "opal": {
            **opal_config,
            "config": "repo:campaign.yaml",
            "candidate_feature_table": {
                "dataset": "candidate_table",
                "role": "opal_candidate_feature_table",
                "x_column": "x",
            },
        }
    }

    surface = inspect_opal_surface(study_context=study_context, default_doc="opal.md")

    assert surface["run_receipt"]["state"] == "ok"
    assert surface["synthesis_handoff"]["state"] == "attention"
    assert surface["integrity_state"] == "attention"
