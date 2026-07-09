"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/measured_reader_vec8/test_staging.py

Tests measured Reader vec8 batch0 staging contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.measured_reader_vec8.staging import (
    build_measured_reader_vec8_staging,
    write_measured_reader_vec8_batch0,
)


def test_measured_reader_vec8_staging_maps_reader_designs_to_campaign_candidates(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    reader_root = tmp_path / "reader"
    _write_dnadesign_sources(repo_root)
    _write_reader_experiment(reader_root, "20260706_sfxi", time_h=12.04)
    _write_reader_experiment(reader_root, "20260707_sfxi", time_h=13.20)

    staging = build_measured_reader_vec8_staging(repo_root=repo_root, reader_root=reader_root)

    assert staging.summary == {
        "reader_vec8_rows": 6,
        "measured_candidate_rows": 3,
        "duplicate_candidate_rows": 3,
        "reader_sources": 2,
    }
    measured = staging.measured_frame.set_index("design_id")
    assert measured.loc["pDual-10-SECG-B0-ETH-01", "candidate_id"] == "candidate-eth-01"
    assert measured.loc["pDual-10-SECG-B0-ETH-01", "reader_experiment_id"] == "20260706_sfxi"
    assert measured.loc["pDual-10-SECG-B0-ETH-01", "campaign_role"] == "round0_observed_label_row"
    assert measured.loc["pDual-10-ES1p", "candidate_id"] == "seq-es1"
    assert measured.loc["pDual-10-spyp", "candidate_id"] == "seq-spyp"


def test_write_measured_reader_vec8_batch0_writes_campaign_ingest_csv(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    reader_root = tmp_path / "reader"
    _write_dnadesign_sources(repo_root)
    _write_reader_experiment(reader_root, "20260706_sfxi", time_h=12.04)

    result = write_measured_reader_vec8_batch0(
        repo_root=repo_root,
        reader_root=reader_root,
        overwrite=True,
    )

    path = result.campaign_inputs["secg_ethanol_rf_sfxi_topn"]
    assert path.exists()
    frame = pd.read_csv(path)
    assert frame["id"].tolist() == ["seq-es1", "candidate-eth-01", "seq-spyp"]
    assert frame["sequence"].tolist() == [
        "A" * 60,
        "GGACCAAATTACACAGTAATGCAAAAATTTTTAGACATTTGGCTGGTCGGAGACTATAAT",  # pragma: allowlist secret
        "T" * 220,
    ]
    assert frame["v00"].tolist() == [0.0, 0.0, 0.0]
    assert result.audit_csv.exists()
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["measured_rows_per_campaign_input"] == {
        "secg_and_rf_sfxi_topn": 3,
        "secg_cipro_rf_sfxi_topn": 3,
        "secg_ethanol_rf_sfxi_topn": 3,
    }
    assert manifest["round0_observed_label_pool"] == {
        "id": "measured_reader_vec8_round0",
        "role": "campaign_shared_observed_label_input",
        "rows_per_campaign_input": 3,
        "campaign_inputs_are_identical": True,
        "requires_existing_candidate_id_sequence_and_x": True,
        "reference_anchor_design_id": "pDual-10",
    }
    assert manifest["batch0_synthesis_seed"] == {
        "handoff_id": "stress-opal-batch0-sfxi-v1",
        "role": "physical_pre_assay_seed_order",
        "does_not_constrain_round0_observed_label_pool": True,
    }
    assert manifest["post_label_active_selection"] == {
        "role": "future_model_scored_active_learning_selection",
        "top_k_per_campaign": 6,
        "pooled_campaign_count": 3,
    }


def test_write_measured_reader_vec8_batch0_writes_reader_evidence_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    reader_root = tmp_path / "reader"
    _write_dnadesign_sources(repo_root)
    _write_reader_experiment(reader_root, "20260706_sfxi", time_h=12.04)

    result = write_measured_reader_vec8_batch0(
        repo_root=repo_root,
        reader_root=reader_root,
        overwrite=True,
    )

    evidence_path = result.campaign_inputs["secg_ethanol_rf_sfxi_topn"].parent / "reader_evidence_manifest.json"
    assert evidence_path.exists()
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "stress_ethanol_cipro_growth.reader_evidence.v1"
    assert payload["campaign_slug"] == "secg_ethanol_rf_sfxi_topn"
    assert payload["round"] == "r0"
    assert payload["summary"] == {
        "rows": 3,
        "distinct_ids": 3,
        "reader_experiments": 1,
        "artifact_count": 12,
        "missing_artifact_rows": 0,
    }
    rows = {row["design_id"]: row for row in payload["rows"]}
    eth = rows["pDual-10-SECG-B0-ETH-01"]
    assert eth["id"] == "candidate-eth-01"
    assert eth["reader_experiment_id"] == "20260706_sfxi"
    assert eth["time_selected_h"] == 12.04
    artifact_kinds = {artifact["semantic_kind"] for artifact in eth["artifacts"]}
    assert artifact_kinds == {
        "reader_vec8_table",
        "raw_kinetics",
        "intensity_overview",
        "sfxi_vec8_heatmap",
    }
    assert all(artifact["exists"] for artifact in eth["artifacts"])


def _write_dnadesign_sources(repo_root: Path) -> None:
    _write_synthesis_manifest(repo_root)
    _write_candidate_records(repo_root)
    _write_sfxi_sources(repo_root)
    _write_promoter_references(repo_root)
    _write_latentdna_rows(repo_root)


def _write_synthesis_manifest(repo_root: Path) -> None:
    base = (
        repo_root
        / "src/dnadesign/opal/campaigns/secg_ethanol_rf_sfxi_topn/outputs/synthesis_handoff"
        / "stress-opal-batch0-sfxi-v1"
    )
    base.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "candidate-eth-01",
        "synthesis_name": "SECG-B0-ETH-01",
        "core_sequence": "GGACCAAATTACACAGTAATGCAAAAATTTTTAGACATTTGGCTGGTCGGAGACTATAAT",  # pragma: allowlist secret
        "campaign_slug": "legacy_secg_ethanol_rf_sfxi_topn",
        "validation_status": "pass",
    }
    pd.DataFrame([row]).to_csv(base / "batch__synthesis_manifest.csv", index=False)
    for slug in ("secg_cipro_rf_sfxi_topn", "secg_and_rf_sfxi_topn"):
        other = (
            repo_root / "src/dnadesign/opal/campaigns" / slug / "outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1"
        )
        other.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    **row,
                    "id": f"{slug}-unused",
                    "synthesis_name": f"UNUSED-{slug}",
                    "campaign_slug": slug,
                }
            ]
        ).to_csv(other / "batch__synthesis_manifest.csv", index=False)


def _write_candidate_records(repo_root: Path) -> None:
    path = repo_root / "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "candidate-eth-01",
                "sequence": "GGACCAAATTACACAGTAATGCAAAAATTTTTAGACATTTGGCTGGTCGGAGACTATAAT",  # pragma: allowlist secret
                "latentdna__evo2_7b__context_anchor_mean_bidir_concat": [0.1, 0.2],
            },
            {
                "id": "seq-es1",
                "sequence": "A" * 60,
                "latentdna__evo2_7b__context_anchor_mean_bidir_concat": [0.1, 0.2],
            },
            {
                "id": "seq-spyp",
                "sequence": "T" * 220,
                "latentdna__evo2_7b__context_anchor_mean_bidir_concat": [0.1, 0.2],
            },
            {
                "id": "secg_cipro_rf_sfxi_topn-unused",
                "sequence": "A" * 60,
                "latentdna__evo2_7b__context_anchor_mean_bidir_concat": [0.1, 0.2],
            },
            {
                "id": "secg_and_rf_sfxi_topn-unused",
                "sequence": "C" * 60,
                "latentdna__evo2_7b__context_anchor_mean_bidir_concat": [0.1, 0.2],
            },
        ]
    ).to_parquet(path, index=False)


def _write_sfxi_sources(repo_root: Path) -> None:
    records_path = repo_root / "src/dnadesign/usr/datasets/usr_sfxi_pdual10_densegen_promoters/records.parquet"
    views_path = (
        repo_root / "src/dnadesign/usr/datasets/usr_sfxi_pdual10_densegen_promoters/_views/sequence_views.parquet"
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    views_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": "seq-es1", "sequence": "A" * 60}]).to_parquet(records_path, index=False)
    pd.DataFrame(
        [
            {
                "sequence_id": "seq-es1",
                "aliases": ["pDual-10-ES1p", "ES1p"],
            }
        ]
    ).to_parquet(views_path, index=False)


def _write_promoter_references(repo_root: Path) -> None:
    path = repo_root / "src/dnadesign/usr/datasets/usr_promoter_references/records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"id": "seq-j23105", "usr_label__primary": "J23105", "sequence": "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC"},
            {"id": "seq-spyp", "usr_label__primary": "spyp", "sequence": "T" * 220},
            {"id": "seq-sulap", "usr_label__primary": "sulAp", "sequence": "G" * 165},
        ]
    ).to_parquet(path, index=False)


def _write_latentdna_rows(repo_root: Path) -> None:
    path = (
        repo_root
        / "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/views"
        / "intermediate_embedding_7b_context_anchor_mean_bidir_concat/rows.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"construct__anchor_id": "seq-es1", "usr_label__primary": None, "sfxi_ref__reference_instance_id": None},
            {"construct__anchor_id": "seq-spyp", "usr_label__primary": "spyp", "sfxi_ref__reference_instance_id": None},
            {
                "construct__anchor_id": "seq-sulap",
                "usr_label__primary": "sulAp",
                "sfxi_ref__reference_instance_id": None,
            },
        ]
    ).to_parquet(path, index=False)


def _write_reader_experiment(reader_root: Path, experiment_id: str, *, time_h: float) -> None:
    exp = reader_root / "experiments/2026" / experiment_id
    outputs = exp / "outputs"
    table = outputs / "artifacts/sfxi_vec8.transform_sfxi__r2/vec8.parquet"
    table.parent.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "config.yaml").write_text("schema: reader/v7\n", encoding="utf-8")
    pd.DataFrame(
        [
            _vec8_row("pDual-10-SECG-B0-ETH-01", time_h=time_h),
            _vec8_row("pDual-10-ES1p", time_h=time_h),
            _vec8_row("pDual-10-spyp", time_h=time_h),
        ]
    ).to_parquet(table, index=False)
    plot_files = [
        "plots/ts_ETH-01.pdf",
        "plots/ts_ES1p.pdf",
        "plots/ts_spyP.pdf",
        "plots/ts_snap_YFP_CFP_design_id_alias_ETH-01.pdf",
        "plots/ts_snap_YFP_CFP_design_id_alias_ES1p.pdf",
        "plots/ts_snap_YFP_CFP_design_id_alias_spyP.pdf",
        "plots/sfxi_vec8_heatmap.pdf",
    ]
    for plot_file in plot_files:
        plot_path = outputs / plot_file
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"%PDF-1.4\n")
    records = {
        "schema_version": 3,
        "latest": {
            "sfxi_vec8/vec8": {
                "path": "artifacts/sfxi_vec8.transform_sfxi__r2/vec8.parquet",
            },
            "plot:raw_kinetics": {
                "files": ["plots/ts_ETH-01.pdf", "plots/ts_ES1p.pdf", "plots/ts_spyP.pdf"],
            },
            "plot:intensity_overview": {
                "files": [
                    "plots/ts_snap_YFP_CFP_design_id_alias_ETH-01.pdf",
                    "plots/ts_snap_YFP_CFP_design_id_alias_ES1p.pdf",
                    "plots/ts_snap_YFP_CFP_design_id_alias_spyP.pdf",
                ],
            },
            "plot:sfxi_vec8_heatmap": {
                "files": ["plots/sfxi_vec8_heatmap.pdf"],
            },
        },
    }
    manifest = outputs / "manifests/records.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records), encoding="utf-8")


def _vec8_row(design_id: str, *, time_h: float) -> dict[str, object]:
    return {
        "design_id": design_id,
        "sequence": None,
        "time_selected_h": time_h,
        "reference_design_id": "pDual-10",
        "intensity_log2_offset_delta": 0.0,
        "r_logic": 1.0,
        "v00": 0.0,
        "v10": 1.0,
        "v01": 0.0,
        "v11": 1.0,
        "y00_star": 0.1,
        "y10_star": 0.2,
        "y01_star": 0.3,
        "y11_star": 0.4,
        "flat_logic": False,
    }
