from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    CloningStrategy,
    SelectedCandidate,
    build_synthesis_manifest,
    campaign_synthesis_output_dir,
    read_azenta_workbook,
    render_azenta_workbook,
    render_campaign_scoped_exports,
    selected_candidates_from_batch0_review,
    selected_candidates_from_opal_round_campaigns,
    validate_azenta_workbook,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.cli import (
    main as synthesis_handoff_main,
)

LEFT_FLANK = "accgggatcctgcag"
RIGHT_FLANK = "tgagggaattcgcga"
CORE_A = "ACGT" * 15
CORE_B = "TGCA" * 15


def _strategy() -> CloningStrategy:
    return CloningStrategy(
        name="sfxi_promoter_insert",
        version="v1",
        left_flank=LEFT_FLANK,
        right_flank=RIGHT_FLANK,
        expected_core_length=60,
    )


def _selected_candidates() -> list[SelectedCandidate]:
    return [
        SelectedCandidate(
            campaign_slug="stress_eth_cip_ethanol_rf_sfxi_topn",
            as_of_round=0,
            run_id="run-eth-r0",
            selection_rank=1,
            id="opal-candidate-a",
            sequence=CORE_A,
            synthesis_name="ES-promoter-32",
        ),
        SelectedCandidate(
            campaign_slug="stress_eth_cip_cipro_rf_sfxi_topn",
            as_of_round=0,
            run_id="run-cipro-r0",
            selection_rank=2,
            id="opal-candidate-b",
            sequence=CORE_B,
            synthesis_name="ES-promoter-33",
        ),
    ]


def test_build_synthesis_manifest_preserves_ids_and_applies_case_aware_flanks() -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )

    assert list(manifest["id"]) == ["opal-candidate-a", "opal-candidate-b"]
    assert list(manifest["synthesis_name"]) == ["ES-promoter-32", "ES-promoter-33"]
    assert list(manifest["core_sequence"]) == [CORE_A, CORE_B]
    assert list(manifest["selection_epoch"]) == ["opal_model_round", "opal_model_round"]
    assert list(manifest["model_as_of_round"]) == [0, 0]
    assert manifest["assay_batch_index"].isna().all()
    assert list(manifest["left_flank"]) == [LEFT_FLANK, LEFT_FLANK]
    assert list(manifest["right_flank"]) == [RIGHT_FLANK, RIGHT_FLANK]
    assert list(manifest["final_sequence"]) == [
        f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}",
        f"{LEFT_FLANK}{CORE_B}{RIGHT_FLANK}",
    ]
    assert list(manifest["core_start"]) == [len(LEFT_FLANK), len(LEFT_FLANK)]
    assert list(manifest["core_end"]) == [len(LEFT_FLANK) + 60, len(LEFT_FLANK) + 60]
    assert list(manifest["final_length"]) == [90, 90]
    assert list(manifest["validation_status"]) == ["pass", "pass"]
    assert manifest.loc[0, "core_sha256"] == hashlib.sha256(CORE_A.encode("ascii")).hexdigest()
    assert (
        manifest.loc[0, "final_sha256"]
        == hashlib.sha256(f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}".encode("ascii")).hexdigest()
    )


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("acgt" * 15, "uppercase ACGT"),
        ("ACGT" * 14, "expected length 60"),
        ("ACGT" * 14 + "NNNN", "uppercase ACGT"),
    ],
)
def test_manifest_fails_fast_on_invalid_promoter_core(sequence: str, message: str) -> None:
    candidate = SelectedCandidate(
        campaign_slug="stress_eth_cip_ethanol_rf_sfxi_topn",
        as_of_round=0,
        run_id="run-eth-r0",
        selection_rank=1,
        id="opal-candidate-a",
        sequence=sequence,
        synthesis_name="ES-promoter-32",
    )

    with pytest.raises(ValueError, match=message):
        build_synthesis_manifest(
            selected=[candidate],
            strategy=_strategy(),
            batch_id="stress-opal-r0-20260617",
        )


def test_manifest_rejects_duplicate_candidate_ids_and_duplicate_order_aliases() -> None:
    duplicate_id = _selected_candidates()
    duplicate_id[1] = SelectedCandidate(
        campaign_slug="stress_eth_cip_cipro_rf_sfxi_topn",
        as_of_round=0,
        run_id="run-cipro-r0",
        selection_rank=2,
        id="opal-candidate-a",
        sequence=CORE_B,
        synthesis_name="ES-promoter-33",
    )
    with pytest.raises(ValueError, match="duplicate candidate id"):
        build_synthesis_manifest(selected=duplicate_id, strategy=_strategy(), batch_id="batch")

    duplicate_alias = _selected_candidates()
    duplicate_alias[1] = SelectedCandidate(
        campaign_slug="stress_eth_cip_cipro_rf_sfxi_topn",
        as_of_round=0,
        run_id="run-cipro-r0",
        selection_rank=2,
        id="opal-candidate-b",
        sequence=CORE_B,
        synthesis_name="ES-promoter-32",
    )
    with pytest.raises(ValueError, match="duplicate synthesis_name"):
        build_synthesis_manifest(selected=duplicate_alias, strategy=_strategy(), batch_id="batch")


def test_strategy_rejects_uppercase_or_non_dna_flanks() -> None:
    with pytest.raises(ValueError, match="left_flank must be lowercase acgt"):
        CloningStrategy(
            name="bad",
            version="v1",
            left_flank=LEFT_FLANK.upper(),
            right_flank=RIGHT_FLANK,
            expected_core_length=60,
        )

    with pytest.raises(ValueError, match="right_flank must be lowercase acgt"):
        CloningStrategy(
            name="bad",
            version="v1",
            left_flank=LEFT_FLANK,
            right_flank=f"{RIGHT_FLANK[:-1]}n",
            expected_core_length=60,
        )


def test_azenta_workbook_round_trip_matches_manifest(tmp_path: Path) -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )
    workbook_path = tmp_path / "azenta-order.xlsx"

    render_azenta_workbook(manifest, workbook_path)
    readback = read_azenta_workbook(workbook_path)
    report = validate_azenta_workbook(manifest, workbook_path)

    assert list(readback.columns[:2]) == ["Sequence Name", "Sequence"]
    assert readback[["Sequence Name", "Sequence"]].to_dict("records") == [
        {"Sequence Name": "ES-promoter-32", "Sequence": f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}"},
        {"Sequence Name": "ES-promoter-33", "Sequence": f"{LEFT_FLANK}{CORE_B}{RIGHT_FLANK}"},
    ]
    assert report == {
        "status": "pass",
        "row_count": 2,
        "workbook_path": str(workbook_path),
    }


def test_azenta_workbook_readback_fails_on_sequence_mismatch(tmp_path: Path) -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )
    workbook_path = tmp_path / "azenta-order.xlsx"
    broken = pd.DataFrame(
        [
            {"Sequence Name": "ES-promoter-32", "Sequence": f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}"},
            {"Sequence Name": "ES-promoter-33", "Sequence": f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}"},
        ]
    )
    broken.to_excel(workbook_path, sheet_name="Azenta Gene Synthesis", index=False)

    with pytest.raises(ValueError, match="sequence mismatch"):
        validate_azenta_workbook(manifest, workbook_path)


def test_batch0_review_rows_become_campaign_scoped_selected_candidates() -> None:
    review = pd.DataFrame(
        [
            {
                "campaign": "stress_eth_cip_ethanol_rf_sfxi_topn",
                "id": "eth-a",
                "sequence": CORE_A,
            },
            {
                "campaign": "stress_eth_cip_ethanol_rf_sfxi_topn",
                "id": "eth-b",
                "sequence": CORE_B,
            },
            {
                "campaign": "stress_eth_cip_cipro_rf_sfxi_topn",
                "id": "cip-a",
                "sequence": CORE_A,
            },
        ]
    )

    selected = selected_candidates_from_batch0_review(review)

    assert [
        (
            row.campaign_slug,
            row.selection_rank,
            row.id,
            row.synthesis_name,
            row.selection_source,
            row.selection_epoch,
            row.assay_batch_index,
            row.model_as_of_round,
        )
        for row in selected
    ] == [
        (
            "stress_eth_cip_ethanol_rf_sfxi_topn",
            1,
            "eth-a",
            "SECG-B0-ETH-01",
            "batch0_pre_assay",
            "pre_assay_seed",
            0,
            None,
        ),
        (
            "stress_eth_cip_ethanol_rf_sfxi_topn",
            2,
            "eth-b",
            "SECG-B0-ETH-02",
            "batch0_pre_assay",
            "pre_assay_seed",
            0,
            None,
        ),
        (
            "stress_eth_cip_cipro_rf_sfxi_topn",
            1,
            "cip-a",
            "SECG-B0-CIP-01",
            "batch0_pre_assay",
            "pre_assay_seed",
            0,
            None,
        ),
    ]


def test_batch0_review_rows_reject_unknown_campaign_for_alias_safety() -> None:
    review = pd.DataFrame(
        [
            {
                "campaign": "unknown_campaign",
                "id": "x",
                "sequence": CORE_A,
            }
        ]
    )

    with pytest.raises(ValueError, match="unknown batch-0 campaign slug"):
        selected_candidates_from_batch0_review(review)


def test_campaign_scoped_exports_write_manifest_and_gene_synthesis_workbook(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "stress_eth_cip_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
                {
                    "campaign": "stress_eth_cip_cipro_rf_sfxi_topn",
                    "id": "cip-a",
                    "sequence": CORE_B,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")

    exports = render_campaign_scoped_exports(
        manifest,
        batch_id="stress-opal-batch0-sfxi-v1",
        output_root=tmp_path,
    )

    assert list(exports["campaign_slug"]) == [
        "stress_eth_cip_ethanol_rf_sfxi_topn",
        "stress_eth_cip_cipro_rf_sfxi_topn",
    ]
    for _, row in exports.iterrows():
        manifest_path = Path(row["manifest_path"])
        workbook_path = Path(row["azenta_workbook_path"])
        assert manifest_path.exists()
        assert workbook_path.exists()
        assert manifest_path.parent == tmp_path / row["campaign_slug"]
        assert validate_azenta_workbook(pd.read_csv(manifest_path), workbook_path)["status"] == "pass"


def test_campaign_synthesis_output_dir_is_campaign_local() -> None:
    path = campaign_synthesis_output_dir(
        Path("/repo"),
        campaign_slug="stress_eth_cip_and_rf_sfxi_topn",
        batch_id="stress-opal-batch0-sfxi-v1",
    )

    assert path == Path(
        "/repo/src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/"
        "outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1"
    )


def _write_opal_round_fixture(
    tmp_path: Path,
    *,
    slug: str = "stress_eth_cip_ethanol_rf_sfxi_topn",
    as_of_round: int = 1,
    run_ids: tuple[str, ...] = ("run-eth-r1",),
    candidate_id_prefix: str = "",
    record_sequences: dict[str, str] | None = None,
    ledger_sequences: dict[str, str] | None = None,
) -> Path:
    campaign_root = tmp_path / slug
    config_dir = campaign_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "campaign.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            campaign:
              name: "test campaign"
              slug: "{slug}"
              workdir: "."
            data:
              location:
                kind: local
                path: records.parquet
              x_column_name: x
              y_column_name: y
              y_expected_length: 8
            transforms_x: {{ name: identity, params: {{}} }}
            transforms_y:
              name: sfxi_vec8_from_table_v1
              params:
                sequence_column: sequence
                logic_columns: ["v00", "v10", "v01", "v11"]
                intensity_columns: ["y00_star", "y10_star", "y01_star", "y11_star"]
                enforce_log2_offset_match: true
                expected_log2_offset_delta: 0.0
            model:
              name: random_forest
              params:
                n_estimators: 10
                random_state: 7
                n_jobs: 1
            objectives:
              - name: sfxi_v1
                params:
                  setpoint_vector: [0, 1, 0, 1]
                  logic_exponent_beta: 1.0
                  intensity_exponent_gamma: 1.0
                  intensity_log2_offset_delta: 0.0
                  scaling: {{ percentile: 95, min_n: 5, eps: 1.0e-8 }}
            selection:
              name: top_n
              params:
                top_k: 2
                score_ref: sfxi_v1/sfxi
                tie_handling: competition_rank
                objective_mode: maximize
            """
        ).format(slug=slug),
        encoding="utf-8",
    )

    candidate_a = f"{candidate_id_prefix}candidate-a"
    candidate_b = f"{candidate_id_prefix}candidate-b"
    candidate_c = f"{candidate_id_prefix}candidate-c"
    record_sequences = record_sequences or {candidate_a: CORE_A, candidate_b: CORE_B, candidate_c: "GATC" * 15}
    records = pd.DataFrame(
        [{"id": candidate_id, "sequence": sequence} for candidate_id, sequence in record_sequences.items()]
    )
    records.to_parquet(campaign_root / "records.parquet", index=False)

    ledger_sequences = ledger_sequences or record_sequences
    ledger_root = campaign_root / "outputs" / "ledger"
    predictions_dir = ledger_root / "predictions"
    predictions_dir.mkdir(parents=True)
    runs_dir = ledger_root / "runs.parquet"
    runs_dir.mkdir(parents=True)

    run_rows = [{"event": "run_meta", "run_id": run_id, "as_of_round": as_of_round} for run_id in run_ids]
    pd.DataFrame(run_rows).to_parquet(runs_dir / "part-runs.parquet", index=False)

    prediction_rows = []
    for run_id in run_ids:
        prediction_rows.extend(
            [
                {
                    "event": "run_pred",
                    "run_id": run_id,
                    "as_of_round": as_of_round,
                    "id": candidate_a,
                    "sequence": ledger_sequences[candidate_a],
                    "sel__rank_competition": 2,
                    "sel__is_selected": True,
                },
                {
                    "event": "run_pred",
                    "run_id": run_id,
                    "as_of_round": as_of_round,
                    "id": candidate_b,
                    "sequence": ledger_sequences[candidate_b],
                    "sel__rank_competition": 1,
                    "sel__is_selected": True,
                },
                {
                    "event": "run_pred",
                    "run_id": run_id,
                    "as_of_round": as_of_round,
                    "id": candidate_c,
                    "sequence": ledger_sequences[candidate_c],
                    "sel__rank_competition": 3,
                    "sel__is_selected": False,
                },
            ]
        )
    pd.DataFrame(prediction_rows).to_parquet(predictions_dir / "part-predictions.parquet", index=False)
    return config_path


def test_opal_round_ledgers_become_campaign_scoped_selected_candidates(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(tmp_path)

    selected, report = selected_candidates_from_opal_round_campaigns(
        [config_path],
        as_of_round=1,
    )

    assert [
        (row.campaign_slug, row.as_of_round, row.run_id, row.selection_rank, row.id, row.synthesis_name)
        for row in selected
    ] == [
        ("stress_eth_cip_ethanol_rf_sfxi_topn", 1, "run-eth-r1", 1, "candidate-b", "SECG-R1-ETH-01"),
        ("stress_eth_cip_ethanol_rf_sfxi_topn", 1, "run-eth-r1", 2, "candidate-a", "SECG-R1-ETH-02"),
    ]
    assert {row.selection_source for row in selected} == {"opal_ledger"}
    assert {row.selection_epoch for row in selected} == {"opal_model_round"}
    assert {row.model_as_of_round for row in selected} == {1}
    assert {row.assay_batch_index for row in selected} == {None}
    assert report["row_count"] == 2
    assert report["campaigns"][0]["selected_count"] == 2


def test_opal_round_source_rejects_ambiguous_reruns_without_run_id(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(tmp_path, run_ids=("run-a", "run-b"))

    with pytest.raises(ValueError, match="Multiple run_id values"):
        selected_candidates_from_opal_round_campaigns([config_path], as_of_round=1)


def test_opal_round_source_rejects_sequence_mismatch_against_records(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(
        tmp_path,
        ledger_sequences={"candidate-a": CORE_B, "candidate-b": CORE_B, "candidate-c": "GATC" * 15},
    )

    with pytest.raises(ValueError, match="sequence mismatch"):
        selected_candidates_from_opal_round_campaigns(
            [config_path],
            as_of_round=1,
            run_id_by_campaign={"stress_eth_cip_ethanol_rf_sfxi_topn": "run-eth-r1"},
        )


def test_cli_writes_opal_round_handoff_from_campaign_ledgers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_opal_round_fixture(tmp_path)
    output_root = tmp_path / "exports"

    exit_code = synthesis_handoff_main(
        [
            "--source",
            "opal-round",
            "--round",
            "1",
            "--campaign-config",
            str(config_path),
            "--output-dir",
            str(output_root),
            "--write",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["source"] == "opal-round"
    assert payload["batch_id"] == "stress-opal-r1-sfxi-v1"
    manifest_path = output_root / "stress_eth_cip_ethanol_rf_sfxi_topn" / "synthesis_manifest.csv"
    workbook_path = output_root / "stress_eth_cip_ethanol_rf_sfxi_topn" / "azenta_gene_synthesis.xlsx"
    assert manifest_path.exists()
    assert workbook_path.exists()
    manifest = pd.read_csv(manifest_path)
    assert manifest[["id", "synthesis_name", "selection_source", "selection_epoch", "model_as_of_round"]].to_dict(
        "records"
    ) == [
        {
            "id": "candidate-b",
            "synthesis_name": "SECG-R1-ETH-01",
            "selection_source": "opal_ledger",
            "selection_epoch": "opal_model_round",
            "model_as_of_round": 1,
        },
        {
            "id": "candidate-a",
            "synthesis_name": "SECG-R1-ETH-02",
            "selection_source": "opal_ledger",
            "selection_epoch": "opal_model_round",
            "model_as_of_round": 1,
        },
    ]
    assert validate_azenta_workbook(manifest, workbook_path)["status"] == "pass"


def test_cli_resolves_batch0_from_checked_in_handoff_record(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = synthesis_handoff_main(
        [
            "--handoff-id",
            "stress-opal-batch0-sfxi-v1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["source"] == "batch0"
    assert payload["batch_id"] == "stress-opal-batch0-sfxi-v1"
    assert payload["handoff_record"]["lifecycle_status"] == "generated_pending_acceptance"
    assert payload["handoff_record"]["selection_epoch"] == "pre_assay_seed"
    assert payload["handoff_record"]["artifact_status"]["summary"]["expected_campaign_count"] == 3
    expected_artifacts = {row["campaign_slug"]: row for row in payload["handoff_record"]["expected_artifacts"]}
    assert set(expected_artifacts) == {
        "stress_eth_cip_ethanol_rf_sfxi_topn",
        "stress_eth_cip_cipro_rf_sfxi_topn",
        "stress_eth_cip_and_rf_sfxi_topn",
    }
    assert expected_artifacts["stress_eth_cip_ethanol_rf_sfxi_topn"]["manifest_path"].endswith(
        "src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/outputs/"
        "synthesis_handoff/stress-opal-batch0-sfxi-v1/synthesis_manifest.csv"
    )
    assert expected_artifacts["stress_eth_cip_ethanol_rf_sfxi_topn"]["vendor_workbook_path"].endswith(
        "src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/outputs/"
        "synthesis_handoff/stress-opal-batch0-sfxi-v1/azenta_gene_synthesis.xlsx"
    )


def test_cli_handoff_record_resolves_opal_round_per_campaign_run_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ethanol_config = _write_opal_round_fixture(
        tmp_path,
        slug="stress_eth_cip_ethanol_rf_sfxi_topn",
        as_of_round=1,
        run_ids=("eth-run-a", "eth-run-b"),
    )
    cipro_config = _write_opal_round_fixture(
        tmp_path,
        slug="stress_eth_cip_cipro_rf_sfxi_topn",
        as_of_round=1,
        run_ids=("cip-run-a", "cip-run-b"),
        candidate_id_prefix="cip-",
    )
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        textwrap.dedent(
            """
            version: 1
            study_id: stress_ethanol_cipro_growth
            handoffs:
              - handoff_id: stress-opal-r1-sfxi-v1
                lifecycle_status: generated_pending_acceptance
                source_authority: opal_selection_set
                selection_epoch: opal_model_round
                assay_batch_index: 1
                model_as_of_round: 1
                run_id: null
                strategy_id: sfxi_promoter_insert:v1
                expected_campaigns:
                  - campaign_slug: stress_eth_cip_ethanol_rf_sfxi_topn
                    expected_rows: 2
                    run_id: eth-run-b
                    manifest_path: out/ethanol/synthesis_manifest.csv
                    vendor_workbook_path: out/ethanol/azenta_gene_synthesis.xlsx
                  - campaign_slug: stress_eth_cip_cipro_rf_sfxi_topn
                    expected_rows: 2
                    run_id: cip-run-a
                    manifest_path: out/cipro/synthesis_manifest.csv
                    vendor_workbook_path: out/cipro/azenta_gene_synthesis.xlsx
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = synthesis_handoff_main(
        [
            "--handoff-id",
            "stress-opal-r1-sfxi-v1",
            "--record-yaml",
            str(record_path),
            "--campaign-config",
            str(ethanol_config),
            "--campaign-config",
            str(cipro_config),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "opal-round"
    assert payload["batch_id"] == "stress-opal-r1-sfxi-v1"
    assert payload["handoff_record"]["manifest_validation"]["status"] == "pass"
    assert payload["handoff_record"]["assay_batch_index"] == 1
    assert {row["campaign_slug"]: row["run_id"] for row in payload["handoff_record"]["expected_artifacts"]} == {
        "stress_eth_cip_ethanol_rf_sfxi_topn": "eth-run-b",
        "stress_eth_cip_cipro_rf_sfxi_topn": "cip-run-a",
    }
    assert {row["campaign_slug"]: row["run_id"] for row in payload["source_report"]["campaigns"]} == {
        "stress_eth_cip_ethanol_rf_sfxi_topn": "eth-run-b",
        "stress_eth_cip_cipro_rf_sfxi_topn": "cip-run-a",
    }


def test_cli_opal_round_handoff_record_requires_explicit_run_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_opal_round_fixture(tmp_path, as_of_round=1)
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        textwrap.dedent(
            """
            version: 1
            study_id: stress_ethanol_cipro_growth
            handoffs:
              - handoff_id: stress-opal-r1-sfxi-v1
                lifecycle_status: generated_pending_acceptance
                source_authority: opal_selection_set
                selection_epoch: opal_model_round
                assay_batch_index: 1
                model_as_of_round: 1
                run_id: null
                strategy_id: sfxi_promoter_insert:v1
                expected_campaigns:
                  - campaign_slug: stress_eth_cip_ethanol_rf_sfxi_topn
                    expected_rows: 2
                    manifest_path: out/ethanol/synthesis_manifest.csv
                    vendor_workbook_path: out/ethanol/azenta_gene_synthesis.xlsx
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-sfxi-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "requires explicit run_id" in stderr
    assert "Traceback" not in stderr


def test_cli_handoff_record_rejects_campaign_count_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        textwrap.dedent(
            """
            version: 1
            study_id: stress_ethanol_cipro_growth
            handoffs:
              - handoff_id: stress-opal-batch0-sfxi-v1
                lifecycle_status: generated_pending_acceptance
                source_authority: study_batch0_selector
                selection_epoch: pre_assay_seed
                assay_batch_index: 0
                model_as_of_round: null
                run_id: batch0_pre_assay_review
                strategy_id: sfxi_promoter_insert:v1
                expected_campaigns:
                  - campaign_slug: stress_eth_cip_ethanol_rf_sfxi_topn
                    expected_rows: 7
                    manifest_path: out/ethanol/synthesis_manifest.csv
                    vendor_workbook_path: out/ethanol/azenta_gene_synthesis.xlsx
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-batch0-sfxi-v1",
                "--record-yaml",
                str(record_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "handoff record campaign row mismatch" in stderr
    assert "stress_eth_cip_ethanol_rf_sfxi_topn" in stderr
    assert "Traceback" not in stderr


def test_cli_opal_round_missing_ledger_exits_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_opal_round_fixture(tmp_path)
    shutil.rmtree(config_path.parent.parent / "outputs")

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--source",
                "opal-round",
                "--round",
                "1",
                "--campaign-config",
                str(config_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "required OPAL parquet artifact is missing" in stderr
    assert "Traceback" not in stderr
