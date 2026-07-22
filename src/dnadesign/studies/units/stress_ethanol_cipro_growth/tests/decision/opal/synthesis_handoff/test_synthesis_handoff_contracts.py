"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_contracts.py

Regression tests for synthesis handoff studies units stress ethanol cipro growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from Bio import SeqIO

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    CloningStrategy,
    SelectedCandidate,
    SelectionMembership,
    artifact_status_for_handoff_record,
    build_genbank_feature_table,
    build_synthesis_manifest,
    campaign_synthesis_artifact_paths,
    campaign_synthesis_output_dir,
    read_azenta_workbook,
    render_azenta_workbook,
    render_campaign_scoped_exports,
    render_genbank_record_set,
    selected_candidates_from_batch0_review,
    selected_candidates_from_opal_round,
    source_evidence_synthesis_output_dir,
    validate_azenta_workbook,
    validate_genbank_record_set,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    cli as synthesis_handoff_cli,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    exports as synthesis_handoff_exports,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    opal_round_source as synthesis_handoff_opal_round_source,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff import (
    records as synthesis_handoff_records,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.batch0_source import (
    batch0_synthesis_name,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.cli import (
    main as synthesis_handoff_main,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.records import (
    ExpectedHandoffArtifact,
    ExpectedSelectionView,
    SynthesisHandoffRecord,
    apply_handoff_record_lifecycle,
    load_synthesis_handoff_records,
    validate_manifest_against_handoff_record,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.strategy import (
    load_cloning_strategy,
)

LEFT_FLANK = "accgggatcctgcag"
RIGHT_FLANK = "tgagggaattcgcga"
CORE_A = "ACGT" * 15
CORE_B = "TGCA" * 15
CORE_ECORI_LEFT_JUNCTION = "AATTC" + "A" * 55


def _membership(selection_view_id: str, rank: int, *, score: float | None = None) -> SelectionMembership:
    return SelectionMembership(
        selection_view_id=selection_view_id,
        rank=rank,
        score=score,
        score_ref=(None if score is None else f"{selection_view_id}/score"),
    )


def _json_cli_error(stderr: str) -> dict[str, Any]:
    payload = json.loads(stderr)
    assert payload["status"] == "error"
    assert payload["context"] == "synthesis_handoff"
    assert "Traceback" not in stderr
    assert "usage:" not in stderr
    return payload


def _strategy() -> CloningStrategy:
    return load_cloning_strategy(
        Path(
            "src/dnadesign/studies/units/stress_ethanol_cipro_growth/"
            "decision/opal/synthesis_handoff/configs/stress_promoter_insert_v1.yaml"
        )
    )


def _materialization_contract(
    *bindings: tuple[str, str, str],
    campaign_config: tuple[str, str] = ("inputs/campaign.yaml", "0" * 64),
    selection_batch: tuple[str, str] = ("inputs/selection_batch.parquet", "1" * 64),
    candidate_records: tuple[str, str] = ("inputs/candidate_records.parquet", "2" * 64),
    promoter_alias_registry: tuple[str, str] = ("inputs/promoter_aliases.yaml", "3" * 64),
    cloning_strategy: tuple[str, str] = ("inputs/strategy.yaml", "4" * 64),
) -> Any:
    receipt = synthesis_handoff_records.MaterializationInputReceipt
    expected_candidate = synthesis_handoff_records.ExpectedMaterializedCandidate
    return synthesis_handoff_records.MeasuredRoundMaterializationContract(
        campaign_config=receipt(path=campaign_config[0], sha256=campaign_config[1]),
        selection_batch=receipt(path=selection_batch[0], sha256=selection_batch[1]),
        candidate_records=receipt(path=candidate_records[0], sha256=candidate_records[1]),
        promoter_alias_registry=receipt(
            path=promoter_alias_registry[0],
            sha256=promoter_alias_registry[1],
        ),
        cloning_strategy=receipt(path=cloning_strategy[0], sha256=cloning_strategy[1]),
        expected_candidates=tuple(
            expected_candidate(
                study_alias=study_alias,
                candidate_id=candidate_id,
                core_sha256=core_sha256,
            )
            for study_alias, candidate_id, core_sha256 in bindings
        ),
    )


def _selected_candidates() -> list[SelectedCandidate]:
    return [
        SelectedCandidate(
            campaign_slug="secg_ethanol_rf_sfxi_topn",
            selection_memberships=(_membership("ethanol", 1),),
            as_of_round=0,
            run_id="run-eth-r0",
            selection_rank=1,
            id="opal-candidate-a",
            sequence=CORE_A,
            synthesis_name="ES-promoter-32",
        ),
        SelectedCandidate(
            campaign_slug="secg_cipro_rf_sfxi_topn",
            selection_memberships=(_membership("ciprofloxacin", 2),),
            as_of_round=0,
            run_id="run-cipro-r0",
            selection_rank=2,
            id="opal-candidate-b",
            sequence=CORE_B,
            synthesis_name="ES-promoter-33",
        ),
    ]


def _batch0_cli_source_fixture() -> list[SelectedCandidate]:
    campaigns = (
        "secg_ethanol_rf_sfxi_topn",
        "secg_cipro_rf_sfxi_topn",
        "secg_and_rf_sfxi_topn",
    )
    selected: list[SelectedCandidate] = []
    sequence_ordinal = 0
    for campaign_slug in campaigns:
        for rank in range(1, 7):
            sequence_ordinal += 1
            prefix = "".join("ACGT"[(sequence_ordinal >> shift) & 3] for shift in (4, 2, 0))
            selected.append(
                SelectedCandidate(
                    campaign_slug=campaign_slug,
                    selection_memberships=(
                        _membership(
                            {
                                "secg_ethanol_rf_sfxi_topn": "ethanol",
                                "secg_cipro_rf_sfxi_topn": "ciprofloxacin",
                                "secg_and_rf_sfxi_topn": "and",
                            }[campaign_slug],
                            rank,
                        ),
                    ),
                    as_of_round=0,
                    run_id="batch0_pre_assay_review",
                    selection_rank=rank,
                    id=f"{campaign_slug}-batch0-{rank:02d}",
                    sequence=prefix + "A" * 57,
                    synthesis_name=batch0_synthesis_name(campaign_slug, rank),
                    selection_source="batch0_pre_assay",
                    selection_epoch="pre_assay_seed",
                    assay_batch_index=0,
                    model_as_of_round=None,
                )
            )
    return selected


def _patch_batch0_cli_source(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _batch0_cli_source_fixture()

    def fake_build_batch0_selected_candidates(*, config_path: Path, repo_root: Path | None = None):
        return selected, {
            "source": "batch0_pre_assay",
            "config_path": str(config_path),
            "campaign_counts": {
                "secg_ethanol_rf_sfxi_topn": 6,
                "secg_cipro_rf_sfxi_topn": 6,
                "secg_and_rf_sfxi_topn": 6,
            },
            "row_count": len(selected),
        }

    monkeypatch.setattr(
        synthesis_handoff_cli, "build_batch0_selected_candidates", fake_build_batch0_selected_candidates
    )


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
        campaign_slug="secg_ethanol_rf_sfxi_topn",
        selection_memberships=(_membership("ethanol", 1),),
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


def test_manifest_fails_fast_on_unexpected_restriction_site_in_assembled_insert() -> None:
    candidate = SelectedCandidate(
        campaign_slug="secg_and_rf_sfxi_topn",
        selection_memberships=(_membership("and", 1),),
        as_of_round=0,
        run_id="run-and-r0",
        selection_rank=1,
        id="opal-candidate-extra-ecori",
        sequence=CORE_ECORI_LEFT_JUNCTION,
        synthesis_name="SECG-B0-AND-99",
    )

    with pytest.raises(ValueError, match="unexpected restriction site"):
        build_synthesis_manifest(
            selected=[candidate],
            strategy=_strategy(),
            batch_id="stress-opal-batch0-sfxi-v1",
        )


def test_strategy_yaml_declares_expected_stress_promoter_restriction_site_policy() -> None:
    strategy = _strategy()

    assert strategy.strategy_id == "stress_promoter_insert:v1"
    assert [(site.enzyme, site.motif, site.allowed_regions) for site in strategy.restriction_sites] == [
        ("BamHI", "GGATCC", ("left_flank",)),
        ("EcoRI", "GAATTC", ("right_flank",)),
    ]


def test_manifest_rejects_duplicate_candidate_ids_aliases_and_sequences() -> None:
    duplicate_id = _selected_candidates()
    duplicate_id[1] = SelectedCandidate(
        campaign_slug="secg_cipro_rf_sfxi_topn",
        selection_memberships=(_membership("ciprofloxacin", 2),),
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
        campaign_slug="secg_cipro_rf_sfxi_topn",
        selection_memberships=(_membership("ciprofloxacin", 2),),
        as_of_round=0,
        run_id="run-cipro-r0",
        selection_rank=2,
        id="opal-candidate-b",
        sequence=CORE_B,
        synthesis_name="ES-promoter-32",
    )
    with pytest.raises(ValueError, match="duplicate synthesis_name"):
        build_synthesis_manifest(selected=duplicate_alias, strategy=_strategy(), batch_id="batch")

    duplicate_sequence = _selected_candidates()
    duplicate_sequence[1] = SelectedCandidate(
        campaign_slug="secg_cipro_rf_sfxi_topn",
        selection_memberships=(_membership("ciprofloxacin", 2),),
        as_of_round=0,
        run_id="run-cipro-r0",
        selection_rank=2,
        id="opal-candidate-b",
        sequence=CORE_A,
        synthesis_name="ES-promoter-33",
    )
    with pytest.raises(ValueError, match="duplicate promoter sequence"):
        build_synthesis_manifest(selected=duplicate_sequence, strategy=_strategy(), batch_id="batch")


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


def test_azenta_workbook_render_is_byte_reproducible(tmp_path: Path) -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )
    first_path = tmp_path / "first.xlsx"
    second_path = tmp_path / "second.xlsx"

    render_azenta_workbook(manifest, first_path)
    render_azenta_workbook(manifest, second_path)

    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == hashlib.sha256(second_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(first_path) as workbook:
        core_properties = workbook.read("docProps/core.xml")
    assert core_properties.count(b"2000-01-01T00:00:00Z") == 2


def test_azenta_workbook_readback_fails_on_sequence_mismatch(tmp_path: Path) -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )
    workbook_path = tmp_path / "azenta-order.xlsx"
    broken = pd.DataFrame(
        [
            {
                "Sequence Name": "ES-promoter-32",
                "Sequence": f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}",
                "Add Protection Nt.": "",
                "5' Phosphorylation": "",
            },
            {
                "Sequence Name": "ES-promoter-33",
                "Sequence": f"{LEFT_FLANK}{CORE_A}{RIGHT_FLANK}",
                "Add Protection Nt.": "",
                "5' Phosphorylation": "",
            },
        ]
    )
    broken.to_excel(workbook_path, sheet_name="Azenta Gene Synthesis", index=False)

    with pytest.raises(ValueError, match="sequence mismatch"):
        validate_azenta_workbook(manifest, workbook_path)


@pytest.mark.parametrize("column", ["Add Protection Nt.", "5' Phosphorylation"])
def test_azenta_workbook_readback_rejects_vendor_option_drift(tmp_path: Path, column: str) -> None:
    manifest = build_synthesis_manifest(
        selected=_selected_candidates(),
        strategy=_strategy(),
        batch_id="stress-opal-r0-20260617",
    )
    workbook_path = tmp_path / "azenta-order.xlsx"
    render_azenta_workbook(manifest, workbook_path)
    drifted = read_azenta_workbook(workbook_path)
    drifted.loc[0, column] = "Yes"
    drifted.to_excel(workbook_path, sheet_name="Azenta Gene Synthesis", index=False)

    with pytest.raises(ValueError, match=column.replace(".", r"\.")):
        validate_azenta_workbook(manifest, workbook_path)


def test_batch0_review_rows_become_campaign_scoped_selected_candidates() -> None:
    review = pd.DataFrame(
        [
            {
                "campaign": "secg_ethanol_rf_sfxi_topn",
                "id": "eth-a",
                "sequence": CORE_A,
            },
            {
                "campaign": "secg_ethanol_rf_sfxi_topn",
                "id": "eth-b",
                "sequence": CORE_B,
            },
            {
                "campaign": "secg_cipro_rf_sfxi_topn",
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
            "secg_ethanol_rf_sfxi_topn",
            1,
            "eth-a",
            "SECG-B0-ETH-01",
            "batch0_pre_assay",
            "pre_assay_seed",
            0,
            None,
        ),
        (
            "secg_ethanol_rf_sfxi_topn",
            2,
            "eth-b",
            "SECG-B0-ETH-02",
            "batch0_pre_assay",
            "pre_assay_seed",
            0,
            None,
        ),
        (
            "secg_cipro_rf_sfxi_topn",
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
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
                {
                    "campaign": "secg_cipro_rf_sfxi_topn",
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
        output_owner="source_evidence",
        output_root=tmp_path,
    )

    assert list(exports["campaign_slug"]) == [
        "secg_ethanol_rf_sfxi_topn",
        "secg_cipro_rf_sfxi_topn",
    ]
    for _, row in exports.iterrows():
        manifest_path = Path(row["manifest_path"])
        workbook_path = Path(row["azenta_workbook_path"])
        assert manifest_path.exists()
        assert workbook_path.exists()
        assert manifest_path.parent == tmp_path / row["campaign_slug"]
        assert manifest_path.name == f"stress-opal-batch0-sfxi-v1__{row['campaign_slug']}__synthesis_manifest.csv"
        assert workbook_path.name == f"stress-opal-batch0-sfxi-v1__{row['campaign_slug']}__azenta_gene_synthesis.xlsx"
        assert validate_azenta_workbook(pd.read_csv(manifest_path), workbook_path)["status"] == "pass"


def test_campaign_scoped_exports_leave_no_partial_handoff_when_readback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                }
            ]
        )
    )
    manifest = build_synthesis_manifest(
        selected=selected,
        strategy=_strategy(),
        batch_id="stress-opal-batch0-sfxi-v1",
    )

    def fail_genbank_readback(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("injected GenBank readback failure")

    monkeypatch.setattr(
        synthesis_handoff_exports,
        "validate_genbank_record_set",
        fail_genbank_readback,
    )

    with pytest.raises(ValueError, match="injected GenBank readback failure"):
        render_campaign_scoped_exports(
            manifest,
            batch_id="stress-opal-batch0-sfxi-v1",
            output_owner="source_evidence",
            output_root=tmp_path,
        )

    assert not (tmp_path / "secg_ethanol_rf_sfxi_topn").exists()
    assert not (tmp_path / "handoff_index.csv").exists()
    assert not list(tmp_path.glob(".synthesis-handoff-*"))


def test_campaign_scoped_exports_write_per_sequence_genbank_files_and_feature_table(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-b",
                    "sequence": CORE_B,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")

    def densegen_detail_for(core: str) -> list[dict[str, object]]:
        return [
            {
                "part_kind": "tfbs",
                "regulator": "baeR_TTTCTSCVHNA",
                "orientation": "fwd",
                "offset": 0,
                "offset_raw": 0,
                "end": 8,
                "sequence": core[0:8],
                "tfbs_id": "tfbs-1",
                "motif_id": "motif-1",
                "score_relative_to_theoretical_max": 0.8123,
                "tier": 1,
            },
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "variant_id": "f",
                "spacer_length": 17,
                "offset": 20,
                "offset_raw": 20,
                "end": 26,
                "sequence": core[20:26],
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "variant_id": "consensus",
                "spacer_length": 17,
                "offset": 43,
                "offset_raw": 43,
                "end": 49,
                "sequence": core[43:49],
            },
        ]

    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": CORE_A,
                "densegen__used_tfbs_detail": json.dumps(densegen_detail_for(CORE_A)),
            },
            {
                "id": "eth-b",
                "sequence": CORE_B,
                "densegen__used_tfbs_detail": json.dumps(densegen_detail_for(CORE_B)),
            },
        ]
    ).to_parquet(candidate_records_path, index=False)

    exports = render_campaign_scoped_exports(
        manifest,
        batch_id="stress-opal-batch0-sfxi-v1",
        output_owner="source_evidence",
        output_root=tmp_path,
        candidate_records_path=candidate_records_path,
    )

    row = exports.iloc[0]
    campaign_slug = row["campaign_slug"]
    genbank_dir_path = Path(row["genbank_dir_path"])
    feature_table_path = Path(row["genbank_feature_table_path"])
    assert genbank_dir_path.name == f"stress-opal-batch0-sfxi-v1__{campaign_slug}__genbank_inserts"
    genbank_files = sorted(genbank_dir_path.glob("*.gb"))
    assert len(genbank_files) == 2
    assert [path.name for path in genbank_files] == [
        f"stress-opal-batch0-sfxi-v1__{campaign_slug}__SECG-B0-ETH-01__annotated_insert.gb",
        f"stress-opal-batch0-sfxi-v1__{campaign_slug}__SECG-B0-ETH-02__annotated_insert.gb",
    ]
    assert not (genbank_dir_path.parent / f"stress-opal-batch0-sfxi-v1__{campaign_slug}__annotated_inserts.gb").exists()
    assert feature_table_path.name == f"stress-opal-batch0-sfxi-v1__{campaign_slug}__genbank_features.csv"
    assert row["genbank_validation_status"] == "pass"
    assert validate_genbank_record_set(manifest, genbank_dir_path, feature_table=feature_table_path)["status"] == "pass"

    records = list(SeqIO.parse(genbank_files[0], "genbank"))
    assert len(records) == 1
    record = records[0]
    assert str(record.seq).upper() == manifest.loc[0, "final_sequence"].upper()
    labels = {value for feature in record.features for value in feature.qualifiers.get("label", [])}
    assert {"5' cloning flank", "60 nt promoter core", "3' cloning flank", "BaeR TFBS"}.issubset(labels)
    assert "-35 (f)" in labels
    assert "-10 (consensus)" in labels
    sigma35_feature = next(feature for feature in record.features if feature.qualifiers.get("label") == ["-35 (f)"])
    assert sigma35_feature.qualifiers["sigma35_variant"] == ["f"]
    assert sigma35_feature.qualifiers["sigma35_sequence"] == [CORE_A[20:26]]
    source_qualifiers = record.features[0].qualifiers
    assert source_qualifiers["campaign_slug"] == ["secg_ethanol_rf_sfxi_topn"]
    assert source_qualifiers["batch_id"] == ["stress-opal-batch0-sfxi-v1"]
    assert source_qualifiers["synthesis_name"] == ["SECG-B0-ETH-01"]
    assert source_qualifiers["selection_views"] == ["ethanol"]
    assert source_qualifiers["selection_membership"] == ["view=ethanol|rank=1"]
    bae_r_feature = next(feature for feature in record.features if feature.qualifiers.get("label") == ["BaeR TFBS"])
    assert bae_r_feature.qualifiers["dg_offset"] == ["0"]
    assert bae_r_feature.qualifiers["dg_offset_raw"] == ["0"]

    feature_rows = pd.read_csv(feature_table_path)
    assert set(feature_rows["label"]) >= {
        "5' cloning flank",
        "60 nt promoter core",
        "3' cloning flank",
        "BaeR TFBS",
    }
    assert "-35 (f)" in set(feature_rows["label"])
    assert "-10 (consensus)" in set(feature_rows["label"])
    assert set(feature_rows.loc[feature_rows["label"] == "BaeR TFBS", "densegen_coordinate_key"]) == {"offset"}
    fixed_mask = feature_rows["label"].astype(str).str.startswith(("-35", "-10"))
    assert set(feature_rows.loc[fixed_mask, "densegen_coordinate_key"]) == {"offset_raw"}


def test_genbank_validation_requires_membership_aware_manifest(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                }
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    render_genbank_record_set(manifest, feature_table, genbank_dir)
    old_schema_manifest = manifest.drop(columns=["selection_view_ids", "selection_memberships"])

    with pytest.raises(
        ValueError,
        match="missing required GenBank columns: selection_view_ids, selection_memberships",
    ):
        validate_genbank_record_set(old_schema_manifest, genbank_dir)


def test_genbank_validation_rejects_membership_qualifier_drift(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                }
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    rendered = render_genbank_record_set(manifest, feature_table, genbank_dir)
    record_path = Path(rendered.iloc[0]["genbank_file_path"])
    record = SeqIO.read(record_path, "genbank")
    record.features[0].qualifiers["selection_views"] = ["and"]
    SeqIO.write(record, record_path, "genbank")

    with pytest.raises(ValueError, match="selection_views mismatch"):
        validate_genbank_record_set(manifest, genbank_dir)


def test_genbank_validation_rejects_unexpected_directory_entries(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    render_genbank_record_set(manifest, feature_table, genbank_dir)
    (genbank_dir / "notes.txt").write_text("not part of the handoff\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected GenBank directory entries: notes.txt"):
        validate_genbank_record_set(manifest, genbank_dir)


def test_genbank_validation_rejects_symlinked_record(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    rendered = render_genbank_record_set(manifest, feature_table, genbank_dir)
    record_path = Path(rendered.iloc[0]["genbank_file_path"])
    outside_record = tmp_path / "outside.gb"
    record_path.replace(outside_record)
    record_path.symlink_to(outside_record)

    with pytest.raises(ValueError, match="GenBank directory entries must be regular files, not symlinks"):
        validate_genbank_record_set(manifest, genbank_dir)

    with pytest.raises(ValueError, match="GenBank directory entries must be regular files, not symlinks"):
        synthesis_handoff_records._sha256_genbank_dir(genbank_dir)


def test_genbank_validation_rejects_record_identity_drift(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    rendered = render_genbank_record_set(manifest, feature_table, genbank_dir)
    record_path = Path(rendered.iloc[0]["genbank_file_path"])
    record = SeqIO.read(record_path, "genbank")
    record.id = "WRONG"
    record.name = "WRONG"
    SeqIO.write(record, record_path, "genbank")

    with pytest.raises(ValueError, match="record identity mismatch"):
        validate_genbank_record_set(manifest, genbank_dir, feature_table=feature_table)


def test_genbank_validation_rejects_feature_table_parity_drift(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    render_genbank_record_set(manifest, feature_table, genbank_dir)
    drifted = feature_table.copy()
    drifted.loc[drifted["label"].eq("60 nt promoter core"), "start_0"] += 1

    with pytest.raises(ValueError, match="feature location mismatch"):
        validate_genbank_record_set(manifest, genbank_dir, feature_table=drifted)


def test_genbank_validation_rejects_extra_duplicate_or_identity_drifted_feature_rows(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    render_genbank_record_set(manifest, feature_table, genbank_dir)

    extra_candidate = feature_table.iloc[[0]].copy()
    extra_candidate["id"] = "unexpected-candidate"
    with pytest.raises(ValueError, match="candidate ID parity mismatch"):
        validate_genbank_record_set(
            manifest,
            genbank_dir,
            feature_table=pd.concat([feature_table, extra_candidate], ignore_index=True),
        )

    with pytest.raises(ValueError, match="duplicate candidate and feature IDs"):
        validate_genbank_record_set(
            manifest,
            genbank_dir,
            feature_table=pd.concat([feature_table, feature_table.iloc[[0]]], ignore_index=True),
        )

    for column in ("batch_id", "campaign_slug", "synthesis_name"):
        drifted = feature_table.copy()
        drifted.loc[0, column] = "wrong"
        with pytest.raises(ValueError, match=f"feature table {column} mismatch"):
            validate_genbank_record_set(manifest, genbank_dir, feature_table=drifted)


def test_genbank_validation_reports_missing_feature_table_as_validation_failure(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame([{"campaign": "secg_ethanol_rf_sfxi_topn", "id": "eth-a", "sequence": CORE_A}])
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    feature_table = build_genbank_feature_table(manifest)
    genbank_dir = tmp_path / "genbank"
    render_genbank_record_set(manifest, feature_table, genbank_dir)

    with pytest.raises(ValueError, match="feature table not found"):
        validate_genbank_record_set(manifest, genbank_dir, feature_table=tmp_path / "missing.csv")


def test_genbank_feature_projection_fails_fast_on_densegen_coordinate_mismatch(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": CORE_A,
                "densegen__used_tfbs_detail": json.dumps(
                    [
                        {
                            "part_kind": "tfbs",
                            "regulator": "baeR_TTTCTSCVHNA",
                            "orientation": "fwd",
                            "offset": 4,
                            "end": 12,
                            "sequence": "TTTTTTTT",
                        }
                    ]
                ),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    with pytest.raises(ValueError, match="DenseGen annotation sequence mismatch"):
        build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)


@pytest.mark.parametrize(
    ("detail", "error_match"),
    [
        (["not-a-mapping"], "DenseGen annotation detail item 0 must be a mapping"),
        ([{"part_kind": "future_element"}], "unsupported DenseGen annotation part_kind.*future_element"),
        (
            [{"part_kind": "fixed_element", "constraint_name": "future_constraint"}],
            "unsupported DenseGen fixed_element constraint.*future_constraint",
        ),
        (
            [{"part_kind": "fixed_element", "constraint_name": "sigma70_core", "role": "upstrem"}],
            "unsupported DenseGen sigma70_core role.*upstrem",
        ),
        ([{"part_kind": "tfbs"}], "DenseGen TFBS annotation.*requires a regulator"),
        (
            [
                {
                    "part_kind": "tfbs",
                    "regulator": "baeR",
                    "orientation": "fwd",
                    "offset": 0.5,
                    "end": 8,
                    "sequence": CORE_A[:8],
                }
            ],
            "DenseGen annotation offset for eth-a must be a non-negative integer",
        ),
        (
            [
                {
                    "part_kind": "tfbs",
                    "regulator": "baeR",
                    "orientation": "fwd",
                    "offset": 0,
                    "end": 8.5,
                    "sequence": CORE_A[:8],
                }
            ],
            "DenseGen annotation end for eth-a must be a non-negative integer",
        ),
    ],
)
def test_genbank_feature_projection_rejects_unrepresented_densegen_annotations(
    tmp_path: Path,
    detail: list[object],
    error_match: str,
) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(
        selected=selected,
        strategy=_strategy(),
        batch_id="stress-opal-batch0-sfxi-v1",
    )
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": CORE_A,
                "densegen__used_tfbs_detail": json.dumps(detail),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    with pytest.raises(ValueError, match=error_match):
        build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)


def test_genbank_feature_projection_rejects_tfbs_offset_raw_fallback(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": CORE_A,
                "densegen__used_tfbs_detail": json.dumps(
                    [
                        {
                            "part_kind": "tfbs",
                            "regulator": "baeR_TTTCTSCVHNA",
                            "orientation": "fwd",
                            "offset": 2,
                            "offset_raw": 4,
                            "end": 10,
                            "sequence": CORE_A[4:12],
                        }
                    ]
                ),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    with pytest.raises(ValueError, match="required coordinate key offset"):
        build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)


def test_genbank_feature_projection_requires_fixed_element_offset_raw(tmp_path: Path) -> None:
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": CORE_A,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": CORE_A,
                "densegen__used_tfbs_detail": json.dumps(
                    [
                        {
                            "part_kind": "fixed_element",
                            "role": "upstream",
                            "constraint_name": "sigma70_core",
                            "variant_id": "f",
                            "spacer_length": 17,
                            "offset": 20,
                            "end": 26,
                            "sequence": CORE_A[20:26],
                        }
                    ]
                ),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    with pytest.raises(ValueError, match="requires coordinate key offset_raw"):
        build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)


def test_genbank_feature_projection_records_fixed_element_offset_raw_source(tmp_path: Path) -> None:
    core = CORE_A
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-a",
                    "sequence": core,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-a",
                "sequence": core,
                "densegen__used_tfbs_detail": json.dumps(
                    [
                        {
                            "part_kind": "fixed_element",
                            "role": "upstream",
                            "constraint_name": "sigma70_core",
                            "variant_id": "f",
                            "spacer_length": 17,
                            "offset": 22,
                            "offset_raw": 20,
                            "end": 28,
                            "sequence": core[20:26],
                        }
                    ]
                ),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    features = build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)

    fixed = features.loc[features["label"].astype(str).str.startswith("-35")].iloc[0]
    assert fixed["label"] == "-35 (f)"
    assert fixed["variant_id"] == "f"
    assert fixed["densegen_coordinate_key"] == "offset_raw"
    assert fixed["start_0"] == len(LEFT_FLANK) + 20
    assert fixed["end_0"] == len(LEFT_FLANK) + 26


def test_genbank_feature_projection_accepts_reverse_orientation_motif_sequences(tmp_path: Path) -> None:
    core = "AACGTTTTTCAGCCTTACCGCAGAATAGTTAGACAAATCTCTGCAGAATTTTAATATAAT"
    selected = selected_candidates_from_batch0_review(
        pd.DataFrame(
            [
                {
                    "campaign": "secg_ethanol_rf_sfxi_topn",
                    "id": "eth-rev",
                    "sequence": core,
                },
            ]
        )
    )
    manifest = build_synthesis_manifest(selected=selected, strategy=_strategy(), batch_id="stress-opal-batch0-sfxi-v1")
    candidate_records_path = tmp_path / "candidate_records.parquet"
    pd.DataFrame(
        [
            {
                "id": "eth-rev",
                "sequence": core,
                "densegen__used_tfbs_detail": json.dumps(
                    [
                        {
                            "part_kind": "tfbs",
                            "regulator": "baeR_TTTCTSCVHNA",
                            "orientation": "rev",
                            "offset": 14,
                            "end": 30,
                            "sequence": "AACTATTCTGCGGTAA",
                        }
                    ]
                ),
            }
        ]
    ).to_parquet(candidate_records_path, index=False)

    features = build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)

    tfbs = features.loc[features["label"] == "BaeR TFBS"].iloc[0]
    assert tfbs["strand"] == -1
    assert tfbs["sequence"] == core[14:30]
    assert tfbs["genbank_location"] == f"complement({len(LEFT_FLANK) + 14 + 1}..{len(LEFT_FLANK) + 30})"


def test_batch0_synthesis_output_dir_is_study_source_evidence_local() -> None:
    path = source_evidence_synthesis_output_dir(
        Path("/repo"),
        campaign_slug="secg_and_rf_sfxi_topn",
        batch_id="stress-opal-batch0-sfxi-v1",
    )

    assert path == Path(
        "/repo/src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/"
        "opal_sfxi_round0/secg_and_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1"
    )


def test_measured_round_synthesis_output_dir_is_campaign_local() -> None:
    path = campaign_synthesis_output_dir(
        Path("/repo"),
        campaign_slug="secg_msrb_greedy",
        batch_id="stress-opal-r1-msrb-v1",
    )

    assert path == Path(
        "/repo/src/dnadesign/opal/campaigns/secg_msrb_greedy/outputs/synthesis_handoff/stress-opal-r1-msrb-v1"
    )


@pytest.mark.parametrize(
    ("campaign_slug", "batch_id"),
    [
        ("../outside", "stress-opal-batch1-msrb-v1"),
        ("secg_msrb_greedy", "../outside"),
    ],
)
def test_campaign_synthesis_output_dir_rejects_path_traversal(campaign_slug: str, batch_id: str) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        campaign_synthesis_output_dir(
            Path("/repo"),
            campaign_slug=campaign_slug,
            batch_id=batch_id,
        )


def _selection_artifact_row(
    *,
    as_of_round: int,
    campaign_slug: str,
    selection_view_id: str,
    candidate_id: str,
    sequence: str,
    rank: int,
    score: float,
) -> dict[str, Any]:
    return {
        "as_of_round": as_of_round,
        "campaign_slug": campaign_slug,
        "selection_view_id": selection_view_id,
        "id": candidate_id,
        "sequence": sequence,
        "selection_batch_key": candidate_id,
        "deduplicate_by": "id",
        "rank_competition": rank,
        "rank_ordinal": rank,
        "score": score,
        "selection_score": score,
        "score_ref": "behavior_score",
        "allocation_slot": None,
        "selection_origin": "preferred_top_k",
    }


def _selection_batch_row(
    *,
    as_of_round: int,
    campaign_slug: str,
    candidate_id: str,
    memberships: list[tuple[str, int, float]],
) -> dict[str, Any]:
    view_ids = [view_id for view_id, _, _ in memberships]
    return {
        "as_of_round": as_of_round,
        "campaign_slug": campaign_slug,
        "id": candidate_id,
        "selection_batch_key": candidate_id,
        "deduplicate_by": "id",
        "selection_view_ids": view_ids,
        "selection_memberships": [
            {
                "selection_view_id": view_id,
                "rank": rank,
                "rank_ordinal": rank,
                "score": score,
                "selection_score": score,
                "score_ref": "behavior_score",
                "allocation_slot": None,
                "selection_origin": "preferred_top_k",
            }
            for view_id, rank, score in memberships
        ],
        "preferred_view_ids": view_ids,
        "allocation_view_id": None,
        "allocation_slot": None,
    }


def _write_opal_round_fixture(
    tmp_path: Path,
    *,
    slug: str = "secg_msrb_greedy",
    as_of_round: int = 1,
    run_ids: tuple[str, ...] = ("run-msrb-r1",),
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
            schema_version: opal.campaign.v3
            ownership:
              owner_scope: study_campaign
              study_id: stress_ethanol_cipro_growth
              dataset_id: test_candidates
              portable: false
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
              name: vector_from_table_v1
              params:
                id_column: id
                value_columns: ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]
            model:
              name: random_forest
              params:
                n_estimators: 10
                random_state: 7
                n_jobs: 1
            selection_views:
              - id: ethanol
                objective:
                  name: multistate_response_behavior_v1
                  params:
                    state_ids: ["00", "10", "01", "11"]
                    target_mask: [0, 1, 0, 1]
                    softmin_scale: 1.0
                selection:
                  name: top_n
                  params:
                    top_k: 2
                    score_ref: behavior_score
                    tie_handling: ordinal
                    objective_mode: maximize
              - id: ciprofloxacin
                objective:
                  name: multistate_response_behavior_v1
                  params:
                    state_ids: ["00", "10", "01", "11"]
                    target_mask: [0, 0, 1, 1]
                    softmin_scale: 1.0
                selection:
                  name: top_n
                  params:
                    top_k: 2
                    score_ref: behavior_score
                    tie_handling: ordinal
                    objective_mode: maximize
            selection_batch:
              deduplicate_by: id
            """
        ).format(slug=slug),
        encoding="utf-8",
    )

    candidate_a = f"{candidate_id_prefix}candidate-a"
    candidate_b = f"{candidate_id_prefix}candidate-b"
    candidate_c = f"{candidate_id_prefix}candidate-c"
    record_sequences = record_sequences or {candidate_a: CORE_A, candidate_b: CORE_B, candidate_c: "GATC" * 15}
    records = pd.DataFrame(
        [
            {
                "id": candidate_id,
                "sequence": sequence,
                "densegen__used_tfbs_detail": [],
            }
            for candidate_id, sequence in record_sequences.items()
        ]
    )
    records.to_parquet(campaign_root / "records.parquet", index=False)

    ledger_sequences = ledger_sequences or record_sequences
    ledger_root = campaign_root / "outputs" / "ledger"
    predictions_dir = ledger_root / "predictions"
    predictions_dir.mkdir(parents=True)
    runs_dir = ledger_root / "runs.parquet"
    runs_dir.mkdir(parents=True)

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
                    "pred__selection_views": [
                        {
                            "selection_view_id": "ethanol",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.8,
                            "selection_score": 0.8,
                            "rank_competition": 2,
                            "is_selected": True,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                        {
                            "selection_view_id": "ciprofloxacin",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.1,
                            "selection_score": 0.1,
                            "rank_competition": 3,
                            "is_selected": False,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                    ],
                },
                {
                    "event": "run_pred",
                    "run_id": run_id,
                    "as_of_round": as_of_round,
                    "id": candidate_b,
                    "sequence": ledger_sequences[candidate_b],
                    "pred__selection_views": [
                        {
                            "selection_view_id": "ethanol",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.9,
                            "selection_score": 0.9,
                            "rank_competition": 1,
                            "is_selected": True,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                        {
                            "selection_view_id": "ciprofloxacin",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.85,
                            "selection_score": 0.85,
                            "rank_competition": 2,
                            "is_selected": True,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                    ],
                },
                {
                    "event": "run_pred",
                    "run_id": run_id,
                    "as_of_round": as_of_round,
                    "id": candidate_c,
                    "sequence": ledger_sequences[candidate_c],
                    "pred__selection_views": [
                        {
                            "selection_view_id": "ethanol",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.2,
                            "selection_score": 0.2,
                            "rank_competition": 3,
                            "is_selected": False,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                        {
                            "selection_view_id": "ciprofloxacin",
                            "objective_name": "multistate_response_behavior_v1",
                            "selection_name": "top_n",
                            "score_ref": "behavior_score",
                            "score": 0.95,
                            "selection_score": 0.95,
                            "rank_competition": 1,
                            "is_selected": True,
                            "top_k": 2,
                            "uncertainty": None,
                            "uncertainty_ref": None,
                            "diagnostics": None,
                        },
                    ],
                },
            ]
        )
    pd.DataFrame(prediction_rows).to_parquet(predictions_dir / "part-predictions.parquet", index=False)
    selection_rows = [
        _selection_artifact_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            selection_view_id="ethanol",
            candidate_id=candidate_b,
            sequence=ledger_sequences[candidate_b],
            rank=1,
            score=0.9,
        ),
        _selection_artifact_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            selection_view_id="ethanol",
            candidate_id=candidate_a,
            sequence=ledger_sequences[candidate_a],
            rank=2,
            score=0.8,
        ),
        _selection_artifact_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            selection_view_id="ciprofloxacin",
            candidate_id=candidate_c,
            sequence=ledger_sequences[candidate_c],
            rank=1,
            score=0.95,
        ),
        _selection_artifact_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            selection_view_id="ciprofloxacin",
            candidate_id=candidate_b,
            sequence=ledger_sequences[candidate_b],
            rank=2,
            score=0.85,
        ),
    ]
    batch_rows = [
        _selection_batch_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            candidate_id=candidate_a,
            memberships=[("ethanol", 2, 0.8)],
        ),
        _selection_batch_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            candidate_id=candidate_b,
            memberships=[("ethanol", 1, 0.9), ("ciprofloxacin", 2, 0.85)],
        ),
        _selection_batch_row(
            as_of_round=as_of_round,
            campaign_slug=slug,
            candidate_id=candidate_c,
            memberships=[("ciprofloxacin", 1, 0.95)],
        ),
    ]
    run_rows = []
    for run_id in run_ids:
        selection_dir = campaign_root / "outputs" / "runs" / run_id / "selection"
        selection_dir.mkdir(parents=True)
        selection_path = selection_dir / "selections.parquet"
        selection_batch_path = selection_dir / "selection_batch.parquet"
        pd.DataFrame([{**row, "run_id": run_id} for row in selection_rows]).to_parquet(selection_path, index=False)
        pd.DataFrame([{**row, "run_id": run_id} for row in batch_rows]).to_parquet(
            selection_batch_path,
            index=False,
        )
        run_rows.append(
            {
                "event": "run_meta",
                "run_id": run_id,
                "as_of_round": as_of_round,
                "artifacts": {
                    "selection/selections.parquet": [
                        hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                        str(selection_path),
                    ],
                    "selection/selection_batch.parquet": [
                        hashlib.sha256(selection_batch_path.read_bytes()).hexdigest(),
                        str(selection_batch_path),
                    ],
                },
            }
        )
    pd.DataFrame(run_rows).to_parquet(runs_dir / "part-runs.parquet", index=False)
    return config_path


def _write_opal_alias_registry(repo_root: Path, config_path: Path) -> Path:
    records_path = config_path.parent.parent / "records.parquet"
    records = pd.read_parquet(records_path, columns=["id", "sequence"])
    by_suffix = {
        str(row.id).rsplit("candidate-", 1)[-1]: (str(row.id), str(row.sequence))
        for row in records.itertuples(index=False)
    }
    assignments = []
    for ordinal, suffix in enumerate(("b", "a", "c"), start=1):
        candidate_id, sequence = by_suffix[suffix]
        assignments.append(
            {
                "ordinal": ordinal,
                "alias": f"SECG-{ordinal:03d}",
                "candidate_id": candidate_id,
                "sequence_sha256": hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest(),
                "first_assignment": {
                    "source_authority": "opal_selection_batch",
                    "source_id": "run-msrb-r1",
                    "nomination_batch_index": 2,
                    "model_as_of_round": 1,
                },
                "source_aliases": [],
            }
        )
    registry_path = repo_root / synthesis_handoff_cli.PROMOTER_ALIAS_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "dnadesign.study.promoter_alias_registry.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "alias_namespace": "study.promoter_alias",
                "format": {"prefix": "SECG", "zero_pad_width": 3},
                "candidate_table": {
                    "dataset_id": "fixture-candidates",
                    "records_path": records_path.relative_to(repo_root).as_posix(),
                },
                "assignments": assignments,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry_path


def test_opal_round_selection_batch_becomes_one_campaign_handoff(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(tmp_path)
    alias_registry_path = _write_opal_alias_registry(tmp_path, config_path)

    selected, report = selected_candidates_from_opal_round(
        config_path,
        as_of_round=1,
        repo_root=tmp_path,
        alias_registry_path=alias_registry_path,
    )

    assert [
        (
            row.campaign_slug,
            row.as_of_round,
            row.run_id,
            row.selection_rank,
            row.id,
            row.synthesis_name,
            row.selection_view_ids,
        )
        for row in selected
    ] == [
        ("secg_msrb_greedy", 1, "run-msrb-r1", 1, "candidate-b", "SECG-001", ("ethanol", "ciprofloxacin")),
        ("secg_msrb_greedy", 1, "run-msrb-r1", 2, "candidate-a", "SECG-002", ("ethanol",)),
        ("secg_msrb_greedy", 1, "run-msrb-r1", 3, "candidate-c", "SECG-003", ("ciprofloxacin",)),
    ]
    assert {row.selection_source for row in selected} == {"opal_selection_batch"}
    assert {row.selection_epoch for row in selected} == {"opal_model_round"}
    assert {row.model_as_of_round for row in selected} == {1}
    assert {row.assay_batch_index for row in selected} == {None}
    assert report["row_count"] == 3
    assert report["unique_candidate_count"] == 3
    assert report["unique_sequence_count"] == 3
    assert report["unique_study_alias_count"] == 3
    assert report["study_aliases"] == ["SECG-001", "SECG-002", "SECG-003"]
    assert report["replay_mismatch_count"] == 0
    assert report["selection_view_counts"] == {"ethanol": 2, "ciprofloxacin": 2}


def test_opal_round_source_rejects_failed_selection_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_opal_round_fixture(tmp_path)
    alias_registry_path = _write_opal_alias_registry(tmp_path, config_path)
    load_selection_set = synthesis_handoff_opal_round_source.load_selection_set

    def load_selection_set_with_failed_replay(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = load_selection_set(*args, **kwargs)
        if kwargs["selection_view_id"] == "ethanol":
            payload = {
                **payload,
                "verification": {
                    **payload["verification"],
                    "status": "fail",
                    "mismatch_count": 1,
                },
            }
        return payload

    monkeypatch.setattr(
        synthesis_handoff_opal_round_source,
        "load_selection_set",
        load_selection_set_with_failed_replay,
    )

    with pytest.raises(ValueError, match="selection replay verification failed.*ethanol"):
        selected_candidates_from_opal_round(
            config_path,
            as_of_round=1,
            repo_root=tmp_path,
            alias_registry_path=alias_registry_path,
        )


def test_opal_round_source_rejects_ambiguous_reruns_without_run_id(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(tmp_path, run_ids=("run-a", "run-b"))

    with pytest.raises(ValueError, match="Multiple run_id values"):
        selected_candidates_from_opal_round(config_path, as_of_round=1)


def test_opal_round_source_rejects_sequence_mismatch_against_records(tmp_path: Path) -> None:
    config_path = _write_opal_round_fixture(
        tmp_path,
        ledger_sequences={"candidate-a": CORE_B, "candidate-b": CORE_B, "candidate-c": "GATC" * 15},
    )

    with pytest.raises(ValueError, match="sequence mismatch"):
        selected_candidates_from_opal_round(
            config_path,
            as_of_round=1,
            run_id="run-msrb-r1",
        )


def test_cli_rejects_raw_opal_round_write_without_checked_in_handoff_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_opal_round_fixture(tmp_path)
    alias_registry_path = _write_opal_alias_registry(tmp_path, config_path)
    output_root = tmp_path / "exports"

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--source",
                "opal-round",
                "--round",
                "1",
                "--campaign-config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--promoter-alias-registry",
                str(alias_registry_path),
                "--output-dir",
                str(output_root),
                "--write",
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "requires --handoff-id" in payload["error"]["message"]
    assert not output_root.exists()


def test_cli_rejects_selected_csv_write_without_lifecycle_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "exports"

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--source",
                "selected-csv",
                "--selected-csv",
                str(tmp_path / "untrusted.csv"),
                "--output-dir",
                str(output_root),
                "--write",
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "selected-csv is preview-only" in payload["error"]["message"]
    assert not output_root.exists()


def test_cli_resolves_batch0_from_checked_in_handoff_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_batch0_cli_source(monkeypatch)

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
    assert payload["handoff_record"]["artifact_status"]["summary"]["expected_artifact_count"] == 3
    expected_artifacts = {row["campaign_slug"]: row for row in payload["handoff_record"]["expected_artifacts"]}
    assert set(expected_artifacts) == {
        "secg_ethanol_rf_sfxi_topn",
        "secg_cipro_rf_sfxi_topn",
        "secg_and_rf_sfxi_topn",
    }
    assert expected_artifacts["secg_ethanol_rf_sfxi_topn"]["manifest_path"].endswith(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/"
        "opal_sfxi_round0/secg_ethanol_rf_sfxi_topn/outputs/"
        "synthesis_handoff/stress-opal-batch0-sfxi-v1/"
        "stress-opal-batch0-sfxi-v1__secg_ethanol_rf_sfxi_topn__synthesis_manifest.csv"
    )
    assert expected_artifacts["secg_ethanol_rf_sfxi_topn"]["vendor_workbook_path"].endswith(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/"
        "opal_sfxi_round0/secg_ethanol_rf_sfxi_topn/outputs/"
        "synthesis_handoff/stress-opal-batch0-sfxi-v1/"
        "stress-opal-batch0-sfxi-v1__secg_ethanol_rf_sfxi_topn__azenta_gene_synthesis.xlsx"
    )
    assert expected_artifacts["secg_ethanol_rf_sfxi_topn"]["genbank_dir_path"].endswith(
        "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/"
        "opal_sfxi_round0/secg_ethanol_rf_sfxi_topn/outputs/"
        "synthesis_handoff/stress-opal-batch0-sfxi-v1/"
        "stress-opal-batch0-sfxi-v1__secg_ethanol_rf_sfxi_topn__genbank_inserts"
    )


def test_checked_in_batch0_record_accepts_current_strategy_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_batch0_cli_source(monkeypatch)

    exit_code = synthesis_handoff_main(
        [
            "--handoff-id",
            "stress-opal-batch0-sfxi-v1",
            "--source",
            "batch0",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_id"] == "stress_promoter_insert:v1"
    assert payload["handoff_record"]["manifest_validation"] == {
        "campaign_counts": {
            "secg_and_rf_sfxi_topn": 6,
            "secg_cipro_rf_sfxi_topn": 6,
            "secg_ethanol_rf_sfxi_topn": 6,
        },
        "status": "pass",
        "strategy_id": "stress_promoter_insert:v1",
    }


def test_cli_handoff_record_preview_does_not_rebuild_batch0_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_batch0_is_rebuilt(*, config_path: Path, repo_root: Path | None = None):
        raise ValueError("batch0 selected candidates should not be rebuilt for record preview")

    monkeypatch.setattr(synthesis_handoff_cli, "build_batch0_selected_candidates", fail_if_batch0_is_rebuilt)

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
    assert payload["mode"] == "handoff_record_preview"
    assert payload["source"] == "batch0"
    assert payload["batch_id"] == "stress-opal-batch0-sfxi-v1"
    assert payload["handoff_record"]["handoff_id"] == "stress-opal-batch0-sfxi-v1"
    assert payload["handoff_record"]["artifact_status"]["summary"]["expected_artifact_count"] == 3


def test_cli_handoff_record_resolves_one_campaign_run_and_view_memberships(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(synthesis_handoff_cli, "_source_checkout_repo_root", lambda: tmp_path.resolve())
    config_path = _write_opal_round_fixture(
        tmp_path / "src/dnadesign/opal/campaigns",
        as_of_round=1,
        run_ids=("msrb-run-a", "msrb-run-b"),
    )
    alias_registry_path = _write_opal_alias_registry(tmp_path, config_path)
    record_path = tmp_path / synthesis_handoff_records.DEFAULT_SYNTHESIS_HANDOFF_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path = tmp_path / synthesis_handoff_cli.DEFAULT_STRESS_PROMOTER_CLONING_STRATEGY
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/")
        / "synthesis_handoff/configs/stress_promoter_insert_v1.yaml",
        strategy_path,
    )
    selection_batch_path = config_path.parent.parent / "outputs/runs/msrb-run-b/selection/selection_batch.parquet"
    candidate_records_path = config_path.parent.parent / "records.parquet"
    record_raw = {
        "version": 3,
        "study_id": "stress_ethanol_cipro_growth",
        "record_kind": "synthesis_handoff_lifecycle",
        "handoffs": [
            {
                "handoff_id": "stress-opal-r1-msrb-v1",
                "lifecycle_status": "generated_pending_acceptance",
                "source_authority": "opal_selection_batch",
                "selection_epoch": "opal_model_round",
                "assay_batch_index": 1,
                "model_as_of_round": 1,
                "campaign_slug": "secg_msrb_greedy",
                "run_id": "msrb-run-b",
                "strategy_id": "stress_promoter_insert:v1",
                "expected_selection_views": [
                    {"selection_view_id": "ethanol", "expected_rows": 2},
                    {"selection_view_id": "ciprofloxacin", "expected_rows": 2},
                ],
                "materialization_contract": _materialization_contract(
                    ("SECG-001", "candidate-b", hashlib.sha256(CORE_B.encode("utf-8")).hexdigest()),
                    ("SECG-002", "candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
                    ("SECG-003", "candidate-c", hashlib.sha256(("GATC" * 15).encode("utf-8")).hexdigest()),
                    campaign_config=(
                        config_path.relative_to(tmp_path).as_posix(),
                        hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    ),
                    selection_batch=(
                        selection_batch_path.relative_to(tmp_path).as_posix(),
                        hashlib.sha256(selection_batch_path.read_bytes()).hexdigest(),
                    ),
                    candidate_records=(
                        candidate_records_path.relative_to(tmp_path).as_posix(),
                        hashlib.sha256(candidate_records_path.read_bytes()).hexdigest(),
                    ),
                    promoter_alias_registry=(
                        alias_registry_path.relative_to(tmp_path).as_posix(),
                        hashlib.sha256(alias_registry_path.read_bytes()).hexdigest(),
                    ),
                    cloning_strategy=(
                        strategy_path.relative_to(tmp_path).as_posix(),
                        hashlib.sha256(strategy_path.read_bytes()).hexdigest(),
                    ),
                ).to_json(),
                "expected_artifact": {
                    "campaign_slug": "secg_msrb_greedy",
                    "expected_rows": 3,
                    "manifest_path": "out/msrb/synthesis_manifest.csv",
                    "vendor_workbook_path": "out/msrb/azenta_gene_synthesis.xlsx",
                    "genbank_dir_path": "out/msrb/genbank_inserts",
                    "genbank_feature_table_path": "out/msrb/genbank_features.csv",
                    "manifest_sha256": "0" * 64,
                    "vendor_workbook_sha256": "1" * 64,
                    "genbank_dir_sha256": "2" * 64,
                    "genbank_feature_table_sha256": "3" * 64,
                    "workbook_readback_status": "pass",
                    "genbank_readback_status": "pass",
                },
            }
        ],
    }
    record_path.write_text(yaml.safe_dump(record_raw, sort_keys=False), encoding="utf-8")

    exit_code = synthesis_handoff_main(
        [
            "--handoff-id",
            "stress-opal-r1-msrb-v1",
            "--record-yaml",
            str(record_path),
            "--campaign-config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--promoter-alias-registry",
            str(alias_registry_path),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "opal-round"
    assert payload["batch_id"] == "stress-opal-r1-msrb-v1"
    assert payload["handoff_record"]["manifest_validation"]["status"] == "pass"
    assert payload["handoff_record"]["assay_batch_index"] == 1
    assert payload["handoff_record"]["run_id"] == "msrb-run-b"
    assert payload["handoff_record"]["expected_study_aliases"] == ["SECG-001", "SECG-002", "SECG-003"]
    assert payload["handoff_record"]["manifest_validation"]["study_aliases"] == [
        "SECG-001",
        "SECG-002",
        "SECG-003",
    ]
    assert payload["source_report"]["campaign_slug"] == "secg_msrb_greedy"
    assert payload["source_report"]["selection_view_counts"] == {"ethanol": 2, "ciprofloxacin": 2}

    output_root = tmp_path / "generated"
    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-msrb-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--promoter-alias-registry",
                str(alias_registry_path),
                "--output-dir",
                str(output_root),
                "--write",
                "--json",
            ]
        )
    assert exc_info.value.code == 2
    error = _json_cli_error(capsys.readouterr().err)
    assert "authorized_for_materialization" in error["error"]["message"]
    assert not output_root.exists()

    record_raw = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    record_raw["handoffs"][0]["lifecycle_status"] = "authorized_for_materialization"
    record_path.write_text(yaml.safe_dump(record_raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-msrb-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--promoter-alias-registry",
                str(alias_registry_path),
                "--strategy-yaml",
                str(strategy_path),
                "--write",
                "--json",
            ]
        )
    assert exc_info.value.code == 2
    error = _json_cli_error(capsys.readouterr().err)
    assert "artifact paths" in error["error"]["message"]
    assert not campaign_synthesis_output_dir(
        tmp_path,
        campaign_slug="secg_msrb_greedy",
        batch_id="stress-opal-r1-msrb-v1",
    ).exists()

    export_dir = campaign_synthesis_output_dir(
        tmp_path,
        campaign_slug="secg_msrb_greedy",
        batch_id="stress-opal-r1-msrb-v1",
    )
    artifact_paths = campaign_synthesis_artifact_paths(
        export_dir,
        batch_id="stress-opal-r1-msrb-v1",
        campaign_slug="secg_msrb_greedy",
    )
    expected_artifact = record_raw["handoffs"][0]["expected_artifact"]
    expected_artifact["manifest_path"] = artifact_paths["manifest"].relative_to(tmp_path).as_posix()
    expected_artifact["vendor_workbook_path"] = artifact_paths["azenta_workbook"].relative_to(tmp_path).as_posix()
    expected_artifact["genbank_dir_path"] = artifact_paths["genbank_dir"].relative_to(tmp_path).as_posix()
    expected_artifact["genbank_feature_table_path"] = (
        artifact_paths["genbank_feature_table"].relative_to(tmp_path).as_posix()
    )
    record_path.write_text(yaml.safe_dump(record_raw, sort_keys=False), encoding="utf-8")

    assert (
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-msrb-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--promoter-alias-registry",
                str(alias_registry_path),
                "--strategy-yaml",
                str(strategy_path),
                "--write",
                "--json",
            ]
        )
        == 0
    )
    written = json.loads(capsys.readouterr().out)
    assert written["mode"] == "written"
    assert written["campaign_exports"][0]["row_count"] == 3
    assert all(
        artifact_paths[key].exists() for key in ("manifest", "azenta_workbook", "genbank_dir", "genbank_feature_table")
    )

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-msrb-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--promoter-alias-registry",
                str(alias_registry_path),
                "--strategy-yaml",
                str(strategy_path),
                "--write",
                "--json",
            ]
        )
    assert exc_info.value.code == 2
    error = _json_cli_error(capsys.readouterr().err)
    assert "must not already exist" in error["error"]["message"]


def test_cli_rejects_rewriting_frozen_batch0_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_batch0_cli_source(monkeypatch)
    output_root = tmp_path / "generated"

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--source",
                "batch0",
                "--output-dir",
                str(output_root),
                "--write",
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    error = _json_cli_error(capsys.readouterr().err)
    assert "frozen source evidence" in error["error"]["message"]
    assert not output_root.exists()


def test_handoff_record_lifecycle_stamps_unified_run_id_onto_selected_rows() -> None:
    source_row = SelectedCandidate(
        campaign_slug="secg_msrb_greedy",
        selection_memberships=(_membership("ethanol", 1),),
        as_of_round=1,
        run_id="source-run-a",
        selection_rank=1,
        id="opal-candidate-a",
        sequence=CORE_A,
        synthesis_name="SECG-001",
    )
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-001", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path="out/msrb/synthesis_manifest.csv",
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        ),
    )

    stamped = apply_handoff_record_lifecycle([source_row], record)

    assert [row.run_id for row in stamped] == ["record-run-b"]


def test_measured_round_handoff_record_requires_materialization_contract() -> None:
    with pytest.raises(ValueError, match="requires materialization_contract"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="authorized_for_materialization",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
            run_id="record-run-b",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/synthesis_manifest.csv",
                vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
                genbank_dir_path="out/msrb/genbank_inserts",
                genbank_feature_table_path="out/msrb/genbank_features.csv",
            ),
        )


def test_measured_round_handoff_record_requires_canonical_study_alias_syntax() -> None:
    with pytest.raises(ValueError, match="canonical stable study alias"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="authorized_for_materialization",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
            run_id="record-run-b",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            materialization_contract=_materialization_contract(
                ("SECG-B0-ETH-01", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            ),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/synthesis_manifest.csv",
                vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
                genbank_dir_path="out/msrb/genbank_inserts",
                genbank_feature_table_path="out/msrb/genbank_features.csv",
            ),
        )


def test_handoff_record_rejects_unknown_lifecycle_status() -> None:
    with pytest.raises(ValueError, match="unsupported synthesis handoff lifecycle_status"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="ready-ish",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
            run_id="record-run-b",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            materialization_contract=_materialization_contract(
                ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            ),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/synthesis_manifest.csv",
                vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
                genbank_dir_path="out/msrb/genbank_inserts",
                genbank_feature_table_path="out/msrb/genbank_features.csv",
            ),
        )


def test_generated_handoff_record_requires_complete_artifact_receipts() -> None:
    with pytest.raises(ValueError, match="requires complete artifact digests and passing readbacks"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="generated_pending_acceptance",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
            run_id="record-run-b",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            materialization_contract=_materialization_contract(
                ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            ),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/synthesis_manifest.csv",
                vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
                genbank_dir_path="out/msrb/genbank_inserts",
                genbank_feature_table_path="out/msrb/genbank_features.csv",
            ),
        )


def test_generated_handoff_record_rejects_malformed_artifact_digest() -> None:
    with pytest.raises(ValueError, match="64 lowercase hexadecimal characters"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="generated_pending_acceptance",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
            run_id="record-run-b",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            materialization_contract=_materialization_contract(
                ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            ),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/synthesis_manifest.csv",
                vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
                genbank_dir_path="out/msrb/genbank_inserts",
                genbank_feature_table_path="out/msrb/genbank_features.csv",
                manifest_sha256="not-a-sha256",
                vendor_workbook_sha256="1" * 64,
                genbank_dir_sha256="2" * 64,
                genbank_feature_table_sha256="3" * 64,
                workbook_readback_status="pass",
                genbank_readback_status="pass",
            ),
        )


@pytest.mark.parametrize("unsafe_manifest_path", ["../outside.csv", "/tmp/outside.csv"])
def test_handoff_artifact_paths_must_remain_repo_relative(unsafe_manifest_path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path=unsafe_manifest_path,
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        )


@pytest.mark.parametrize("unsafe_input_path", ["", "../outside.yaml", "/tmp/outside.yaml"])
def test_materialization_input_paths_must_remain_repo_relative(unsafe_input_path: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        synthesis_handoff_records.MaterializationInputReceipt(
            path=unsafe_input_path,
            sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "identity_override",
    [
        {"version": 1},
        {"study_id": "other_study"},
        {"record_kind": "other_record"},
    ],
)
def test_handoff_record_loader_rejects_wrong_root_identity(
    tmp_path: Path,
    identity_override: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "version": 3,
        "study_id": "stress_ethanol_cipro_growth",
        "record_kind": "synthesis_handoff_lifecycle",
        "handoffs": [],
    }
    payload.update(identity_override)
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="root identity"):
        load_synthesis_handoff_records(record_path)


def test_opal_handoff_record_requires_explicit_round_and_physical_batch_semantics() -> None:
    with pytest.raises(ValueError, match="non-negative assay_batch_index"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r1-msrb-v1",
            lifecycle_status="authorized_for_materialization",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=None,
            model_as_of_round=1,
            run_id="record-run",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            materialization_contract=_materialization_contract(
                ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            ),
            expected_artifact=ExpectedHandoffArtifact(
                campaign_slug="secg_msrb_greedy",
                expected_rows=1,
                manifest_path="out/msrb/manifest.csv",
                vendor_workbook_path="out/msrb/order.xlsx",
                genbank_dir_path="out/msrb/genbank",
                genbank_feature_table_path="out/msrb/features.csv",
            ),
        )


def test_artifact_readiness_requires_manifest_lifecycle_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "out/msrb/manifest.csv"
    workbook_path = tmp_path / "out/msrb/order.xlsx"
    genbank_dir = tmp_path / "out/msrb/genbank"
    feature_table_path = tmp_path / "out/msrb/features.csv"
    manifest_path.parent.mkdir(parents=True)
    genbank_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "batch_id": "stress-opal-r1-msrb-v1",
                "strategy_id": "stress_promoter_insert:v1",
                "campaign_slug": "secg_msrb_greedy",
                "run_id": "wrong-run",
                "selection_epoch": "opal_model_round",
                "assay_batch_index": 1,
                "model_as_of_round": 1,
                "synthesis_name": "SECG-019",
                "selection_view_ids": '["ethanol"]',
            }
        ]
    ).to_csv(manifest_path, index=False)
    workbook_path.write_bytes(b"workbook")
    genbank_file = genbank_dir / "record.gb"
    genbank_file.write_bytes(b"genbank")
    feature_table_path.write_bytes(b"features")

    genbank_digest = hashlib.sha256()
    genbank_digest.update(genbank_file.name.encode("utf-8"))
    genbank_digest.update(b"\0")
    genbank_digest.update(genbank_file.read_bytes())
    genbank_digest.update(b"\0")
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="generated_pending_acceptance",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path=manifest_path.relative_to(tmp_path).as_posix(),
            vendor_workbook_path=workbook_path.relative_to(tmp_path).as_posix(),
            genbank_dir_path=genbank_dir.relative_to(tmp_path).as_posix(),
            genbank_feature_table_path=feature_table_path.relative_to(tmp_path).as_posix(),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            vendor_workbook_sha256=hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
            genbank_dir_sha256=genbank_digest.hexdigest(),
            genbank_feature_table_sha256=hashlib.sha256(feature_table_path.read_bytes()).hexdigest(),
            workbook_readback_status="pass",
            genbank_readback_status="pass",
        ),
    )
    monkeypatch.setattr(
        synthesis_handoff_records,
        "validate_azenta_workbook",
        lambda *args, **kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        synthesis_handoff_records,
        "validate_genbank_record_set",
        lambda *args, **kwargs: {"status": "pass"},
    )

    status = artifact_status_for_handoff_record(record, repo_root=tmp_path)

    assert status["summary"]["current_contract_ready"] is False
    artifact = status["artifacts"][0]
    assert artifact["manifest_lifecycle_status"] == "fail"
    assert "run_id" in artifact["manifest_lifecycle_error"]


def test_measured_round_handoff_record_rejects_study_alias_membership_drift() -> None:
    selected = [
        SelectedCandidate(
            campaign_slug="secg_msrb_greedy",
            selection_memberships=(_membership("ethanol", 1),),
            as_of_round=1,
            run_id="record-run-b",
            selection_rank=1,
            id="opal-candidate-a",
            sequence=CORE_A,
            synthesis_name="SECG-001",
            selection_source="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
        )
    ]
    manifest = build_synthesis_manifest(
        selected=selected,
        strategy=_strategy(),
        batch_id="stress-opal-r1-msrb-v1",
    )
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-999", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path="out/msrb/synthesis_manifest.csv",
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        ),
    )

    with pytest.raises(ValueError, match="study alias membership mismatch"):
        validate_manifest_against_handoff_record(
            manifest,
            record,
            strategy_id="stress_promoter_insert:v1",
        )


def test_measured_round_manifest_rejects_alias_candidate_binding_swap() -> None:
    selected = [
        SelectedCandidate(
            campaign_slug="secg_msrb_greedy",
            selection_memberships=(_membership("ethanol", rank),),
            as_of_round=1,
            run_id="record-run-b",
            selection_rank=rank,
            id=candidate_id,
            sequence=sequence,
            synthesis_name=study_alias,
            selection_source="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=1,
            model_as_of_round=1,
        )
        for rank, candidate_id, sequence, study_alias in (
            (1, "opal-candidate-a", CORE_A, "SECG-019"),
            (2, "opal-candidate-b", CORE_B, "SECG-020"),
        )
    ]
    manifest = build_synthesis_manifest(
        selected=selected,
        strategy=_strategy(),
        batch_id="stress-opal-r1-msrb-v1",
    )
    manifest.loc[:, "synthesis_name"] = ["SECG-020", "SECG-019"]
    materialization_contract = _materialization_contract(
        ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
        ("SECG-020", "opal-candidate-b", hashlib.sha256(CORE_B.encode("utf-8")).hexdigest()),
    )
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=2),),
        materialization_contract=materialization_contract,
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=2,
            manifest_path="out/msrb/synthesis_manifest.csv",
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        ),
    )

    with pytest.raises(ValueError, match="study alias candidate binding mismatch"):
        validate_manifest_against_handoff_record(
            manifest,
            record,
            strategy_id="stress_promoter_insert:v1",
        )


@pytest.mark.parametrize(
    "drift_field",
    [
        "campaign_config",
        "selection_batch",
        "candidate_records",
        "promoter_alias_registry",
        "cloning_strategy",
    ],
)
def test_materialization_contract_rejects_input_digest_drift(tmp_path: Path, drift_field: str) -> None:
    inputs = {
        "campaign_config": tmp_path / "inputs/campaign.yaml",
        "selection_batch": tmp_path / "inputs/selection_batch.parquet",
        "candidate_records": tmp_path / "inputs/candidate_records.parquet",
        "promoter_alias_registry": tmp_path / "inputs/promoter_aliases.yaml",
        "cloning_strategy": tmp_path / "inputs/strategy.yaml",
    }
    for label, path in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label, encoding="utf-8")
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            campaign_config=(
                inputs["campaign_config"].relative_to(tmp_path).as_posix(),
                hashlib.sha256(inputs["campaign_config"].read_bytes()).hexdigest(),
            ),
            selection_batch=(
                inputs["selection_batch"].relative_to(tmp_path).as_posix(),
                hashlib.sha256(inputs["selection_batch"].read_bytes()).hexdigest(),
            ),
            candidate_records=(
                inputs["candidate_records"].relative_to(tmp_path).as_posix(),
                hashlib.sha256(inputs["candidate_records"].read_bytes()).hexdigest(),
            ),
            promoter_alias_registry=(
                inputs["promoter_alias_registry"].relative_to(tmp_path).as_posix(),
                hashlib.sha256(inputs["promoter_alias_registry"].read_bytes()).hexdigest(),
            ),
            cloning_strategy=(
                inputs["cloning_strategy"].relative_to(tmp_path).as_posix(),
                hashlib.sha256(inputs["cloning_strategy"].read_bytes()).hexdigest(),
            ),
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path="out/msrb/synthesis_manifest.csv",
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        ),
    )
    inputs[drift_field].write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{drift_field} sha256 mismatch"):
        synthesis_handoff_records.validate_materialization_contract_inputs(
            record,
            repo_root=tmp_path,
            campaign_config_path=inputs["campaign_config"],
            selection_batch_path=inputs["selection_batch"],
            candidate_records_path=inputs["candidate_records"],
            promoter_alias_registry_path=inputs["promoter_alias_registry"],
            cloning_strategy_path=inputs["cloning_strategy"],
        )


def test_materialization_contract_rejects_candidate_records_symlink_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    inputs_dir = repo_root / "inputs"
    inputs_dir.mkdir(parents=True)
    outside_records = tmp_path / "outside-candidate-records.parquet"
    outside_records.write_text("candidate records", encoding="utf-8")
    inputs = {
        "campaign_config": inputs_dir / "campaign.yaml",
        "selection_batch": inputs_dir / "selection_batch.parquet",
        "candidate_records": inputs_dir / "candidate_records.parquet",
        "promoter_alias_registry": inputs_dir / "promoter_aliases.yaml",
        "cloning_strategy": inputs_dir / "strategy.yaml",
    }
    for field, path in inputs.items():
        if field != "candidate_records":
            path.write_text(field, encoding="utf-8")
    inputs["candidate_records"].symlink_to(outside_records)
    contract_kwargs = {
        field: (
            path.relative_to(repo_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for field, path in inputs.items()
    }
    record = SynthesisHandoffRecord(
        handoff_id="stress-opal-r1-msrb-v1",
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug="secg_msrb_greedy",
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
            **contract_kwargs,
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug="secg_msrb_greedy",
            expected_rows=1,
            manifest_path="out/msrb/synthesis_manifest.csv",
            vendor_workbook_path="out/msrb/azenta_gene_synthesis.xlsx",
            genbank_dir_path="out/msrb/genbank_inserts",
            genbank_feature_table_path="out/msrb/genbank_features.csv",
        ),
    )

    with pytest.raises(ValueError, match="candidate_records input must remain inside repository root"):
        synthesis_handoff_records.validate_materialization_contract_inputs(
            record,
            repo_root=repo_root,
            campaign_config_path=inputs["campaign_config"],
            selection_batch_path=inputs["selection_batch"],
            candidate_records_path=inputs["candidate_records"],
            promoter_alias_registry_path=inputs["promoter_alias_registry"],
            cloning_strategy_path=inputs["cloning_strategy"],
        )


def test_materialization_write_rejects_generated_artifact_symlink_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    batch_id = "stress-opal-r1-msrb-v1"
    campaign_slug = "secg_msrb_greedy"
    export_dir = campaign_synthesis_output_dir(
        repo_root,
        campaign_slug=campaign_slug,
        batch_id=batch_id,
    )
    artifact_paths = campaign_synthesis_artifact_paths(
        export_dir,
        batch_id=batch_id,
        campaign_slug=campaign_slug,
    )
    export_dir.parent.mkdir(parents=True)
    export_dir.symlink_to(tmp_path / "outside-artifacts", target_is_directory=True)
    record = SynthesisHandoffRecord(
        handoff_id=batch_id,
        lifecycle_status="authorized_for_materialization",
        source_authority="opal_selection_batch",
        selection_epoch="opal_model_round",
        assay_batch_index=1,
        model_as_of_round=1,
        run_id="record-run-b",
        strategy_id="stress_promoter_insert:v1",
        campaign_slug=campaign_slug,
        expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
        materialization_contract=_materialization_contract(
            ("SECG-019", "opal-candidate-a", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
        ),
        expected_artifact=ExpectedHandoffArtifact(
            campaign_slug=campaign_slug,
            expected_rows=1,
            manifest_path=artifact_paths["manifest"].relative_to(repo_root).as_posix(),
            vendor_workbook_path=artifact_paths["azenta_workbook"].relative_to(repo_root).as_posix(),
            genbank_dir_path=artifact_paths["genbank_dir"].relative_to(repo_root).as_posix(),
            genbank_feature_table_path=artifact_paths["genbank_feature_table"].relative_to(repo_root).as_posix(),
        ),
    )

    with pytest.raises(ValueError, match="generated manifest_path must remain inside repository root"):
        artifact_status_for_handoff_record(record, repo_root=repo_root)

    with pytest.raises(ValueError, match="manifest_path must remain inside repository root"):
        synthesis_handoff_cli._validate_record_write_paths(
            handoff_record=record,
            repo_root=repo_root,
            batch_id=batch_id,
            output_dir=None,
        )


@pytest.mark.parametrize(
    "override",
    [
        "record_yaml",
        "campaign_config",
        "promoter_alias_registry",
        "cloning_strategy",
    ],
)
def test_materialization_write_rejects_noncanonical_input_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    monkeypatch.setattr(synthesis_handoff_cli, "_source_checkout_repo_root", lambda: tmp_path.resolve())
    canonical = synthesis_handoff_cli._canonical_materialization_paths(tmp_path)
    supplied = dict(canonical)
    supplied[override] = tmp_path / f"alternate/{override}.yaml"

    with pytest.raises(ValueError, match=f"{override} must use the repository-canonical path"):
        synthesis_handoff_cli._validate_canonical_materialization_paths(
            repo_root=tmp_path,
            **supplied,
        )


def test_materialization_write_rejects_non_checkout_repo_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_root must be the active source checkout"):
        synthesis_handoff_cli._canonical_materialization_paths(tmp_path)


def test_record_loader_rejects_alias_reuse_across_committed_handoffs(tmp_path: Path) -> None:
    record_path = tmp_path / "synthesis_handoffs.yaml"
    artifact_receipts = {
        "manifest_sha256": "0" * 64,
        "vendor_workbook_sha256": "1" * 64,
        "genbank_dir_sha256": "2" * 64,
        "genbank_feature_table_sha256": "3" * 64,
        "workbook_readback_status": "pass",
        "genbank_readback_status": "pass",
    }
    record_path.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "study_id": "stress_ethanol_cipro_growth",
                "record_kind": "synthesis_handoff_lifecycle",
                "handoffs": [
                    {
                        "handoff_id": "stress-opal-r1-msrb-v1",
                        "lifecycle_status": "accepted_for_order",
                        "source_authority": "opal_selection_batch",
                        "selection_epoch": "opal_model_round",
                        "assay_batch_index": 1,
                        "model_as_of_round": 1,
                        "campaign_slug": "secg_msrb_greedy",
                        "run_id": "run-r1",
                        "strategy_id": "stress_promoter_insert:v1",
                        "expected_selection_views": [{"selection_view_id": "ethanol", "expected_rows": 1}],
                        "materialization_contract": _materialization_contract(
                            ("SECG-019", "candidate-r1", hashlib.sha256(CORE_A.encode("utf-8")).hexdigest()),
                        ).to_json(),
                        "expected_artifact": {
                            "campaign_slug": "secg_msrb_greedy",
                            "expected_rows": 1,
                            "manifest_path": "out/r1/manifest.csv",
                            "vendor_workbook_path": "out/r1/order.xlsx",
                            "genbank_dir_path": "out/r1/genbank",
                            "genbank_feature_table_path": "out/r1/features.csv",
                            **artifact_receipts,
                        },
                    },
                    {
                        "handoff_id": "stress-opal-r2-msrb-v1",
                        "lifecycle_status": "ordered",
                        "source_authority": "opal_selection_batch",
                        "selection_epoch": "opal_model_round",
                        "assay_batch_index": 2,
                        "model_as_of_round": 2,
                        "campaign_slug": "secg_msrb_greedy",
                        "run_id": "run-r2",
                        "strategy_id": "stress_promoter_insert:v1",
                        "expected_selection_views": [{"selection_view_id": "ethanol", "expected_rows": 1}],
                        "materialization_contract": _materialization_contract(
                            ("SECG-019", "candidate-r2", hashlib.sha256(CORE_B.encode("utf-8")).hexdigest()),
                        ).to_json(),
                        "expected_artifact": {
                            "campaign_slug": "secg_msrb_greedy",
                            "expected_rows": 1,
                            "manifest_path": "out/r2/manifest.csv",
                            "vendor_workbook_path": "out/r2/order.xlsx",
                            "genbank_dir_path": "out/r2/genbank",
                            "genbank_feature_table_path": "out/r2/features.csv",
                            **artifact_receipts,
                        },
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="committed synthesis handoffs reuse study alias SECG-019"):
        load_synthesis_handoff_records(record_path)

    authorized_text = record_path.read_text(encoding="utf-8").replace(
        "lifecycle_status: ordered",
        "lifecycle_status: authorized_for_materialization",
        1,
    )
    record_path.write_text(authorized_text, encoding="utf-8")
    with pytest.raises(ValueError, match="authorized synthesis handoff reuses committed study alias SECG-019"):
        load_synthesis_handoff_records(record_path)

    both_authorized_text = authorized_text.replace(
        "lifecycle_status: accepted_for_order",
        "lifecycle_status: authorized_for_materialization",
        1,
    )
    record_path.write_text(both_authorized_text, encoding="utf-8")
    with pytest.raises(ValueError, match="authorized synthesis handoffs reuse study alias SECG-019"):
        load_synthesis_handoff_records(record_path)

    pending_text = authorized_text.replace(
        "lifecycle_status: accepted_for_order",
        "lifecycle_status: generated_pending_acceptance",
        1,
    )
    record_path.write_text(pending_text, encoding="utf-8")
    assert set(load_synthesis_handoff_records(record_path)) == {
        "stress-opal-r1-msrb-v1",
        "stress-opal-r2-msrb-v1",
    }


def test_legacy_batch0_record_cannot_claim_committed_status_without_alias_disposition(tmp_path: Path) -> None:
    canonical = Path("docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml")
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        canonical.read_text(encoding="utf-8").replace(
            "lifecycle_status: generated_pending_acceptance",
            "lifecycle_status: accepted_for_order",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy batch-0 handoff cannot enter committed lifecycle_status"):
        load_synthesis_handoff_records(record_path)


def test_cli_opal_round_handoff_record_requires_explicit_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_opal_round_fixture(tmp_path, as_of_round=1)
    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        textwrap.dedent(
            """
            version: 3
            study_id: stress_ethanol_cipro_growth
            record_kind: synthesis_handoff_lifecycle
            handoffs:
              - handoff_id: stress-opal-r1-msrb-v1
                lifecycle_status: authorized_for_materialization
                source_authority: opal_selection_batch
                selection_epoch: opal_model_round
                assay_batch_index: 1
                model_as_of_round: 1
                campaign_slug: secg_msrb_greedy
                run_id: null
                strategy_id: stress_promoter_insert:v1
                expected_selection_views:
                  - selection_view_id: ethanol
                    expected_rows: 2
                materialization_contract:
                  campaign_config:
                    path: inputs/campaign.yaml
                    sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                  selection_batch:
                    path: inputs/selection_batch.parquet
                    sha256: "1111111111111111111111111111111111111111111111111111111111111111"
                  candidate_records:
                    path: inputs/candidate_records.parquet
                    sha256: "2222222222222222222222222222222222222222222222222222222222222222"
                  promoter_alias_registry:
                    path: inputs/promoter_aliases.yaml
                    sha256: "3333333333333333333333333333333333333333333333333333333333333333"
                  cloning_strategy:
                    path: inputs/strategy.yaml
                    sha256: "4444444444444444444444444444444444444444444444444444444444444444"
                  expected_candidates:
                    - study_alias: SECG-001
                      candidate_id: candidate-b
                      core_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                    - study_alias: SECG-002
                      candidate_id: candidate-a
                      core_sha256: "1111111111111111111111111111111111111111111111111111111111111111"
                    - study_alias: SECG-003
                      candidate_id: candidate-c
                      core_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
                expected_artifact:
                  campaign_slug: secg_msrb_greedy
                  expected_rows: 3
                  manifest_path: out/msrb/synthesis_manifest.csv
                  vendor_workbook_path: out/msrb/azenta_gene_synthesis.xlsx
                  genbank_dir_path: out/msrb/genbank_inserts
                  genbank_feature_table_path: out/msrb/genbank_features.csv
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--handoff-id",
                "stress-opal-r1-msrb-v1",
                "--record-yaml",
                str(record_path),
                "--campaign-config",
                str(config_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "requires explicit run_id" in payload["error"]["message"]


def test_cli_handoff_record_rejects_campaign_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_batch0_cli_source(monkeypatch)

    record_path = tmp_path / "synthesis_handoffs.yaml"
    record_path.write_text(
        textwrap.dedent(
            """
            version: 3
            study_id: stress_ethanol_cipro_growth
            record_kind: synthesis_handoff_lifecycle
            handoffs:
              - handoff_id: stress-opal-batch0-sfxi-v1
                lifecycle_status: generated_pending_acceptance
                source_authority: study_batch0_selector
                selection_epoch: pre_assay_seed
                assay_batch_index: 0
                model_as_of_round: null
                run_id: batch0_pre_assay_review
                strategy_id: stress_promoter_insert:v1
                expected_campaigns:
                  - campaign_slug: secg_ethanol_rf_sfxi_topn
                    expected_rows: 7
                    manifest_path: out/ethanol/synthesis_manifest.csv
                    vendor_workbook_path: out/ethanol/azenta_gene_synthesis.xlsx
                    genbank_dir_path: out/ethanol/genbank_inserts
                    genbank_feature_table_path: out/ethanol/genbank_features.csv
                    manifest_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                    vendor_workbook_sha256: "1111111111111111111111111111111111111111111111111111111111111111"
                    genbank_dir_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
                    genbank_feature_table_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
                    workbook_readback_status: pass
                    genbank_readback_status: pass
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
                "--source",
                "batch0",
                "--record-yaml",
                str(record_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "handoff record campaign row mismatch" in payload["error"]["message"]
    assert "secg_ethanol_rf_sfxi_topn" in payload["error"]["message"]


def test_raw_opal_round_preview_requires_explicit_physical_batch_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        synthesis_handoff_main(
            [
                "--source",
                "opal-round",
                "--round",
                "1",
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "requires an explicit --batch-id" in payload["error"]["message"]


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
                "--batch-id",
                "stress-opal-batch1-msrb-v1",
                "--campaign-config",
                str(config_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 2
    payload = _json_cli_error(capsys.readouterr().err)
    assert "required OPAL parquet artifact is missing" in payload["error"]["message"]


def _repository_msrb_round0_handoff_artifacts(repo_root: Path) -> tuple[Path, ...]:
    campaign_root = repo_root / "src/dnadesign/opal/campaigns/secg_msrb_greedy"
    return (
        repo_root / "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet",
        campaign_root / "outputs/ledger/predictions",
        campaign_root / "outputs/ledger/runs.parquet",
        campaign_root / "outputs/rounds/round_0/run_artifacts",
    )


def _require_repository_msrb_round0_handoff_artifacts(repo_root: Path) -> None:
    required = _repository_msrb_round0_handoff_artifacts(repo_root)
    missing = tuple(path for path in required if not path.exists())
    if not missing:
        return
    if len(missing) == len(required):
        pytest.skip(f"requires local MSRB round-0 OPAL artifacts; missing {missing[0].relative_to(repo_root)}")
    missing_paths = [str(path.relative_to(repo_root)) for path in missing]
    raise AssertionError(f"MSRB round-0 OPAL artifacts are partially materialized; missing={missing_paths}")


def test_repository_msrb_round0_handoff_gate_skips_unmaterialized_inputs(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="requires local MSRB round-0 OPAL artifacts"):
        _require_repository_msrb_round0_handoff_artifacts(tmp_path)


def test_repository_msrb_round0_handoff_gate_rejects_partial_materialization(tmp_path: Path) -> None:
    candidate_records = tmp_path / "src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet"
    candidate_records.parent.mkdir(parents=True)
    candidate_records.touch()

    with pytest.raises(AssertionError, match="partially materialized"):
        _require_repository_msrb_round0_handoff_artifacts(tmp_path)


def test_current_msrb_round0_receipt_is_complete_sequence_unique_alias_handoff() -> None:
    repo_root = Path(__file__).resolve().parents[9]
    run_id = "r0-2026-07-19T22:21:41+00:00-01784499701298508000-24e5927eb1ce4d0daf013dc0c352c584"
    baseline_path = (
        repo_root / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/"
        "multistate_response_behavior/evaluation_baseline.yaml"
    )
    alias_registry_path = repo_root / "docs/studies/stress_ethanol_cipro_growth/record/promoter_aliases.yaml"
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    alias_registry = yaml.safe_load(alias_registry_path.read_text(encoding="utf-8"))
    allocations = baseline["allocations"]
    aliases = [f"SECG-{ordinal:03d}" for ordinal in range(19, 37)]

    assert baseline["campaign"]["run_id"] == run_id
    assert [row["study_alias"] for row in allocations] == aliases
    assert [row["selection_view"] for row in allocations] == ["ethanol"] * 6 + ["ciprofloxacin"] * 6 + ["and"] * 6
    assert [row["allocation_slot"] for row in allocations] == [*range(1, 7)] * 3
    assert len({row["candidate_id"] for row in allocations}) == 18
    assert len({row["sequence_sha256"] for row in allocations}) == 18

    assignments_by_alias = {row["alias"]: row for row in alias_registry["assignments"]}
    assert len(assignments_by_alias) == 36
    for allocation in allocations:
        assignment = assignments_by_alias[allocation["study_alias"]]
        assert assignment["candidate_id"] == allocation["candidate_id"]
        assert assignment["sequence_sha256"] == allocation["sequence_sha256"]
        assert assignment["first_assignment"] == {
            "source_authority": "opal_selection_batch",
            "source_id": run_id,
            "nomination_batch_index": 1,
            "model_as_of_round": 0,
        }


def test_current_accepted_msrb_handoff_bundle_is_durable_and_verified() -> None:
    repo_root = Path(__file__).resolve().parents[9]
    record_path = repo_root / "docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml"
    record = load_synthesis_handoff_records(record_path)["stress-opal-assay-b1-r0-msrb-v1"]

    status = artifact_status_for_handoff_record(record, repo_root=repo_root)

    assert record.lifecycle_status == "accepted_for_order"
    assert status["summary"] == {
        "expected_artifact_count": 1,
        "present_artifact_count": 1,
        "manifest_lifecycle_pass_count": 1,
        "workbook_readback_pass_count": 1,
        "genbank_readback_pass_count": 1,
        "current_contract_ready": True,
    }
    artifact = status["artifacts"][0]
    assert artifact["manifest_row_count"] == 18
    assert artifact["manifest_lifecycle_validation"]["selection_view_counts"] == {
        "ethanol": 6,
        "ciprofloxacin": 6,
        "and": 6,
    }
    assert artifact["manifest_lifecycle_validation"]["study_aliases"] == [
        f"SECG-{ordinal:03d}" for ordinal in range(19, 37)
    ]


def test_local_current_msrb_round0_is_complete_sequence_unique_genbank_handoff(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[9]
    _require_repository_msrb_round0_handoff_artifacts(repo_root)
    config_path = repo_root / "src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml"
    run_id = "r0-2026-07-19T22:21:41+00:00-01784499701298508000-24e5927eb1ce4d0daf013dc0c352c584"
    selected, report = selected_candidates_from_opal_round(
        config_path,
        as_of_round=0,
        run_id=run_id,
        repo_root=repo_root,
    )
    manifest = build_synthesis_manifest(
        selected=selected,
        strategy=_strategy(),
        batch_id="stress-opal-batch1-msrb-v1",
    )
    feature_table = build_genbank_feature_table(
        manifest,
        candidate_records_path=report["candidate_records_path"],
    )

    assert manifest["synthesis_name"].tolist() == [f"SECG-{ordinal:03d}" for ordinal in range(19, 37)]
    assert manifest["id"].nunique() == 18
    assert manifest["core_sequence"].nunique() == 18
    assert manifest["synthesis_name"].nunique() == 18
    assert report["selection_view_counts"] == {"ethanol": 6, "ciprofloxacin": 6, "and": 6}
    assert report["unique_candidate_count"] == 18
    assert report["unique_sequence_count"] == 18
    assert report["unique_study_alias_count"] == 18
    assert report["study_aliases"] == [f"SECG-{ordinal:03d}" for ordinal in range(19, 37)]
    assert report["replay_mismatch_count"] == 0
    assert report["promoter_alias_registry"]["next_alias"] == "SECG-037"
    assert len(feature_table) == 134
    assert feature_table["feature_id"].astype(str).str.len().max() <= 32
    assert feature_table["source"].value_counts().to_dict() == {
        "cloning_strategy": 36,
        "densegen_fixed_element": 36,
        "densegen_tfbs": 26,
        "synthesis_handoff": 18,
        "opal_candidate": 18,
    }

    exports = render_campaign_scoped_exports(
        manifest,
        batch_id="stress-opal-batch1-msrb-v1",
        output_owner="campaign",
        output_root=tmp_path,
        candidate_records_path=report["candidate_records_path"],
    )
    assert exports[["row_count", "azenta_validation_status", "genbank_validation_status"]].to_dict("records") == [
        {"row_count": 18, "azenta_validation_status": "pass", "genbank_validation_status": "pass"}
    ]
    assert len(list(Path(exports.iloc[0]["genbank_dir_path"]).glob("*.gb"))) == 18
