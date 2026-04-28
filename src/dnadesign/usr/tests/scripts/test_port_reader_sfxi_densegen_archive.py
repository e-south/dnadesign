from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dnadesign.baserender import adapt_record
from dnadesign.densegen import densegen_notebook_render_contract
from dnadesign.densegen.src.core.metadata_schema import META_FIELDS, validate_metadata
from dnadesign.usr import Dataset
from dnadesign.usr.scripts import port_reader_sfxi_densegen_archive as port
from dnadesign.usr.src.contracts import compute_id
from dnadesign.usr.src.overlays import overlay_metadata
from dnadesign.usr.src.registry import registry_hash


def _revcomp(sequence: str) -> str:
    table = str.maketrans("ACGT", "TGCA")
    return sequence.translate(table)[::-1]


def _sequence_with_parts(parts: list[dict[str, object]]) -> str:
    sequence = ["A"] * 60
    for part in parts:
        literal = str(part["tfbs"])
        offset = int(part["offset"])
        placed = literal if str(part.get("orientation", "fwd")) == "fwd" else _revcomp(literal)
        sequence[offset : offset + len(placed)] = list(placed)
    return "".join(sequence)


TFBS_PARTS = [
    {"offset": 2, "orientation": "fwd", "tf": "cpxr", "tfbs": "GTCGTA"},
    {"offset": 13, "orientation": "rev", "tf": "cpxr", "tfbs": "CCATGA"},
    {"offset": 35, "orientation": "fwd", "tf": "lexa", "tfbs": "GATCTA"},
]
SIGMA70_MID_PARTS = [
    {"offset": 24, "orientation": "fwd", "tf": "sigma70_mid_upstream", "tfbs": "ACCGCG"},
    {"offset": 48, "orientation": "fwd", "tf": "sigma70_mid_downstream", "tfbs": "TATAAT"},
]
SIGMA70_HIGH_PARTS = [
    {"offset": 24, "orientation": "fwd", "tf": "sigma70_high_upstream", "tfbs": "TTGACA"},
    {"offset": 48, "orientation": "fwd", "tf": "sigma70_high_downstream", "tfbs": "TATAAT"},
]

SEQ_ES1_PDUAL = _sequence_with_parts([*TFBS_PARTS, *SIGMA70_MID_PARTS])
SEQ_ES1_PSINGLE = "C" * 51
SEQ_ES6_SHARED = _sequence_with_parts([*TFBS_PARTS, *SIGMA70_HIGH_PARTS])
SEQ_ES8_READER_ONLY = "G" * 60
SEQ_ES5_ES9_AMBIGUOUS = "T" * 60
SEQ_ES9_AMBIGUOUS = "ATGC" * 15


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, sheet_name="induced", index=False)


def _reader_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "reader"
    _write_metadata(
        root / "experiments" / "2025" / "20250915_sfxi_pSingle_ref" / "inputs" / "metadata.xlsx",
        [
            {"design_id": "pDual-10-ES1p", "sequence": SEQ_ES1_PSINGLE, "treatment": "mock"},
            {"design_id": "pDual-10-ES6p", "sequence": SEQ_ES6_SHARED, "treatment": "mock"},
        ],
    )
    _write_metadata(
        root / "experiments" / "2025" / "20251103_sfxi_pES1-8_ref-pDual10" / "inputs" / "metadata.xlsx",
        [
            {"design_id": "pDual-10-ES1p", "sequence": SEQ_ES1_PDUAL, "treatment": "mock"},
            {"design_id": "pDual-10-ES6p", "sequence": SEQ_ES6_SHARED, "treatment": "mock"},
            {"design_id": "pDual-10-ES8p", "sequence": SEQ_ES8_READER_ONLY, "treatment": "mock"},
            {"design_id": "pDual-10-ES5p", "sequence": SEQ_ES5_ES9_AMBIGUOUS, "treatment": "mock"},
            {"design_id": "pDual-10-ES9p", "sequence": SEQ_ES5_ES9_AMBIGUOUS, "treatment": "mock"},
            {"design_id": "pDual-10-ES9p", "sequence": SEQ_ES9_AMBIGUOUS, "treatment": "mock"},
            {"design_id": "pDual-10", "sequence": None, "treatment": "mock"},
        ],
    )
    (root / "experiments" / "2025" / "20251103_sfxi_pES1-8_ref-pDual10" / "config.yaml").write_text(
        "protocol:\n  id: logic/sfxi_screen\n",
        encoding="utf-8",
    )
    return root


def _archive_row(
    sequence: str,
    *,
    plan: str = "sigma70_mid",
    promoter_constraint: object = "sigma70_mid_35_strong_10",
    sigma70_parts: list[dict[str, object]] | None = None,
    gap_fill_bases: int = 0,
    gap_fill_end: str | None = None,
) -> dict[str, object]:
    sigma70_parts = sigma70_parts or SIGMA70_MID_PARTS
    used_parts = [*TFBS_PARTS, *sigma70_parts]
    gap_fill_used = gap_fill_bases > 0
    return {
        "id": f"legacy-{compute_id('dna', sequence)[:12]}",
        "bio_type": "dna",
        "sequence": sequence,
        "alphabet": "dna_4",
        "length": 60,
        "source": "archive-fixture",
        "created_at": "2025-09-07T01:33:01Z",
        "densegen__compression_ratio": 1.05,
        "densegen__covers_all_tfs_in_solution": True,
        "densegen__gap_fill_bases": gap_fill_bases if gap_fill_used else 0,
        "densegen__gap_fill_end": gap_fill_end if gap_fill_used else None,
        "densegen__gap_fill_used": gap_fill_used,
        "densegen__library_size": 10,
        "densegen__min_count_per_tf_required": 1,
        "densegen__plan": plan,
        "densegen__promoter_constraint": promoter_constraint,
        "densegen__sequence_length": 60,
        "densegen__tf_list": ["cpxr", "lexa"],
        "densegen__used_tf_counts": {"cpxr": 2, "lexa": 1},
        "densegen__used_tf_list": ["cpxr", "lexa"],
        "densegen__used_tfbs": ["cpxr:GTCGTA", "cpxr:CCATGA", "lexa:GATCTA"],
        "densegen__used_tfbs_detail": used_parts,
    }


def _archive_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "archive" / "records.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            _archive_row(SEQ_ES1_PDUAL),
            _archive_row(
                SEQ_ES6_SHARED,
                plan="sigma70_high",
                promoter_constraint=float("nan"),
                sigma70_parts=SIGMA70_HIGH_PARTS,
                gap_fill_bases=2,
                gap_fill_end="5prime",
            ),
            _archive_row(SEQ_ES5_ES9_AMBIGUOUS),
            _archive_row(SEQ_ES9_AMBIGUOUS),
        ]
    ).to_parquet(path, index=False)
    return path


def test_refined_selection_keeps_pdual_archive_matches_and_drops_ambiguous_rows(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)

    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)

    assert [row.design_id for row in audit.included] == ["pDual-10-ES1p", "pDual-10-ES6p"]
    excluded = {(row.design_id, row.reason) for row in audit.excluded}
    assert ("pDual-10-ES1p", "non_pdual_reader_evidence") in excluded
    assert ("pDual-10-ES8p", "no_archive_densegen_match") in excluded
    assert ("pDual-10-ES5p", "ambiguous_pdual_design_or_sequence") in excluded
    assert ("pDual-10-ES9p", "ambiguous_pdual_design_or_sequence") in excluded
    assert audit.blank_sequence_rows == 1


def test_archive_rows_map_to_registered_modern_densegen_overlay_shape(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)
    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)

    overlay = port.modern_densegen_overlay_frame(audit.included)

    assert "densegen__gap_fill_used" not in overlay.columns
    assert "densegen__schema_version" in overlay.columns
    assert overlay["densegen__schema_version"].tolist() == [port.MODERN_DENSEGEN_SCHEMA_VERSION] * 2
    assert overlay["densegen__input_mode"].tolist() == ["plan_pool", "plan_pool"]
    assert overlay["densegen__plan"].tolist() == ["ethanol_ciprofloxacin", "ethanol_ciprofloxacin"]
    assert overlay["densegen__required_regulators"].tolist() == [
        ["cpxR_MANWWHTTTAM", "lexA_CTGTATAWAWWHACA"],
        ["cpxR_MANWWHTTTAM", "lexA_CTGTATAWAWWHACA"],
    ]
    assert overlay["densegen__used_tf_counts"].iloc[0] == [
        {"tf": "cpxR_MANWWHTTTAM", "count": 2},
        {"tf": "lexA_CTGTATAWAWWHACA", "count": 1},
    ]
    tfbs_detail = overlay["densegen__used_tfbs_detail"].iloc[0][0]
    assert set(tfbs_detail) == set(port.MODERN_TFBS_DETAIL_FIELDS)
    assert tfbs_detail["part_kind"] == "tfbs"
    assert tfbs_detail["regulator"] == "cpxR_MANWWHTTTAM"
    assert tfbs_detail["sequence"] == "GTCGTA"
    sigma_detail = overlay["densegen__used_tfbs_detail"].iloc[0][3]
    assert set(sigma_detail) == set(port.MODERN_TFBS_DETAIL_FIELDS)
    assert sigma_detail["part_kind"] == "fixed_element"
    assert sigma_detail["role"] == "upstream"
    assert sigma_detail["constraint_name"] == "sigma70_core"
    assert sigma_detail["variant_id"] == "ACCGCG"
    assert sigma_detail["spacer_length"] == 18
    assert overlay["densegen__used_tfbs_detail"].iloc[0][4]["role"] == "downstream"
    assert overlay["densegen__used_tfbs_detail"].iloc[0][4]["variant_id"] == "consensus"
    assert overlay["densegen__used_tfbs_detail"].iloc[1][3]["variant_id"] == "f"
    assert overlay["densegen__pad_used"].tolist() == [False, True]
    assert overlay["densegen__pad_bases"].tolist() == [None, 2]
    assert overlay["densegen__pad_end"].tolist() == [None, "5prime"]
    assert overlay["densegen__pad_literal"].tolist() == [None, SEQ_ES6_SHARED[:2]]


def test_modern_densegen_rows_pass_densegen_and_baserender_contracts(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)
    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)

    overlay = port.modern_densegen_overlay_frame(audit.included)
    for row in overlay.to_dict(orient="records"):
        meta = {field.name: row[f"densegen__{field.name}"] for field in META_FIELDS}
        validate_metadata(meta)

    row = overlay.iloc[0].to_dict() | {"sequence": SEQ_ES1_PDUAL}
    contract = densegen_notebook_render_contract()
    policies = dict(contract.adapter_policies)
    policies.update({"require_non_empty": True, "min_per_record": 5})
    record = adapt_record(
        row,
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=policies,
        alphabet="DNA",
    )
    tfbs_features = [feature for feature in record.features if any(tag.startswith("tf:") for tag in feature.tags)]
    promoter_features = [feature for feature in record.features if feature.attrs.get("source") == "densegen_promoter"]
    assert len(tfbs_features) == 3
    assert len(promoter_features) == 2
    assert len(record.effects) == 1
    assert {feature.attrs.get("component") for feature in promoter_features} == {"upstream", "downstream"}
    assert {feature.attrs.get("variant_id") for feature in promoter_features} == {"ACCGCG", "consensus"}


def test_write_port_dataset_uses_base_records_and_densegen_sidecar(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)
    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")

    result = port.write_port_dataset(
        audit,
        usr_root=usr_root,
        output_dataset="usr_sfxi_pdual10_densegen_promoters",
    )

    assert result.rows_written == 2
    dataset = Dataset(usr_root, "usr_sfxi_pdual10_densegen_promoters")
    base = pq.read_table(dataset.records_path)
    assert base.num_rows == 2
    assert not any(name.startswith("densegen__") for name in base.column_names)
    densegen = pq.read_table(dataset.dir / "_derived" / "densegen.parquet")
    assert densegen.num_rows == 2
    assert "densegen__used_tfbs_detail" in densegen.column_names
    assert densegen.column("id").to_pylist() == [compute_id("dna", SEQ_ES1_PDUAL), compute_id("dna", SEQ_ES6_SHARED)]
    labels = pq.read_table(dataset.dir / "_derived" / "usr_label.parquet")
    assert labels.num_rows == 2
    assert any(alias.startswith("archive_id:legacy-") for alias in labels.column("usr_label__aliases").to_pylist()[0])
    dataset.validate(strict=True)


def test_overlap_report_flags_existing_sfxi_sequences_by_dataset(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)
    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")

    source = Dataset(usr_root, "densegen_source")
    with source.write_session() as session:
        session.init(source="fixture", notes="fixture source")
        session.import_rows(port.base_records_frame([audit.included[0]]), source="fixture")
    anchor = Dataset(usr_root, "study_anchor")
    with anchor.write_session() as session:
        session.init(source="fixture", notes="fixture anchor")
        session.import_rows(port.base_records_frame([audit.included[1]]), source="fixture")

    report = port.compare_port_plan_to_datasets(
        audit,
        usr_root=usr_root,
        dataset_names=("densegen_source", "study_anchor", "missing_dataset"),
    )

    assert report["included_total"] == 2
    assert report["datasets"]["densegen_source"]["matched_count"] == 1
    assert report["datasets"]["densegen_source"]["matched_design_ids"] == ["pDual-10-ES1p"]
    assert report["datasets"]["study_anchor"]["matched_count"] == 1
    assert report["datasets"]["study_anchor"]["matched_design_ids"] == ["pDual-10-ES6p"]
    assert report["datasets"]["missing_dataset"]["exists"] is False


def test_refresh_existing_port_dataset_overlays_uses_current_registry_metadata(tmp_path: Path) -> None:
    reader_root = _reader_fixture(tmp_path)
    archive_path = _archive_fixture(tmp_path)
    audit = port.build_port_plan(reader_root=reader_root, archive_records=archive_path)
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")

    port.write_port_dataset(
        audit,
        usr_root=usr_root,
        output_dataset="usr_sfxi_pdual10_densegen_promoters",
    )

    result = port.refresh_existing_port_dataset(
        audit,
        usr_root=usr_root,
        output_dataset="usr_sfxi_pdual10_densegen_promoters",
    )

    dataset = Dataset(usr_root, "usr_sfxi_pdual10_densegen_promoters")
    dataset.validate(strict=True)
    current_hash = registry_hash(usr_root, required=True)
    densegen_meta = overlay_metadata(dataset.dir / "_derived" / "densegen.parquet")
    label_meta = overlay_metadata(dataset.dir / "_derived" / "usr_label.parquet")
    assert result.rows_written == 0
    assert result.densegen_overlay_rows == 2
    assert result.label_overlay_rows == 2
    assert densegen_meta["registry_hash"] == current_hash
    assert label_meta["registry_hash"] == current_hash
