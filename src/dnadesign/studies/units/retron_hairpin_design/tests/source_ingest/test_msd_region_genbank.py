"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/source_ingest/test_msd_region_genbank.py

MSD region GenBank ingest contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import resolve_msd_compiler_spec_payload
from dnadesign.studies.units.retron_hairpin_design.source_ingest.msd_region_genbank import (
    compare_records_to_existing_sources,
    compiler_spec_payload_from_records,
    parse_msd_region_genbank,
    write_msd_region_record_bundle,
)


def test_parse_msd_region_genbank_filters_unresolved_copy_records_and_reverse_complements(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    display_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(
        source,
        [
            _record("msd-retron-170", display_sequence),
            _unresolved_copy_record(),
        ],
    )

    bundle = parse_msd_region_genbank(source)

    assert [record.variant_id for record in bundle.records] == ["retron170"]
    assert bundle.skipped_records[0].record_id == "Copy_of_Copy_of_Copy_of_m"
    assert bundle.skipped_records[0].reason == "unresolved_variant_id"
    record = bundle.records[0]
    assert record.display_id == "pES-retron-170"
    assert record.msd_sequence_5to3 == display_sequence
    assert record.file_stem == "pes-retron-170-msd-region"
    assert record.primitive("stem_base_left").sequence_5to3 == "CGGG"
    assert record.primitive("payload_primary").sequence_5to3 == "TCCCTATCAGTGATAGAGA"
    assert record.primitive("snapback_foldback_geometry").sequence_5to3 == "AGGC"
    assert record.annotation_status == "label_only_normalized"


def test_write_bundle_emits_clean_per_variant_records_manifest_and_compiler_spec(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    display_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source, [_record("msd-retron-170", display_sequence)])
    bundle = parse_msd_region_genbank(source)

    written = write_msd_region_record_bundle(bundle, output_dir=tmp_path / "bundle")

    manifest = yaml.safe_load(Path(written.manifest_path).read_text(encoding="utf-8"))
    assert manifest["contract"] == "retron_msd_region_record_bundle_v1"
    assert manifest["source_policy"] == "decomposed_records_are_authority"
    assert manifest["included_record_count"] == 1
    variant_path = Path(written.variant_record_paths["retron170"])
    assert variant_path.name == "pes-retron-170-msd-region.yaml"
    variant_payload = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
    assert variant_payload["contract"] == "retron_msd_region_record_v1"
    assert variant_payload["msd_sequence_5to3"] == display_sequence
    compiler_spec = yaml.safe_load(Path(written.compiler_spec_path).read_text(encoding="utf-8"))
    assert compiler_spec == compiler_spec_payload_from_records(bundle.records)


def test_generated_compiler_spec_loads_through_existing_genetic_compiler_boundary(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    display_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source, [_record("msd-retron-170", display_sequence)])
    bundle = parse_msd_region_genbank(source)

    resolved = resolve_msd_compiler_spec_payload(
        compiler_spec_payload_from_records(bundle.records),
        study_dir=_study_dir(),
        allow_non_ligatable_s0=True,
    )

    assert len(resolved.catalog.records) == 1
    assert resolved.catalog.records[0].construct_id == "pES-retron-170"
    assert resolved.payload_sequences["MSDRegion170_payload"] == "TCCCTATCAGTGATAGAGA"
    assert resolved.cap_sequences["C170_msd_region"] == "AGGC"


def test_oversized_stem_base_annotations_are_review_warnings_not_compiler_bases(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    _write_genbank(source, [_record_with_oversized_stem_base_annotations("msd-retron-45")])
    record = parse_msd_region_genbank(source).records[0]

    assert record.primitive("stem_base_left").sequence_5to3 == "CGGG"
    assert record.primitive("stem_base_right").sequence_5to3 == "ACAG"
    assert {feature.role for feature in record.features} >= {
        "stem_base_left_annotated_span",
        "stem_base_right_annotated_span",
    }
    assert [warning.kind for warning in record.annotation_warnings] == [
        "stem_base_annotation_span_adjusted",
        "stem_base_annotation_span_adjusted",
    ]

    resolved = resolve_msd_compiler_spec_payload(
        compiler_spec_payload_from_records((record,)),
        study_dir=_study_dir(),
        allow_non_ligatable_s0=True,
    )

    parsed = resolved.catalog.records[0]
    assert parsed.scar_nick.left_base == "CGGG"
    assert parsed.scar_nick.right_base == "ACAG"


def test_compare_records_to_existing_sources_flags_sequence_and_annotation_drift(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    display_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source, [_record("msd-retron-170", display_sequence)])
    record = parse_msd_region_genbank(source).records[0]
    existing_dir = tmp_path / "existing"
    existing_variant = existing_dir / "variants" / "pES-retron-170__demo"
    sequences_dir = existing_variant / "sequences"
    sequences_dir.mkdir(parents=True)
    drifted = display_sequence[:-1] + "T"
    _write_genbank(sequences_dir / "forward.gb", [_record("pES-retron-170", drifted, source_oriented=False)])
    _write_features_csv(
        sequences_dir / "features.csv",
        [
            {
                "role": "stem_base_left",
                "start_0": "12",
                "end_0": "16",
                "sequence": "GGGT",
            }
        ],
    )
    manifest_dir = existing_dir / "manifest" / "indexes"
    manifest_dir.mkdir(parents=True)
    _write_sequence_index(
        manifest_dir / "sequence_index.tsv",
        genbank_path="variants/pES-retron-170__demo/sequences/forward.gb",
        features_path="variants/pES-retron-170__demo/sequences/features.csv",
    )

    report = compare_records_to_existing_sources((record,), existing_roots=(existing_dir,))

    assert report.comparison_count == 1
    assert report.discrepancy_count == 2
    assert {item.kind for item in report.discrepancies} == {"sequence_mismatch", "annotation_mismatch"}


def _write_genbank(path: Path, records: list[SeqRecord]) -> None:
    for record in records:
        record.annotations["molecule_type"] = "DNA"
    SeqIO.write(records, path, "genbank")


def _study_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "docs/studies/retron_hairpin_design"
    raise RuntimeError("repo root not found")


def _record(record_id: str, display_sequence: str, *, source_oriented: bool = True) -> SeqRecord:
    source_sequence = str(Seq(display_sequence).reverse_complement()).upper() if source_oriented else display_sequence
    record = SeqRecord(Seq(source_sequence), id=record_id, name=record_id, description=record_id)
    if not source_oriented:
        return record
    length = len(source_sequence)
    display_features = (
        ("flank_5p", "5' Flanking", 0, 15, -1),
        ("stem_base_left", "Left Base", 11, 15, -1),
        ("payload_primary", "tet operator", 15, 34, -1),
        ("snapback_foldback_geometry", "WT loop", 34, 38, 1),
        ("payload_complement", "tet operator [complement]", 38, 57, 1),
        ("stem_base_right", "Right base", 57, 61, 1),
        ("flank_3p", "3' Flanking", 61, length, 1),
    )
    record.features = [
        SeqFeature(
            FeatureLocation(length - end, length - start, strand=strand),
            type="misc_feature",
            qualifiers={"label": [label]},
        )
        for _role, label, start, end, strand in display_features
    ]
    return record


def _unresolved_copy_record() -> SeqRecord:
    record = SeqRecord(Seq("ACGTACGT"), id="Copy_of_Copy_of_Copy_of_m", name="Copy_of_Copy_of_Copy_of_m")
    record.description = "Copy_of_Copy_of_Copy_of_m"
    return record


def _record_with_oversized_stem_base_annotations(record_id: str) -> SeqRecord:
    display_sequence = "GTCAGAAAAAACGGGTTTCTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAGAAAACAGACAGTAACTCAGA"
    source_sequence = str(Seq(display_sequence).reverse_complement()).upper()
    record = SeqRecord(Seq(source_sequence), id=record_id, name=record_id, description=record_id)
    length = len(source_sequence)
    display_features = (
        ("5' Flanking", 0, 15, -1),
        ("Left base", 11, 19, -1),
        ("tet operator", 19, 38, -1),
        ("WT loop", 38, 42, 1),
        ("tet operator [complement]", 42, 61, 1),
        ("Right base", 61, 69, 1),
        ("3' Flanking", 65, length, 1),
    )
    record.features = [
        SeqFeature(
            FeatureLocation(length - end, length - start, strand=strand),
            type="misc_feature",
            qualifiers={"label": [label]},
        )
        for label, start, end, strand in display_features
    ]
    return record


def _write_sequence_index(path: Path, *, genbank_path: str, features_path: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("construct_id", "genbank", "features_csv"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "construct_id": "pES-retron-170",
                "genbank": genbank_path,
                "features_csv": features_path,
            }
        )


def _write_features_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "start_0", "end_0", "sequence"))
        writer.writeheader()
        writer.writerows(rows)
