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
import inspect
import json
from pathlib import Path

import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import resolve_msd_compiler_spec_payload
from dnadesign.studies.units.retron_hairpin_design.compiler.msd_unit import compile_msd_design_unit
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app
from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.msd_region_ingest import ingest_msd_regions_command
from dnadesign.studies.units.retron_hairpin_design.source_ingest.msd_region_genbank import (
    compare_records_to_existing_sources,
    compiler_spec_payload_from_records,
    load_payload_binding_catalog,
    parse_msd_region_genbank,
    parse_msd_region_genbank_dir,
    parse_msd_region_genbank_with_replacements,
    write_msd_region_record_bundle,
    write_variant_genbank_sources,
)

from ..support.cli import RUNNER
from ..support.pwm_fixtures import write_test_tetr_meme_pwm


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


def test_replacement_genbanks_overlay_matching_base_records(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    replacement = tmp_path / "msd-retron-170.gb"
    base_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    replacement_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGATCCTCAGCCCGCTGAGGATCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source, [_record("msd-retron-170", base_sequence)])
    _write_genbank(replacement, [_record_with_narrow_foldback_annotation("msd-retron-170", replacement_sequence)])

    bundle = parse_msd_region_genbank_with_replacements(source, replacement_paths=(replacement,))

    assert bundle.included_record_count == 1
    assert bundle.source_record_count == 2
    assert bundle.records[0].msd_sequence_5to3 == replacement_sequence
    assert bundle.replacement_sources[0]["included_variant_ids"] == ["retron170"]

    written = write_msd_region_record_bundle(bundle, output_dir=tmp_path / "bundle")
    manifest = yaml.safe_load(Path(written.manifest_path).read_text(encoding="utf-8"))
    assert manifest["replacement_sources"][0]["included_variant_ids"] == ["retron170"]


def test_variant_genbank_source_dir_is_steady_state_authority(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    replacement = tmp_path / "msd-retron-170.gb"
    base_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    replacement_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGATCCTCAGCCCGCTGAGGATCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source, [_record("msd-retron-170", base_sequence), _unresolved_copy_record()])
    _write_genbank(replacement, [_record_with_narrow_foldback_annotation("msd-retron-170", replacement_sequence)])
    source_inputs = tmp_path / "source_inputs"

    source_manifest = write_variant_genbank_sources(
        source,
        output_dir=source_inputs,
        replacement_paths=(replacement,),
    )
    bundle = parse_msd_region_genbank_dir(Path(str(source_manifest["variant_source_dir"])))
    written = write_msd_region_record_bundle(bundle, output_dir=tmp_path / "bundle")

    assert source_manifest["variant_source_count"] == 1
    assert (source_inputs / "variants/pes-retron-170.gb").exists()
    variant_sources = yaml.safe_load((source_inputs / "variant_sources.yaml").read_text(encoding="utf-8"))
    assert variant_sources["source_policy"] == "per_variant_genbank_sources_are_authority"
    assert variant_sources["retired_migration_sources"][0]["source_name"] == "msd-regions.gb"
    assert variant_sources["retired_migration_sources"][0]["active_source"] is False
    assert bundle.source_kind == "variant_genbank_dir"
    assert bundle.included_record_count == 1
    assert bundle.skipped_records == ()
    manifest = yaml.safe_load(Path(written.manifest_path).read_text(encoding="utf-8"))
    assert manifest["source_policy"] == "per_variant_genbank_sources_are_authority"
    assert manifest["source_path"].endswith("source_inputs/variants")
    assert "msd-regions - all DNA RNA.gb" not in Path(written.manifest_path).read_text(encoding="utf-8")


def test_ingest_msd_regions_cli_uses_only_variant_source_dir_inputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "variants"
    source_dir.mkdir()
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    display_sequence = "GTCAGAAAAAACGGGTCCCTATCAGTGATAGAGAAGGCTCTCTATCACTGATAGGGAACAGACAGTAACTCAGA"
    _write_genbank(source_dir / "pes-retron-170.gb", [_record("msd-retron-170", display_sequence)])
    out_dir = tmp_path / "bundle"

    command_parameters = inspect.signature(ingest_msd_regions_command).parameters
    assert "source_dir" in command_parameters
    assert "source_genbank" not in command_parameters
    assert "replacement_genbank" not in command_parameters
    assert "write_variant_source_inputs" not in command_parameters

    result = RUNNER.invoke(
        app,
        [
            "ingest-msd-regions",
            "--source-dir",
            source_dir.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["source_kind"] == "variant_genbank_dir"
    assert payload["variant_source_input_count"] == 1
    assert payload["included_record_count"] == 1
    assert "migrated_variant_sources" not in payload


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


def test_oversized_stem_base_annotations_are_notes_not_compiler_bases(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    _write_genbank(source, [_record_with_oversized_stem_base_annotations("msd-retron-45")])
    record = parse_msd_region_genbank(source).records[0]

    assert record.primitive("stem_base_left").sequence_5to3 == "CGGG"
    assert record.primitive("stem_base_right").sequence_5to3 == "ACAG"
    assert {feature.role for feature in record.features} >= {
        "stem_base_left_annotated_span",
        "stem_base_right_annotated_span",
    }
    assert record.annotation_warnings == ()
    note_kinds = [note.kind for note in record.annotation_notes]
    assert note_kinds.count("stem_base_boundary_derived_from_extended_annotation") == 2

    resolved = resolve_msd_compiler_spec_payload(
        compiler_spec_payload_from_records((record,)),
        study_dir=_study_dir(),
        allow_non_ligatable_s0=True,
    )

    parsed = resolved.catalog.records[0]
    assert parsed.scar_nick.left_base == "CGGG"
    assert parsed.scar_nick.right_base == "ACAG"
    compiled = compile_msd_design_unit(
        parsed,
        payload_sequences=resolved.payload_sequences,
        cap_sequences=resolved.cap_sequences,
        payload_complement_sequences=resolved.payload_complement_sequences,
    )
    assert compiled.sequence_5to3 == record.msd_sequence_5to3


def test_compiler_cap_uses_full_interval_between_payload_and_complement(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    display_sequence = "GTCAGAAAAAACAAGTCCCTATCAGTGATAGAGATCCTCAGCCCGCTGAGGATCTCTATCACTGATAGGGACTCGACAGTAACTCAGA"
    _write_genbank(source, [_record_with_narrow_foldback_annotation("msd-retron-49", display_sequence)])
    record = parse_msd_region_genbank(source).records[0]

    spec_payload = compiler_spec_payload_from_records((record,))
    assert spec_payload["cap_sequences"]["C49_msd_region"]["sequence"] == "TCCTCAGCCCGCTGAGGA"
    assert record.annotation_warnings == ()
    assert record.annotation_notes[0].kind == "foldback_feature_boundary_granularity"
    foldback_pairing = {segment.segment: segment for segment in record.pairing_segments}["foldback_stem"]
    assert foldback_pairing.length_bp == 7
    assert foldback_pairing.pairing_status == "canonical_wc"

    resolved = resolve_msd_compiler_spec_payload(
        spec_payload,
        study_dir=_study_dir(),
        allow_non_ligatable_s0=True,
    )
    compiled = compile_msd_design_unit(
        resolved.catalog.records[0],
        payload_sequences=resolved.payload_sequences,
        cap_sequences=resolved.cap_sequences,
        payload_complement_sequences=resolved.payload_complement_sequences,
    )

    assert compiled.sequence_5to3 == display_sequence


def test_payload_binding_catalog_classifies_payload_family_trim_and_pwm_alignment(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    catalog_path = _payload_binding_catalog_path(tmp_path)
    retron26_payload = "TCCCTATCAGTGATAGAGA"
    tet_pwm_trim = "CTATATCTGATATAG"
    _write_genbank(
        source,
        [
            _record_with_payload_stem("msd-retron-26", retron26_payload),
            _record_with_payload_stem("msd-retron-195", tet_pwm_trim),
        ],
    )
    catalog = load_payload_binding_catalog(catalog_path)

    records = {
        record.variant_id: record for record in parse_msd_region_genbank(source, payload_catalog=catalog).records
    }

    retron26_site = records["retron26"].payload_binding_sites[0]
    assert retron26_site.payload_family_id == "tetO_ecoli_working"
    assert retron26_site.payload_member_id == "tetO_ecoli_working_w00_19"
    assert retron26_site.payload_class == "catalog_parent_payload"
    assert retron26_site.reference_comparisons[0].comparison_class == "identical"
    assert retron26_site.motif_alignments[0].consensus_score_fraction > 0.9

    retron195_site = records["retron195"].payload_binding_sites[0]
    assert retron195_site.payload_family_id == "tetr_pwm_elite"
    assert retron195_site.payload_member_id == "TetR_w02_17"
    assert retron195_site.payload_class == "catalog_trim_payload"
    comparison = retron195_site.reference_comparisons[0]
    assert comparison.reference_payload_id == "tetO_ecoli_working_w00_19"
    assert comparison.reference_span_0 == {"start": 2, "end": 17}
    assert comparison.mismatch_count == 4
    assert comparison.compared_nt == 15
    assert comparison.comparison_class == "moderate_difference"


def test_write_bundle_emits_payload_binding_sites_in_records_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "msd-regions.gb"
    catalog_path = _payload_binding_catalog_path(tmp_path)
    _write_genbank(source, [_record_with_payload_stem("msd-retron-195", "CTATATCTGATATAG")])
    catalog = load_payload_binding_catalog(catalog_path)
    bundle = parse_msd_region_genbank(source, payload_catalog=catalog)

    written = write_msd_region_record_bundle(bundle, output_dir=tmp_path / "bundle")

    variant_payload = yaml.safe_load(Path(written.variant_record_paths["retron195"]).read_text(encoding="utf-8"))
    manifest = yaml.safe_load(Path(written.manifest_path).read_text(encoding="utf-8"))
    assert variant_payload["payload_binding_sites"][0]["payload_member_id"] == "TetR_w02_17"
    assert manifest["records"][0]["payload_binding_sites"][0]["payload_class"] == "catalog_trim_payload"


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


def _record_with_payload_stem(record_id: str, payload_primary: str) -> SeqRecord:
    payload_complement = str(Seq(payload_primary).reverse_complement()).upper()
    display_sequence = f"GTCAGAAAAAACGGG{payload_primary}AGGC{payload_complement}ACAGACAGTAACTCAGA"
    source_sequence = str(Seq(display_sequence).reverse_complement()).upper()
    record = SeqRecord(Seq(source_sequence), id=record_id, name=record_id, description=record_id)
    length = len(source_sequence)
    payload_start = 15
    payload_end = payload_start + len(payload_primary)
    cap_start = payload_end
    cap_end = cap_start + 4
    complement_start = cap_end
    complement_end = complement_start + len(payload_complement)
    display_features = (
        ("5' Flanking", 0, 15, -1),
        ("Left Base", 11, 15, -1),
        ("tet operator", payload_start, payload_end, -1),
        ("WT loop", cap_start, cap_end, 1),
        ("tet operator [complement]", complement_start, complement_end, 1),
        ("Right base", complement_end, complement_end + 4, 1),
        ("3' Flanking", complement_end, length, 1),
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


def _payload_binding_catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "payload_binding_sites.yaml"
    meme_path = write_test_tetr_meme_pwm(tmp_path / "fixtures/tetR__westmann_tetr_mitomi__tetR.meme")
    path.write_text(
        """
contract: retron_payload_binding_site_catalog_v1
schema_version: 1
motif_models:
  tetr_westmann:
    source_ref: cruncher:westmann_tetr_mitomi:tetR
    meme_path: __MEME_PATH__
    congruence_threshold_fraction: 0.65
payload_families:
  tetO_ecoli_working:
    parent_payload_id: tetO_ecoli_working_w00_19
    motif_model_id: tetr_westmann
    primary_sequence_5to3: TCCCTATCAGTGATAGAGA
    members:
      tetO_ecoli_working_w00_19: {retained_parent_span_0: {start: 0, end: 19}}
  tetr_pwm_elite:
    parent_payload_id: TetR_w00_19
    motif_model_id: tetr_westmann
    primary_sequence_5to3: CTCTATATCTGATATAGAG
    members:
      TetR_w00_19: {retained_parent_span_0: {start: 0, end: 19}}
      TetR_w02_17: {retained_parent_span_0: {start: 2, end: 17}}
reference_payloads:
  - reference_payload_id: tetO_ecoli_working_w00_19
""".replace("__MEME_PATH__", meme_path.as_posix()).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _record_with_narrow_foldback_annotation(record_id: str, display_sequence: str) -> SeqRecord:
    source_sequence = str(Seq(display_sequence).reverse_complement()).upper()
    record = SeqRecord(Seq(source_sequence), id=record_id, name=record_id, description=record_id)
    length = len(source_sequence)
    display_features = (
        ("5' Flanking", 0, 15, -1),
        ("Left base", 11, 15, -1),
        ("tet operator", 15, 34, -1),
        ("Foldback", 35, 51, 1),
        ("Cap", 41, 45, 1),
        ("Foldback return", 45, 51, 1),
        ("tet operator [complement]", 52, 71, 1),
        ("Right base", 71, 75, 1),
        ("3' Flanking", 71, length, 1),
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
