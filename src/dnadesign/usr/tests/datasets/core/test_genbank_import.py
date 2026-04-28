"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/core/test_genbank_import.py

GenBank import tests for USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import AfterPosition, BeforePosition, CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from dnadesign.usr.src.contracts import SchemaError
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.genbank import BiopythonGenBankParser, import_genbank_manifest
from dnadesign.usr.src.genbank.importer import load_genbank_import_manifest
from dnadesign.usr.src.overlays import overlay_path
from dnadesign.usr.src.sequence_views import load_sequence_views


def _write_genbank_fixture(path: Path) -> None:
    record = SeqRecord(Seq("ACGT" * 50), id="NC_TEST", name="NC_TEST", description="Synthetic promoter record")
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.features = [
        SeqFeature(FeatureLocation(12, 52, strand=1), type="promoter", qualifiers={"label": ["pred. sulAp"]}),
        SeqFeature(FeatureLocation(20, 26, strand=1), type="misc_feature", qualifiers={"label": ["-35"]}),
        SeqFeature(FeatureLocation(32, 38, strand=1), type="misc_feature", qualifiers={"label": ["-10"]}),
        SeqFeature(FeatureLocation(60, 84, strand=-1), type="regulatory", qualifiers={"label": ["LexA"]}),
        SeqFeature(
            CompoundLocation(
                [
                    FeatureLocation(90, 96, strand=1),
                    FeatureLocation(100, 108, strand=1),
                ],
                operator="join",
            ),
            type="misc_feature",
            qualifiers={"label": ["joined_site"], "note": ["joined example"]},
        ),
        SeqFeature(
            FeatureLocation(BeforePosition(110), AfterPosition(118), strand=1),
            type="misc_feature",
            qualifiers={"label": ["fuzzy_site"]},
        ),
    ]
    with Path(path).open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def _write_negative_strand_fixture(path: Path) -> None:
    sequence = "A" * 60 + "ATGCCCAAAGTT" + "C" * 60
    record = SeqRecord(Seq(sequence), id="NC_NEG", name="NC_NEG", description="Negative strand feature record")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(60, 72, strand=-1),
            type="misc_feature",
            qualifiers={"label": ["neg_feature"]},
        )
    ]
    with Path(path).open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def _write_blank_qualifier_fixture(path: Path) -> None:
    record = SeqRecord(Seq("ACGT" * 10), id="NC_BLANK", name="NC_BLANK", description="Blank qualifier record")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(
            FeatureLocation(4, 12, strand=1),
            type="misc_feature",
            qualifiers={"label": ["blank_feature"], "note": [""]},
        )
    ]
    with Path(path).open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def _write_manifest(
    path: Path,
    *,
    output_dataset: str,
    source_file: Path,
    extract_features: list[dict] | None = None,
) -> None:
    payload = {
        "kind": "usr.genbank_import",
        "version": 1,
        "output_dataset": output_dataset,
        "on_conflict": "error",
        "copy_source_artifacts": True,
        "role_hint_rules": [
            {"match_label": "-35", "role_hint": "sigma70_minus35"},
            {"match_label": "-10", "role_hint": "sigma70_minus10"},
            {"match_any_label": ["LexA"], "role_hint": "TFBS"},
        ],
        "records": [
            {
                "source_file": source_file.name,
                "label": "sulAp_native",
                "aliases": ["sulAp", "solA_cipro_control"],
                "product_kind": "source_record",
            }
        ],
        "extract_features": extract_features or [],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_genbank_import_preserves_annotation_overlay_and_native_view(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(manifest_path, output_dataset="usr_reference_genbank_native", source_file=gb_path)

    result = import_genbank_manifest(root=root, manifest_path=manifest_path)
    dataset = Dataset(root, result.dataset)

    overlay_table = pq.read_table(overlay_path(dataset.dir, "seq_annot"))
    overlay_row = overlay_table.to_pylist()[0]
    views = load_sequence_views(dataset)

    assert result.source_records == 1
    assert overlay_row["seq_annot__format"] == "genbank"
    assert overlay_row["seq_annot__source_sha256"]
    assert overlay_row["seq_annot__topology"] == "linear"
    assert overlay_row["seq_annot__molecule_type"] == "DNA"
    assert overlay_row["seq_annot__source_artifact_uri"].startswith("_artifacts/genbank/")
    assert [feature["feature_order"] for feature in overlay_row["seq_annot__features"]] == [0, 1, 2, 3, 4, 5]
    assert overlay_row["seq_annot__features"][1]["role_hint"] == "sigma70_minus35"
    assert overlay_row["seq_annot__features"][2]["role_hint"] == "sigma70_minus10"
    assert overlay_row["seq_annot__features"][3]["role_hint"] == "TFBS"
    assert views[0].product_kind == "source_record"
    assert views[0].orientation == "unknown"


def test_biopython_parser_preserves_complement_join_and_fuzzy_locations(tmp_path: Path) -> None:
    gb_path = tmp_path / "locations.gb"
    _write_genbank_fixture(gb_path)

    parser = BiopythonGenBankParser()
    parsed = parser.parse_file(gb_path)
    features = parsed[0].features

    complement = next(feature for feature in features if feature.label == "LexA")
    compound = next(feature for feature in features if feature.label == "joined_site")
    fuzzy = next(feature for feature in features if feature.label == "fuzzy_site")

    assert complement.strand == -1
    assert complement.intervals_0[0].strand == -1
    assert compound.is_compound is True
    assert [(interval.start_0, interval.end_0) for interval in compound.intervals_0] == [(90, 96), (100, 108)]
    assert fuzzy.is_fuzzy is True
    assert fuzzy.confidence == "low"
    assert fuzzy.location_raw


def test_biopython_parser_preserves_blank_qualifier_values(tmp_path: Path) -> None:
    gb_path = tmp_path / "blank.gb"
    _write_blank_qualifier_fixture(gb_path)

    parsed = BiopythonGenBankParser().parse_file(gb_path)
    feature = parsed[0].features[0]

    assert {"key": "note", "value": ""} in [qualifier.model_dump() for qualifier in feature.qualifiers]


def test_genbank_import_rejects_path_style_dataset_ids(tmp_path: Path) -> None:
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(manifest_path, output_dataset="references/genbank_native", source_file=gb_path)

    with pytest.raises(ValueError, match="flat owner-first dataset id"):
        load_genbank_import_manifest(manifest_path)


def test_genbank_import_validates_feature_extractions_before_writing_dataset(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(
        manifest_path,
        output_dataset="usr_reference_genbank_native",
        source_file=gb_path,
        extract_features=[
            {
                "source_label": "sulAp_native",
                "selector": {"kind": "label", "label": "missing_feature"},
                "product_kind": "selected_region",
                "view_name": "missing_insert",
                "on_ambiguous": "error",
            }
        ],
    )

    with pytest.raises(SchemaError, match="matched 0 features"):
        import_genbank_manifest(root=root, manifest_path=manifest_path)

    assert not (root / "usr_reference_genbank_native").exists()


def test_genbank_import_is_idempotent_when_requested(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(manifest_path, output_dataset="usr_reference_genbank_native", source_file=gb_path)

    first = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    first["on_conflict"] = "idempotent"
    manifest_path.write_text(yaml.safe_dump(first, sort_keys=False), encoding="utf-8")

    import_genbank_manifest(root=root, manifest_path=manifest_path)
    import_genbank_manifest(root=root, manifest_path=manifest_path)

    dataset = Dataset(root, "usr_reference_genbank_native")
    overlay_table = pq.read_table(overlay_path(dataset.dir, "seq_annot"))

    assert dataset.head(10).shape[0] == 1
    assert overlay_table.num_rows == 1
    assert len(load_sequence_views(dataset)) == 1


def test_genbank_import_feature_extraction_is_idempotent_when_requested(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(
        manifest_path,
        output_dataset="usr_reference_genbank_native",
        source_file=gb_path,
        extract_features=[
            {
                "source_label": "sulAp_native",
                "selector": {"kind": "label", "label": "pred. sulAp"},
                "product_kind": "selected_region",
                "view_name": "sulAp_selected_region",
                "aliases": ["sulAp_insert"],
                "on_ambiguous": "error",
            }
        ],
    )
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["on_conflict"] = "idempotent"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    import_genbank_manifest(root=root, manifest_path=manifest_path)
    import_genbank_manifest(root=root, manifest_path=manifest_path)

    dataset = Dataset(root, "usr_reference_genbank_native")
    derived_table = pq.read_table(overlay_path(dataset.dir, "derived"))
    views = load_sequence_views(dataset)

    assert dataset.head(10).shape[0] == 2
    assert derived_table.num_rows == 1
    assert {view.product_kind for view in views} == {"source_record", "selected_region"}


def test_genbank_import_extracts_feature_with_parent_derivation(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "sulap.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_genbank_fixture(gb_path)
    _write_manifest(
        manifest_path,
        output_dataset="usr_reference_genbank_native",
        source_file=gb_path,
        extract_features=[
            {
                "source_label": "sulAp_native",
                "selector": {"kind": "label", "label": "pred. sulAp"},
                "product_kind": "selected_region",
                "view_name": "sulAp_selected_region",
                "aliases": ["sulAp_insert"],
                "on_ambiguous": "error",
            }
        ],
    )

    import_genbank_manifest(root=root, manifest_path=manifest_path)
    dataset = Dataset(root, "usr_reference_genbank_native")
    derived_table = pq.read_table(overlay_path(dataset.dir, "derived"))
    derived_row = derived_table.to_pylist()[0]
    views = load_sequence_views(dataset)

    assert dataset.head(10).shape[0] == 2
    assert derived_row["derived__operation"] == "extract_feature"
    assert derived_row["derived__product_kind"] == "selected_region"
    assert derived_row["derived__parent_dataset"] == dataset.name
    assert derived_row["derived__source_interval_start_0"] == 12
    assert derived_row["derived__source_interval_end_0"] == 52
    assert {view.product_kind for view in views} == {"source_record", "selected_region"}


def test_genbank_import_extracts_negative_strand_features_as_reverse_complement_views(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    gb_path = tmp_path / "negative.gb"
    manifest_path = tmp_path / "import.yaml"
    _write_negative_strand_fixture(gb_path)
    _write_manifest(
        manifest_path,
        output_dataset="usr_reference_genbank_native",
        source_file=gb_path,
        extract_features=[
            {
                "source_label": "sulAp_native",
                "selector": {"kind": "label", "label": "neg_feature"},
                "product_kind": "selected_region",
                "view_name": "neg_feature_insert",
                "on_ambiguous": "error",
            }
        ],
    )

    import_genbank_manifest(root=root, manifest_path=manifest_path)
    dataset = Dataset(root, "usr_reference_genbank_native")
    frame = dataset.head(10)
    extracted = frame[frame["sequence"] == "AACTTTGGGCAT"]
    derived_row = pq.read_table(overlay_path(dataset.dir, "derived")).to_pylist()[0]
    view = next(view for view in load_sequence_views(dataset) if view.product_kind == "selected_region")

    assert extracted.shape[0] == 1
    assert derived_row["derived__orientation"] == "reverse_complement"
    assert view.orientation == "reverse_complement"
