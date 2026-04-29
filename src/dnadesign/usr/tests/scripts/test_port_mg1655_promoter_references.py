from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from dnadesign.usr import Dataset
from dnadesign.usr.scripts import port_mg1655_promoter_references as port
from dnadesign.usr.src.contracts import compute_id
from dnadesign.usr.src.overlays import overlay_path
from dnadesign.usr.src.sequence_views import load_sequence_views

SPYP_INSERT = "ATGC" * 15
SOXSP_INSERT = "GCTA" * 15
J23105 = "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC"
J23103 = "CTGATAGCTAGCTCAGTCCTAGGGATTATGCTAGC"
W9 = "TTATCAAAAAGAGTATTGACATAAAGTCTAACCTATAGGAGTATTACAGCCATCGAGAGGGACACGGCGAA"
T7A1 = "TTATCAAAAAGAGTATTGACTTAAAGTCTAACCTATAGGATACTTACAGCCATCGAGAGGGACACGGCGAA"


def _write_promoter_genbank(
    path: Path,
    *,
    record_id: str,
    insert: str,
    broad_label: str,
    regulator_label: str,
) -> None:
    sequence = "G" * 12 + insert + "C" * 8
    record = SeqRecord(Seq(sequence), id=record_id, name=record_id, description=f"{record_id} source record")
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.features = [
        SeqFeature(FeatureLocation(12, 72, strand=1), type="misc_feature", qualifiers={"label": [broad_label]}),
        SeqFeature(FeatureLocation(20, 60, strand=1), type="promoter", qualifiers={"label": [broad_label.split()[0]]}),
        SeqFeature(FeatureLocation(28, 34, strand=1), type="misc_feature", qualifiers={"label": ["-35"]}),
        SeqFeature(FeatureLocation(49, 55, strand=1), type="misc_feature", qualifiers={"label": ["-10"]}),
        SeqFeature(FeatureLocation(38, 46, strand=1), type="regulatory", qualifiers={"label": [regulator_label]}),
        SeqFeature(FeatureLocation(12, 23, strand=1), type="primer_bind", qualifiers={"label": ["forward primer"]}),
        SeqFeature(FeatureLocation(62, 72, strand=-1), type="primer_bind", qualifiers={"label": ["reverse primer"]}),
    ]
    with path.open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def _write_synthetic_standard_genbank(
    path: Path,
    *,
    record_id: str,
    display_name: str,
    collection_id: str,
    role: str,
    sequence: str,
    strength_metric: str,
    strength_value: str,
    strength_reference: str,
    source_record: str,
) -> None:
    record = SeqRecord(Seq(sequence), id=record_id, name=record_id, description=f"{display_name} synthetic standard")
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.features = [
        SeqFeature(
            FeatureLocation(0, len(sequence), strand=1),
            type="source",
            qualifiers={"organism": ["synthetic DNA construct"], "note": [f"source_record={source_record}"]},
        ),
        SeqFeature(
            FeatureLocation(0, len(sequence), strand=1),
            type="promoter",
            qualifiers={
                "label": [display_name],
                "note": [
                    f"collection_id={collection_id}",
                    f"promoter_id={record_id}",
                    f"source_record={source_record}",
                    f"role={role}",
                    f"strength_metric={strength_metric}",
                    f"strength_value={strength_value}",
                    f"strength_reference={strength_reference}",
                    "fixture standard",
                ],
            },
        ),
        SeqFeature(
            FeatureLocation(0, 6, strand=1),
            type="misc_feature",
            qualifiers={"label": ["-35"], "note": [f"feature_sequence={sequence[:6]}"]},
        ),
        SeqFeature(
            FeatureLocation(23, 29, strand=1),
            type="misc_feature",
            qualifiers={"label": ["-10"], "note": [f"feature_sequence={sequence[23:29]}"]},
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        SeqIO.write(record, handle, "genbank")


def _write_synthetic_standards_fixture(root: Path) -> Path:
    standards_dir = root / "synthetic_promoter_standards"
    data_dir = standards_dir / "data"
    anderson_dir = standards_dir / "genbank" / "anderson_igem"
    t7_dir = standards_dir / "genbank" / "t7_w_collection"
    data_dir.mkdir(parents=True)
    anderson_dir.mkdir(parents=True)
    t7_dir.mkdir(parents=True)
    (data_dir / "promoters.csv").write_text(
        "\n".join(
            [
                "collection_id,promoter_id,display_name,role,sequence,strength_metric,strength_value,strength_reference,source_record,notes",
                (
                    "anderson_igem,BBa_J23105,J23105,constitutive_promoter,"
                    f"{J23105},relative_fluorescence_to_BBa_J23100,0.24,BBa_J23100=1.0,"
                    "anderson_igem_promoters_catalog,fixture Anderson standard"
                ),
                (
                    "anderson_igem,BBa_J23103,J23103,constitutive_promoter,"
                    f"{J23103},relative_fluorescence_to_BBa_J23100,0.01,BBa_J23100=1.0,"
                    "anderson_igem_promoters_catalog,retained twin sequence"
                ),
                (
                    "anderson_igem,BBa_J23112,J23112,constitutive_promoter,"
                    f"{J23103},relative_fluorescence_to_BBa_J23100,0.00,BBa_J23100=1.0,"
                    "anderson_igem_promoters_catalog,excluded conflicting twin sequence"
                ),
                (
                    "t7_w_collection,T7A1,T7A1,parent_promoter,"
                    f"{T7A1},not_reported,NA,NA,dunlop_lab_t7_w_collection,not reported"
                ),
                (
                    "t7_w_collection,W9,W9,library_member,"
                    f"{W9},ordinal_rank_weakest_to_strongest,9,W1=weakest; W9=strongest,"
                    "dunlop_lab_t7_w_collection,ordinal fixture standard"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "promoter_export_policy.csv").write_text(
        "\n".join(
            [
                "collection_id,promoter_id,export_to_genbank,exclusion_reason",
                "anderson_igem,BBa_J23105,true,",
                "anderson_igem,BBa_J23103,true,",
                "anderson_igem,BBa_J23112,false,duplicate_sequence_with_BBa_J23103_conflicting_strength",
                "t7_w_collection,T7A1,true,",
                "t7_w_collection,W9,true,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_synthetic_standard_genbank(
        anderson_dir / "BBa_J23105.gb",
        record_id="BBa_J23105",
        display_name="J23105",
        collection_id="anderson_igem",
        role="constitutive_promoter",
        sequence=J23105,
        strength_metric="relative_fluorescence_to_BBa_J23100",
        strength_value="0.24",
        strength_reference="BBa_J23100=1.0",
        source_record="anderson_igem_promoters_catalog",
    )
    _write_synthetic_standard_genbank(
        anderson_dir / "BBa_J23103.gb",
        record_id="BBa_J23103",
        display_name="J23103",
        collection_id="anderson_igem",
        role="constitutive_promoter",
        sequence=J23103,
        strength_metric="relative_fluorescence_to_BBa_J23100",
        strength_value="0.01",
        strength_reference="BBa_J23100=1.0",
        source_record="anderson_igem_promoters_catalog",
    )
    _write_synthetic_standard_genbank(
        t7_dir / "T7A1.gb",
        record_id="T7A1",
        display_name="T7A1",
        collection_id="t7_w_collection",
        role="parent_promoter",
        sequence=T7A1,
        strength_metric="not_reported",
        strength_value="NA",
        strength_reference="NA",
        source_record="dunlop_lab_t7_w_collection",
    )
    _write_synthetic_standard_genbank(
        t7_dir / "W9.gb",
        record_id="W9",
        display_name="W9",
        collection_id="t7_w_collection",
        role="library_member",
        sequence=W9,
        strength_metric="ordinal_rank_weakest_to_strongest",
        strength_value="9",
        strength_reference="W1=weakest; W9=strongest",
        source_record="dunlop_lab_t7_w_collection",
    )
    return standards_dir


def _write_legacy_reference_dataset(root: Path) -> None:
    dataset = Dataset(root, "usr_mg1655_promoter_controls")
    with dataset.write_session() as session:
        session.init(source="legacy-fixture", notes="legacy J23105 reference")
        session.import_rows(
            [
                {
                    "id": compute_id("dna", J23105),
                    "bio_type": "dna",
                    "sequence": J23105,
                    "alphabet": "dna_4",
                    "source": "legacy-fixture",
                }
            ],
            source="legacy-fixture",
        )


def _tmp_usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def test_plan_strips_primer_flanks_and_normalizes_reference_labels(tmp_path: Path) -> None:
    archive_dir = tmp_path / "MG1655_noncoding_set"
    archive_dir.mkdir()
    _write_promoter_genbank(
        archive_dir / "spyp-upstream-of-spy.gb",
        record_id="SPYP",
        insert=SPYP_INSERT,
        broad_label="spyp (upstream of spy)",
        regulator_label="CpxR+",
    )
    _write_promoter_genbank(
        archive_dir / "soxsp-upstream-soxs.gb",
        record_id="SOXSP",
        insert=SOXSP_INSERT,
        broad_label="soxSp (200 bp upstream CDS)",
        regulator_label="SoxR+",
    )

    plan = port.build_promoter_reference_plan(
        archive_dir=archive_dir,
        synthetic_standards_dir=None,
        legacy_usr_root=None,
        include_legacy_j23105=False,
    )

    assert [row.label for row in plan.promoters] == ["soxSp", "spyp"]
    by_label = {row.label: row for row in plan.promoters}
    assert by_label["spyp"].sequence == SPYP_INSERT.lower()
    assert by_label["spyp"].source_interval_start_0 == 12
    assert by_label["spyp"].source_interval_end_0 == 72
    labels = [feature["label"] for feature in by_label["spyp"].seq_annot_features]
    assert "forward primer" not in labels
    assert "reverse primer" not in labels
    minus35 = next(feature for feature in by_label["spyp"].seq_annot_features if feature["label"] == "-35")
    minus10 = next(feature for feature in by_label["spyp"].seq_annot_features if feature["label"] == "-10")
    cpxr = next(feature for feature in by_label["spyp"].seq_annot_features if feature["label"] == "CpxR+")
    assert (minus35["start_0"], minus35["end_0"], minus35["role_hint"]) == (16, 22, "sigma70_minus35")
    assert (minus10["start_0"], minus10["end_0"], minus10["role_hint"]) == (37, 43, "sigma70_minus10")
    assert (cpxr["start_0"], cpxr["end_0"], cpxr["role_hint"]) == (26, 34, "TFBS")


def test_write_promoter_reference_dataset_uses_projected_rows_and_modern_overlays(tmp_path: Path) -> None:
    archive_dir = tmp_path / "MG1655_noncoding_set"
    archive_dir.mkdir()
    _write_promoter_genbank(
        archive_dir / "spyp-upstream-of-spy.gb",
        record_id="SPYP",
        insert=SPYP_INSERT,
        broad_label="spyp (upstream of spy)",
        regulator_label="CpxR+",
    )
    usr_root = _tmp_usr_root(tmp_path)
    _write_legacy_reference_dataset(usr_root)
    plan = port.build_promoter_reference_plan(
        archive_dir=archive_dir,
        synthetic_standards_dir=None,
        legacy_usr_root=usr_root,
        include_legacy_j23105=True,
    )

    result = port.write_promoter_reference_dataset(
        plan,
        usr_root=usr_root,
        output_dataset="usr_promoter_references",
        expected_genbank_count=1,
        include_legacy_j23105=True,
    )

    assert result.rows_written == 2
    assert result.genbank_rows_written == 1
    dataset = Dataset(usr_root, "usr_promoter_references")
    records = pq.read_table(dataset.records_path).to_pylist()
    labels = {row["usr_label__primary"] for row in records}
    assert labels == {"J23105", "spyp"}
    assert {row["sequence"] for row in records} == {SPYP_INSERT.lower(), J23105.lower()}
    assert all(row["sequence"] != ("G" * 12 + SPYP_INSERT + "C" * 8).lower() for row in records)
    assert {row["construct_seed__label"] for row in records} == {"J23105", "spyp"}

    seq_annot = pq.read_table(overlay_path(dataset.dir, "seq_annot")).to_pylist()
    assert len(seq_annot) == 1
    assert seq_annot[0]["seq_annot__sequence_region_start_0"] == 0
    assert seq_annot[0]["seq_annot__sequence_region_end_0"] == 60
    projected_labels = [feature["label"] for feature in seq_annot[0]["seq_annot__features"]]
    assert projected_labels == ["spyp (upstream of spy)", "spyp", "-35", "-10", "CpxR+"]

    derived = pq.read_table(overlay_path(dataset.dir, "derived")).to_pylist()
    assert derived[0]["derived__product_kind"] == "selected_region"
    assert derived[0]["derived__operation"] == "project_genbank_upstream_feature"
    assert derived[0]["derived__source_interval_start_0"] == 12
    assert derived[0]["derived__source_interval_end_0"] == 72
    assert derived[0]["derived__analysis_only"] is False
    assert {feature["label"] for feature in derived[0]["derived__features_lost"]} == {
        "forward primer",
        "reverse primer",
    }

    views = load_sequence_views(dataset)
    by_view_name = {view.view_name: view for view in views}
    assert by_view_name["spyp"].product_kind == "selected_region"
    assert by_view_name["spyp"].recommended_pooling == "seq_mean"
    assert "spyP" in (by_view_name["spyp"].aliases or [])
    assert by_view_name["J23105"].product_kind == "selected_region"
    dataset.validate(strict=True)


def test_synthetic_standards_refresh_j23105_and_write_strength_overlay(tmp_path: Path) -> None:
    archive_dir = tmp_path / "MG1655_noncoding_set"
    archive_dir.mkdir()
    _write_promoter_genbank(
        archive_dir / "spyp-upstream-of-spy.gb",
        record_id="SPYP",
        insert=SPYP_INSERT,
        broad_label="spyp (upstream of spy)",
        regulator_label="CpxR+",
    )
    standards_dir = _write_synthetic_standards_fixture(tmp_path)
    usr_root = _tmp_usr_root(tmp_path)
    _write_legacy_reference_dataset(usr_root)

    plan = port.build_promoter_reference_plan(
        archive_dir=archive_dir,
        synthetic_standards_dir=standards_dir,
        legacy_usr_root=usr_root,
        include_legacy_j23105=True,
    )

    assert plan.legacy_references == ()
    assert sorted(row.label for row in plan.promoters) == ["J23103", "J23105", "T7A1", "W9", "spyp"]
    assert "J23112" not in {row.label for row in plan.promoters}
    by_label = {row.label: row for row in plan.promoters}
    assert by_label["J23105"].sequence == J23105.lower()
    assert by_label["J23105"].standard_metadata is not None
    assert by_label["J23105"].standard_metadata.strength_value == "0.24"
    assert by_label["W9"].standard_metadata is not None
    assert by_label["W9"].standard_metadata.strength_value_numeric == 9.0
    assert by_label["T7A1"].standard_metadata is not None
    assert by_label["T7A1"].standard_metadata.strength_value_numeric is None

    result = port.write_promoter_reference_dataset(
        plan,
        usr_root=usr_root,
        output_dataset="usr_promoter_references",
        expected_genbank_count=5,
        include_legacy_j23105=True,
    )

    assert result.rows_written == 5
    assert result.legacy_rows_written == 0
    assert result.promoter_standard_overlay_rows == 4
    dataset = Dataset(usr_root, "usr_promoter_references")
    records = pq.read_table(dataset.records_path).to_pylist()
    assert {row["usr_label__primary"] for row in records} == {"spyp", "J23105", "J23103", "T7A1", "W9"}

    seq_annot = pq.read_table(overlay_path(dataset.dir, "seq_annot")).to_pylist()
    assert len(seq_annot) == 5
    j23105_annot = next(row for row in seq_annot if row["id"] == compute_id("dna", J23105.lower()))
    j23105_labels = [feature["label"] for feature in j23105_annot["seq_annot__features"]]
    assert j23105_labels == ["J23105", "-35", "-10"]

    standard = pq.read_table(overlay_path(dataset.dir, "promoter_standard")).to_pylist()
    by_standard_label = {row["promoter_standard__display_name"]: row for row in standard}
    assert set(by_standard_label) == {"J23105", "J23103", "T7A1", "W9"}
    assert by_standard_label["J23105"]["promoter_standard__strength_metric"] == ("relative_fluorescence_to_BBa_J23100")
    assert by_standard_label["J23105"]["promoter_standard__strength_value"] == "0.24"
    assert by_standard_label["J23105"]["promoter_standard__strength_value_numeric"] == 0.24
    assert by_standard_label["W9"]["promoter_standard__strength_reference"] == "W1=weakest; W9=strongest"
    assert by_standard_label["T7A1"]["promoter_standard__strength_value"] == "NA"
    assert by_standard_label["T7A1"]["promoter_standard__strength_value_numeric"] is None

    views = load_sequence_views(dataset)
    by_view_name = {view.view_name: view for view in views}
    assert "BBa_J23105" in (by_view_name["J23105"].aliases or [])
    assert by_view_name["J23105"].source_interval_start_0 == 0
    assert by_view_name["J23105"].source_interval_end_0 == len(J23105)
    dataset.validate(strict=True)
