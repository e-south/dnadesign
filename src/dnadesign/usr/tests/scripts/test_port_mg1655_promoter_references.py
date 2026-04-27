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
    assert derived[0]["derived__product_kind"] == "biological_insert"
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
    assert by_view_name["spyp"].product_kind == "biological_insert"
    assert by_view_name["spyp"].recommended_pooling == "seq_mean"
    assert "spyP" in (by_view_name["spyp"].aliases or [])
    assert by_view_name["J23105"].product_kind == "biological_insert"
    dataset.validate(strict=True)
