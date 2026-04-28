from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dnadesign.cruncher.ingest.promoters import (
    PromoterQuery,
    PromoterSchemaError,
    PromoterSourceFile,
    PromoterSourceInventory,
    build_promoter_source_inventory,
    discover_dnadesign_data_promoter_sources,
    export_dnadesign_data_promoter_superset,
    export_promoter_records,
    load_promoter_export,
    parse_promoter_source_file,
    parse_regulondb_promoter_payload,
    parse_regulondb_promoter_set_tsv,
    triage_promoter_sources,
)

FETCHED_AT = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def _promoter_payload(
    *,
    promoter_id: str = "RDBECOLIPMC00001",
    sequence: str = "aaTATAATggTTGACA",
    name: str = "cpxP",
) -> dict[str, object]:
    return {
        "_id": promoter_id,
        "name": name,
        "sequence": sequence,
        "strand": "+",
        "posTSS": 12345,
        "confidenceLevel": "Confirmed",
        "sigmaFactors": [
            {
                "_id": "RDBECOLISIG00070",
                "name": "RNA polymerase sigma factor RpoD",
                "abbreviatedName": "RpoD",
                "gene": {"_id": "RDBECOLIGNC03067", "name": "rpoD"},
                "evidence": ["EXP"],
                "citations": [{"pmid": "123456"}],
            }
        ],
        "boxes": [
            {
                "type": "minus_35",
                "sequence": "TTGACA",
                "leftEndPosition": 110,
                "rightEndPosition": 115,
                "strand": "+",
            },
            {
                "type": "minus_10",
                "sequence": "TATAAT",
                "leftEndPosition": 132,
                "rightEndPosition": 137,
                "strand": "+",
            },
        ],
        "regulatoryInteractions": [
            {
                "_id": "RI0001",
                "function": "activator",
                "mechanism": "transcription initiation",
                "confidenceLevel": "Confirmed",
                "regulator": {
                    "_id": "RDBECOLITFC00099",
                    "name": "CpxR response regulator",
                    "abbreviatedName": "CpxR",
                },
                "regulon": {"_id": "RDBECOLIRGN00099", "name": "CpxR regulon"},
                "regulatoryBindingSites": {
                    "_id": "BS0001",
                    "leftEndPosition": 120,
                    "rightEndPosition": 126,
                    "strand": "+",
                    "sequence": "ACGTACA",
                },
                "evidence": ["EXP"],
                "citations": [{"pmid": "234567"}],
            }
        ],
        "transcriptionUnits": [{"_id": "TU0001", "name": "cpxP-tu"}],
        "operon": {"_id": "OP0001", "name": "cpxP-op"},
        "firstGene": {"_id": "GENE0001", "name": "cpxP"},
    }


def test_parse_regulondb_promoter_payload_preserves_provenance_and_normalizes_features() -> None:
    record = parse_regulondb_promoter_payload(
        _promoter_payload(),
        source_release="14.5.0",
        source_route="operon_tu_promoter",
        query={"route": "fixture", "limit": 1},
        fetched_at=FETCHED_AT,
    )

    assert record.promoter_id == "RDBECOLIPMC00001"
    assert record.promoter_name == "cpxP"
    assert record.sequence == "AATATAATGGTTGACA"
    assert record.raw_sequence == "aaTATAATggTTGACA"
    assert record.sequence_case_policy == "uppercase_canonical_preserve_raw"
    assert record.tss_interval_0based == (12344, 12345)
    assert record.provenance.source_release == "14.5.0"
    assert record.provenance.source_route == "operon_tu_promoter"
    assert record.provenance.raw_payload_sha256
    assert record.provenance.query_sha256
    assert record.boxes[0].kind == "minus_35"
    assert record.boxes[0].interval_0based == (109, 115)
    assert record.sigma_affiliations[0].abbrev == "RpoD"
    assert record.regulatory_sites[0].regulator_abbrev == "CpxR"
    assert record.regulatory_sites[0].interval_0based == (119, 126)


def test_parse_regulondb_promoter_payload_fails_without_stable_promoter_id() -> None:
    payload = _promoter_payload()
    payload.pop("_id")

    with pytest.raises(PromoterSchemaError, match="promoter id"):
        parse_regulondb_promoter_payload(
            payload,
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={},
            fetched_at=FETCHED_AT,
        )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"sequence": None}, "missing required sequence"),
        ({"sequence": "ACGTNN"}, "strict A/C/G/T"),
        ({"score": "not-a-number"}, "score must be numeric"),
        ({"strand": "sideways"}, "Unrecognized RegulonDB strand"),
        ({"posTSS": 0}, "1-based position must be positive"),
        (
            {"boxes": [{"type": "minus_10", "sequence": "TATAAT", "leftEndPosition": 10, "rightEndPosition": 9}]},
            "Invalid RegulonDB 1-based inclusive interval",
        ),
    ],
)
def test_parse_regulondb_promoter_payload_fails_fast_on_invalid_required_fields(
    patch: dict[str, object], message: str
) -> None:
    payload = _promoter_payload()
    payload.update(patch)

    with pytest.raises(PromoterSchemaError, match=message):
        parse_regulondb_promoter_payload(
            payload,
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={},
            fetched_at=FETCHED_AT,
        )


def test_promoter_export_manifest_is_deterministic_and_reports_inventory(tmp_path) -> None:
    records = [
        parse_regulondb_promoter_payload(
            _promoter_payload(promoter_id="RDBECOLIPMC00001", name="cpxP"),
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        ),
        parse_regulondb_promoter_payload(
            _promoter_payload(promoter_id="RDBECOLIPMC00002", name="cpxP_alt"),
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        ),
    ]
    inventory = build_promoter_source_inventory(records, route_failure_count=1)

    export_dir = tmp_path / "export"
    manifest = export_promoter_records(
        records,
        export_dir,
        query=PromoterQuery(limit=2, include_relations=True),
        inventory=inventory,
        source_selection_status="primary_candidate",
    )
    first_manifest_bytes = (export_dir / "manifest.json").read_bytes()
    export_promoter_records(
        records,
        export_dir,
        query=PromoterQuery(limit=2, include_relations=True),
        inventory=inventory,
        source_selection_status="primary_candidate",
    )

    assert (export_dir / "manifest.json").read_bytes() == first_manifest_bytes
    assert manifest.complete is False
    assert manifest.source_inventory.promoter_row_count == 2
    assert manifest.source_inventory.sequence_present_rate == 1.0
    assert manifest.source_inventory.sigma_present_rate == 1.0
    assert manifest.source_inventory.duplicate_sequence_count == 1
    assert (export_dir / "promoters.jsonl").exists()
    assert (export_dir / "relations" / "promoter_boxes.jsonl").exists()

    loaded_manifest, loaded_records = load_promoter_export(export_dir)
    assert loaded_manifest.schema_version == manifest.schema_version
    assert [record.promoter_id for record in loaded_records] == ["RDBECOLIPMC00001", "RDBECOLIPMC00002"]


def test_parse_local_regulondb_promoter_set_tsv_preserves_release_table_and_row_refs(tmp_path) -> None:
    promoter_set = tmp_path / "PromoterSet.tsv"
    promoter_set.write_text(
        "\n".join(
            [
                "# release-pinned RegulonDB fixture",
                "\t".join(
                    [
                        "promoter_id",
                        "promoter_name",
                        "sequence",
                        "strand",
                        "posTSS",
                        "confidence_level",
                        "sigma_factor_abbrev",
                        "minus_35_sequence",
                        "minus_35_left",
                        "minus_35_right",
                        "minus_10_sequence",
                        "minus_10_left",
                        "minus_10_right",
                        "first_gene_name",
                        "tu_id",
                        "operon_id",
                        "evidence",
                        "citations",
                    ]
                ),
                "\t".join(
                    [
                        "RDBECOLIPMC00001",
                        "cpxP",
                        "aaTATAATggTTGACA",
                        "+",
                        "12345",
                        "Confirmed",
                        "RpoD",
                        "TTGACA",
                        "110",
                        "115",
                        "TATAAT",
                        "132",
                        "137",
                        "cpxP",
                        "TU0001",
                        "OP0001",
                        "EXP;IDA",
                        "123456;234567",
                    ]
                ),
                "\t".join(
                    [
                        "RDBECOLIPMC00002",
                        "hypA",
                        "ccggttaa",
                        "-",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "hypA",
                        "",
                        "",
                        "",
                        "",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = parse_regulondb_promoter_set_tsv(
        promoter_set,
        source_release="13.0",
        fetched_at=FETCHED_AT,
    )
    first, second = records

    assert len(records) == 2
    assert first.source_release == "13.0"
    assert first.source_route == "local_promoter_set"
    assert first.provenance.source_table == "PromoterSet.tsv"
    assert first.provenance.source_stratum == "local_release_pinned_curated"
    assert first.provenance.raw_payload_ref == f"{promoter_set}:3"
    assert first.sequence == "AATATAATGGTTGACA"
    assert first.raw_sequence == "aaTATAATggTTGACA"
    assert first.tss_interval_0based == (12344, 12345)
    assert [sigma.abbrev for sigma in first.sigma_affiliations] == ["RpoD"]
    assert [box.kind for box in first.boxes] == ["minus_35", "minus_10"]
    assert first.first_gene is not None
    assert first.first_gene.name == "cpxP"
    assert first.transcription_units[0].tu_id == "TU0001"
    assert first.operon is not None
    assert first.operon.operon_id == "OP0001"
    assert first.evidence == ("EXP", "IDA")
    assert first.citations == ("123456", "234567")
    assert second.tss_interval_0based is None
    assert second.sigma_affiliations == ()
    inventory = build_promoter_source_inventory(records)
    assert inventory.source_routes == ("local_promoter_set",)
    assert inventory.sigma_present_rate == 0.5
    assert inventory.box_annotation_rate == 0.5


def test_parse_local_regulondb_promoter_set_tsv_handles_numbered_release_headers(tmp_path) -> None:
    promoter_set = tmp_path / "PromoterSet.tsv"
    promoter_set.write_text(
        "\n".join(
            [
                "# RegulonDB Release: 13.0",
                "1)pmId\t2)pmName\t3)strand\t4)posTSS\t5)sigmaFactor\t6)pmSequence\t"
                "7)firstGeneName\t8)distToFirstGene\t9)pmEvidence\t10)addEvidence\t"
                "11)confidenceLevel\t12)pmids",
                "PM1\tspyp\treverse\t1825688\tsigma70\t"
                "atatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatTcg\t"
                "spy\t-63\t[EXP-IDA-TRANSCRIPTION-INIT-MAPPING:S]\t\tS\t14529615;9068658",
                "RDBECOLIPMC00002\tmissingp\tforward\t\t\tNone\tgeneA\t\t\t\tW\t",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = parse_regulondb_promoter_set_tsv(
        promoter_set,
        source_release="13.0",
        fetched_at=FETCHED_AT,
        skip_missing_sequence=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record.promoter_id == "PM1"
    assert record.promoter_name == "spyp"
    assert record.sequence.endswith("TCG")
    assert record.strand == "-"
    assert record.tss_interval_0based == (1825687, 1825688)
    assert record.sigma_affiliations[0].abbrev == "sigma70"
    assert record.first_gene is not None
    assert record.first_gene.name == "spy"
    assert record.evidence == ("[EXP-IDA-TRANSCRIPTION-INIT-MAPPING:S]",)
    assert record.citations == ("14529615", "9068658")


def test_discover_dnadesign_data_promoter_sources_uses_public_provider() -> None:
    source = PromoterSourceFile(
        source_id="regulondb_13_promoter_set",
        source="regulondb",
        release="13.0",
        path="RegulonDB_13/promoters/PromoterSet.tsv",
        table="PromoterSet.tsv",
        stratum="local_release_pinned_curated",
        role="curated_base",
        file_format="tsv",
        parser_hint="regulondb_promoter_set",
        creates_base_rows=True,
    )

    discovered = discover_dnadesign_data_promoter_sources(provider=lambda _root: [source])

    assert discovered == (source,)


def test_parse_promoter_source_file_dispatches_public_source_descriptor(tmp_path) -> None:
    promoter_set = tmp_path / "RegulonDB_11/promoters/PromoterSet.csv"
    promoter_set.parent.mkdir(parents=True)
    promoter_set.write_text(
        "\n".join(
            [
                "# release-pinned RegulonDB 11 fixture,,,,,,,,",
                "1)pmId,2)pmName,3)strand,4)posTSS,5)sigmaF,6)pmSequence,7)pmEvidence,8)addEvidence,9)confidenceLevel",
                "ECK125302590,C0293p,forward,1196711,Sigma70,tatgaattaccactccttacacccgctcaaatattgttaaattgccggttttgtatcaacTactc,[EXP|S],-,Strong",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = PromoterSourceFile(
        source_id="regulondb_11_promoter_set",
        source="regulondb",
        release="11.0",
        path="RegulonDB_11/promoters/PromoterSet.csv",
        table="PromoterSet.csv",
        stratum="historical_curated_release",
        role="historical_curated_comparison",
        file_format="csv",
        parser_hint="regulondb_promoter_set",
        creates_base_rows=True,
    )

    records = parse_promoter_source_file(source, data_root=tmp_path, fetched_at=FETCHED_AT)

    assert len(records) == 1
    assert records[0].source_release == "11.0"
    assert records[0].source_route == "regulondb_11_promoter_set"
    assert records[0].provenance.source_stratum == "historical_curated_release"
    assert records[0].promoter_id == "ECK125302590"
    assert records[0].sigma_affiliations[0].abbrev == "Sigma70"


def test_export_dnadesign_data_promoter_superset_records_all_sources_but_only_base_rows(tmp_path) -> None:
    promoter_set = tmp_path / "RegulonDB_13/promoters/PromoterSet.tsv"
    promoter_set.parent.mkdir(parents=True)
    promoter_set.write_text(
        "\n".join(
            [
                "1)pmId\t2)pmName\t3)strand\t4)posTSS\t5)sigmaFactor\t6)pmSequence\t"
                "7)firstGeneName\t8)pmEvidence\t9)confidenceLevel",
                "PM1\tspyp\treverse\t1825688\tsigma70\t"
                "atatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatTcg\t"
                "spy\t[EXP-IDA-TRANSCRIPTION-INIT-MAPPING:S]\tS",
                "PM2\tsequence_less\tforward\t\tsigma70\tNone\tgeneA\t[EXP]\tW",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_set = tmp_path / "RegulonDB_11/promoters/PromoterPredictionSet.csv"
    prediction_set.parent.mkdir(parents=True, exist_ok=True)
    prediction_set.write_text(
        "\n".join(
            [
                ";Computational Prediction of Promoters in the Escherichia coli genome.,,,",
                ";(1)Lend,(2)Rend,(3)Strand,(4)Gene,(5)Promoter_Name,(6)Sigma,(21)Sequence",
                "5234,5530,F,yaaX,yaaXp1,Sigma70,ggagaATAACAaccgccgttctcatcgagTAATCTccggAtatcg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    curated = PromoterSourceFile(
        source_id="regulondb_13_promoter_set",
        source="regulondb",
        release="13.0",
        path="RegulonDB_13/promoters/PromoterSet.tsv",
        table="PromoterSet.tsv",
        stratum="local_release_pinned_curated",
        role="curated_base",
        file_format="tsv",
        parser_hint="regulondb_promoter_set",
        creates_base_rows=True,
    )
    prediction = PromoterSourceFile(
        source_id="regulondb_11_prediction_set",
        source="regulondb",
        release="11.0",
        path="RegulonDB_11/promoters/PromoterPredictionSet.csv",
        table="PromoterPredictionSet.csv",
        stratum="historical_computational_prediction",
        role="prediction_overlay",
        file_format="csv",
        parser_hint="regulondb_promoter_prediction_set",
        creates_base_rows=False,
    )

    export_dir = tmp_path / "superset_export"
    manifest = export_dnadesign_data_promoter_superset(
        export_dir,
        data_root=tmp_path,
        provider=lambda _root: [curated, prediction],
        fetched_at=FETCHED_AT,
    )

    assert manifest.source_selection_status == "dnadesign_data_superset"
    assert manifest.record_count == 1
    loaded_manifest, records = load_promoter_export(export_dir)
    assert loaded_manifest.record_count == 1
    assert [record.source_route for record in records] == ["regulondb_13_promoter_set"]
    assert loaded_manifest.artifacts["skipped_source_rows"] == "skipped_source_rows.jsonl"
    source_files = (export_dir / "source_files.json").read_text(encoding="utf-8")
    assert "regulondb_13_promoter_set" in source_files
    assert "regulondb_11_prediction_set" in source_files
    source_inventory = json.loads((export_dir / "source_file_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["source_id"]: row for row in source_inventory}
    assert by_id["regulondb_13_promoter_set"]["parsed_record_count"] == 1
    assert by_id["regulondb_13_promoter_set"]["skipped_record_count"] == 1
    assert by_id["regulondb_11_prediction_set"]["skipped_reason"] == "non_base_source_deferred"
    skipped_rows = [
        json.loads(line)
        for line in (export_dir / "skipped_source_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(skipped_rows) == 1
    skipped = skipped_rows[0]
    assert skipped["source_release"] == "13.0"
    assert skipped["source_route"] == "regulondb_13_promoter_set"
    assert skipped["source_table"] == "PromoterSet.tsv"
    assert skipped["source_stratum"] == "local_release_pinned_curated"
    assert skipped["promoter_id"] == "RDBECOLIPMC00002"
    assert skipped["promoter_name"] == "sequence_less"
    assert skipped["raw_sequence"] == "None"
    assert skipped["skip_reason"] == "missing_sequence"
    assert skipped["source_row_ref"].endswith("PromoterSet.tsv:3")
    assert skipped["raw_payload_sha256"]
    assert skipped["query_sha256"]


def test_export_dnadesign_data_promoter_superset_accounts_for_real_missing_sequence_rows(tmp_path) -> None:
    data_root = Path("/Users/Shockwing/Dropbox/projects/phd/dnadesign-data")
    if not (data_root / "RegulonDB_13/promoters/PromoterSet.tsv").exists():
        pytest.skip("sibling dnadesign-data checkout is not available")
    sys.path.insert(0, str(data_root / "src"))
    from dnadesign_data.regulatory_parts import iter_promoter_source_files

    export_dir = tmp_path / "superset_export"
    manifest = export_dnadesign_data_promoter_superset(
        export_dir,
        data_root=data_root,
        provider=iter_promoter_source_files,
        fetched_at=FETCHED_AT,
    )
    skipped_rows = [
        json.loads(line)
        for line in (export_dir / "skipped_source_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    skipped_by_route = Counter(row["source_route"] for row in skipped_rows)
    inventory = json.loads((export_dir / "source_file_inventory.json").read_text(encoding="utf-8"))
    inventory_by_id = {row["source_id"]: row for row in inventory}

    assert manifest.record_count == 7914
    assert skipped_by_route == {
        "regulondb_13_promoter_set": 92,
        "regulondb_11_promoter_set": 92,
    }
    assert inventory_by_id["regulondb_13_promoter_set"]["parsed_record_count"] == 3960
    assert inventory_by_id["regulondb_13_promoter_set"]["skipped_record_count"] == 92
    assert inventory_by_id["regulondb_11_promoter_set"]["parsed_record_count"] == 3954
    assert inventory_by_id["regulondb_11_promoter_set"]["skipped_record_count"] == 92


def test_export_dnadesign_data_promoter_superset_fails_without_base_records(tmp_path) -> None:
    prediction = PromoterSourceFile(
        source_id="regulondb_11_prediction_set",
        source="regulondb",
        release="11.0",
        path="RegulonDB_11/promoters/PromoterPredictionSet.csv",
        table="PromoterPredictionSet.csv",
        stratum="historical_computational_prediction",
        role="prediction_overlay",
        file_format="csv",
        parser_hint="regulondb_promoter_prediction_set",
        creates_base_rows=False,
    )

    with pytest.raises(PromoterSchemaError, match="No base-row-capable"):
        export_dnadesign_data_promoter_superset(
            tmp_path / "superset_export",
            data_root=tmp_path,
            provider=lambda _root: [prediction],
            fetched_at=FETCHED_AT,
        )


def test_promoter_source_triage_uses_local_fallback_when_live_inventory_is_incomplete() -> None:
    live_inventory = PromoterSourceInventory(
        source_releases=("14.5.0",),
        source_routes=("operon_tu_promoter",),
        promoter_row_count=20,
        sequence_present_rate=0.8,
        promoter_id_present_rate=1.0,
        tss_present_rate=0.7,
        sigma_present_rate=0.4,
        box_annotation_rate=0.2,
        confidence_present_rate=0.5,
        regulatory_context_rate=0.5,
        duplicate_sequence_count=0,
        conflict_count=0,
        route_failure_count=1,
    )
    local_inventory = PromoterSourceInventory(
        source_releases=("13.0",),
        source_routes=("local_promoter_set",),
        promoter_row_count=200,
        sequence_present_rate=1.0,
        promoter_id_present_rate=1.0,
        tss_present_rate=0.9,
        sigma_present_rate=0.6,
        box_annotation_rate=0.6,
        confidence_present_rate=0.8,
        regulatory_context_rate=0.0,
        duplicate_sequence_count=4,
        conflict_count=0,
        route_failure_count=0,
    )

    report = triage_promoter_sources(
        {
            "live_regulondb_graphql": live_inventory,
            "local_regulondb_13_promoter_set": local_inventory,
        },
        preferred_sources=("live_regulondb_graphql", "local_regulondb_13_promoter_set"),
    )

    assert report.blocked is False
    assert report.primary_source == "local_regulondb_13_promoter_set"
    assert report.supplemental_sources == ("live_regulondb_graphql",)
    assert report.candidate_status["live_regulondb_graphql"].startswith("blocked:")
    assert report.candidate_status["local_regulondb_13_promoter_set"] == "eligible"
