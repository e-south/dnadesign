from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.cruncher.ingest.promoters import (
    PromoterQuery,
    SkippedPromoterSourceRow,
    export_dnadesign_data_promoter_superset,
    export_promoter_records,
    parse_regulondb_promoter_payload,
)
from dnadesign.usr.scripts import create_regulondb_native_promoters as native
from dnadesign.usr.src.contracts import SchemaError, compute_id

FETCHED_AT = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)


def _tmp_usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def _payload(
    *,
    promoter_id: str,
    sequence: str = "aaTATAATggTTGACA",
    name: str | None = None,
    sigma: str | None = "RpoD",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "_id": promoter_id,
        "name": name or promoter_id,
        "sequence": sequence,
        "strand": "+",
        "posTSS": 12345,
        "confidenceLevel": "Confirmed",
        "boxes": [
            {"type": "minus_35", "sequence": "TTGACA", "leftEndPosition": 110, "rightEndPosition": 115},
            {"type": "minus_10", "sequence": "TATAAT", "leftEndPosition": 132, "rightEndPosition": 137},
        ],
        "regulatoryInteractions": [
            {
                "_id": "RI0001",
                "function": "activator",
                "regulator": {"_id": "REG0001", "name": "CpxR response regulator", "abbreviatedName": "CpxR"},
                "regulon": {"_id": "RGN0001", "name": "CpxR regulon"},
            }
        ],
        "transcriptionUnits": [{"_id": "TU0001", "name": "fixture-tu"}],
        "operon": {"_id": "OP0001", "name": "fixture-op"},
        "firstGene": {"_id": "GENE0001", "name": "fixture-gene"},
    }
    if sigma:
        payload["sigmaFactors"] = [{"_id": "SIG0001", "name": sigma, "abbreviatedName": sigma}]
    return payload


def _write_export(tmp_path: Path, payloads: list[dict[str, object]]) -> Path:
    records = [
        parse_regulondb_promoter_payload(
            payload,
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        )
        for payload in payloads
    ]
    export_dir = tmp_path / "cruncher_export"
    export_promoter_records(records, export_dir, query=PromoterQuery(limit=len(records)))
    return export_dir


def test_build_import_plan_collapses_duplicate_sequences_and_keeps_sig35_out(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", name="cpxP"),
            _payload(promoter_id="PM0002", name="cpxP_alt"),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    expected_sequence = "aatataatggttgaca"
    assert len(plan.base_rows) == 1
    assert plan.base_rows[0]["id"] == compute_id("dna", expected_sequence)
    assert plan.base_rows[0]["bio_type"] == "dna"
    assert plan.base_rows[0]["sequence"] == expected_sequence
    assert "sig35_variant" not in plan.base_rows[0]
    assert len(plan.relation_rows["promoter_aliases"]) == 2
    overlay = plan.regulondb_overlay_rows[0]
    assert overlay["regulondb__primary_promoter_id"] is None
    assert overlay["regulondb__promoter_alias_count"] == 2
    assert overlay["regulondb__source_strata_set"] == ["curated"]
    assert overlay["regulondb__sigma_factor_set"] == ["sigma70"]
    assert overlay["regulondb__has_minus10_box"] is True
    assert overlay["regulondb__has_minus35_box"] is True
    assert overlay["regulondb__regulator_composition"] == "activator"
    assert len(plan.relation_rows["source_rows"]) == 2


def test_build_import_plan_deduplicates_aliases_but_preserves_source_rows(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", name="cpxP"),
            _payload(promoter_id="PM0001", name="cpxP"),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    assert len(plan.base_rows) == 1
    assert len(plan.relation_rows["promoter_aliases"]) == 1
    assert len(plan.relation_rows["source_rows"]) == 2
    assert plan.regulondb_overlay_rows[0]["regulondb__promoter_alias_count"] == 1
    assert plan.validation_report["duplicate_promoter_alias_collapses"] == 1


def test_build_import_plan_deduplicates_repeated_sigma_affiliations(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", name="cpxP", sigma="RpoD"),
            _payload(promoter_id="PM0001", name="cpxP", sigma="RpoD"),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    assert len(plan.relation_rows["sigma_affiliations"]) == 1
    assert plan.relation_rows["sigma_affiliations"][0]["sigma_canonical_label"] == "sigma70"
    assert plan.validation_report["duplicate_sigma_affiliation_collapses"] == 1
    assert plan.validation_report["duplicate_relation_row_counts"]["sigma_affiliations"] == 0


def test_build_import_plan_canonicalizes_sigma_summary_without_losing_raw_labels(tmp_path: Path) -> None:
    records = [
        parse_regulondb_promoter_payload(
            _payload(promoter_id="PM0001", sigma="Sigma70"),
            source_release="11.0",
            source_route="regulondb_11_promoter_set",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        ),
        parse_regulondb_promoter_payload(
            _payload(promoter_id="PM0002", sigma="sigma70"),
            source_release="13.0",
            source_route="regulondb_13_promoter_set",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        ),
    ]
    export_dir = tmp_path / "cruncher_export"
    export_promoter_records(records, export_dir, query=PromoterQuery(limit=2))
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    assert plan.regulondb_overlay_rows[0]["regulondb__sigma_factor_set"] == ["sigma70"]
    assert plan.regulondb_overlay_rows[0]["regulondb__sigma_factor_count"] == 1
    assert {row["sigma_abbrev"] for row in plan.relation_rows["sigma_affiliations"]} == {"Sigma70", "sigma70"}
    assert {row["sigma_canonical_label"] for row in plan.relation_rows["sigma_affiliations"]} == {"sigma70"}


def test_build_import_plan_reports_sequence_and_metadata_fidelity(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", name="cpxP", sequence="aaTATAATggTTGACA", sigma="RpoD"),
            _payload(promoter_id="PM0002", name="cpxP_alt", sequence="CCCCGGGGTTTTAAAA", sigma=None),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    report = plan.validation_report
    assert report["base_duplicate_sequence_count"] == 0
    assert report["base_invalid_sequence_count"] == 0
    assert report["base_length_mismatch_count"] == 0
    assert report["required_overlay_metadata_null_counts"] == {}
    assert report["base_row_required_metadata"] == ["sigma"]
    assert report["included_export_record_count"] == 1
    assert report["metadata_excluded_base_row_count"] == 1
    assert report["metadata_excluded_source_row_count"] == 1
    assert report["metadata_excluded_source_rows_by_reason"] == {"missing_sigma": 1}
    assert report["sigma_missing_base_row_count"] == 0
    assert report["sigma_missing_alias_row_count"] == 0
    assert len(plan.base_rows) == 1
    assert len(plan.relation_rows["excluded_source_rows"]) == 1
    excluded = plan.relation_rows["excluded_source_rows"][0]
    assert excluded["promoter_id"] == "PM0002"
    assert excluded["exclusion_reason"] == "missing_sigma"


def test_build_import_plan_reports_fuzzy_name_collision_candidates(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", name="lacZp", sequence="AAAACCCCGGGGTTTT"),
            _payload(promoter_id="PM0002", name="lacZ-p", sequence="TTTTGGGGCCCCAAAA"),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    assert plan.validation_report["fuzzy_promoter_name_collision_count"] == 1
    collision = plan.validation_report["fuzzy_promoter_name_collisions"][0]
    assert {collision["left_name"], collision["right_name"]} == {"lacZp", "lacZ-p"}
    assert collision["kind"] == "normalized_name_match"


def test_build_import_plan_carries_skipped_source_rows_as_dataset_sidecar(tmp_path: Path) -> None:
    records = [
        parse_regulondb_promoter_payload(
            _payload(promoter_id="PM0001"),
            source_release="13.0",
            source_route="regulondb_13_promoter_set",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
            source_table="PromoterSet.tsv",
            raw_payload_ref="PromoterSet.tsv:2",
            source_stratum="local_release_pinned_curated",
        )
    ]
    skipped = SkippedPromoterSourceRow(
        source="regulondb",
        source_release="13.0",
        source_release_date=None,
        source_route="regulondb_13_promoter_set",
        source_table="PromoterSet.tsv",
        source_stratum="local_release_pinned_curated",
        promoter_id="PM_MISSING_SEQUENCE",
        promoter_name="sequence_less",
        raw_sequence="None",
        skip_reason="missing_sequence",
        source_row_ref="PromoterSet.tsv:3",
        raw_payload_sha256="raw-sha",
        query_sha256="query-sha",
    )
    export_dir = tmp_path / "cruncher_export"
    export_promoter_records(
        records,
        export_dir,
        query=PromoterQuery(limit=1),
        skipped_source_rows=[skipped],
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)

    assert len(plan.base_rows) == 1
    assert len(plan.relation_rows["skipped_source_rows"]) == 1
    row = plan.relation_rows["skipped_source_rows"][0]
    assert row["promoter_id"] == "PM_MISSING_SEQUENCE"
    assert row["promoter_name"] == "sequence_less"
    assert row["skip_reason"] == "missing_sequence"
    assert row["source_row_ref"] == "PromoterSet.tsv:3"
    assert plan.validation_report["skipped_source_row_count"] == 1
    assert plan.validation_report["skipped_source_rows_by_reason"] == {"missing_sequence": 1}


def test_dry_run_does_not_create_usr_dataset(tmp_path: Path) -> None:
    export_dir = _write_export(tmp_path, [_payload(promoter_id="PM0001")])
    usr_root = _tmp_usr_root(tmp_path)

    payload = native.run_import(export_dir=export_dir, usr_root=usr_root, write=False)

    assert payload["write"] is False
    assert payload["plan"]["dataset"] == "usr_regulondb_native_promoters"
    assert not (usr_root / "usr_regulondb_native_promoters" / "records.parquet").exists()


def test_same_promoter_id_conflicting_sequence_fails(tmp_path: Path) -> None:
    export_dir = _write_export(
        tmp_path,
        [
            _payload(promoter_id="PM0001", sequence="AACCGGTT"),
            _payload(promoter_id="PM0001", sequence="AACCGGTA"),
        ],
    )
    usr_root = _tmp_usr_root(tmp_path)

    with pytest.raises(SchemaError, match="conflicting canonical sequence"):
        native.build_import_plan(export_dir=export_dir, usr_root=usr_root)


def test_write_mode_creates_dataset_relations_and_event_log(tmp_path: Path) -> None:
    records = [
        parse_regulondb_promoter_payload(
            _payload(promoter_id="PM0001"),
            source_release="14.5.0",
            source_route="operon_tu_promoter",
            query={"route": "fixture"},
            fetched_at=FETCHED_AT,
        )
    ]
    skipped = SkippedPromoterSourceRow(
        source="regulondb",
        source_release="13.0",
        source_release_date=None,
        source_route="regulondb_13_promoter_set",
        source_table="PromoterSet.tsv",
        source_stratum="local_release_pinned_curated",
        promoter_id="PM_MISSING_SEQUENCE",
        promoter_name="sequence_less",
        raw_sequence="None",
        skip_reason="missing_sequence",
        source_row_ref="PromoterSet.tsv:3",
        raw_payload_sha256="raw-sha",
        query_sha256="query-sha",
    )
    export_dir = tmp_path / "cruncher_export"
    export_promoter_records(records, export_dir, query=PromoterQuery(limit=1), skipped_source_rows=[skipped])
    usr_root = _tmp_usr_root(tmp_path)

    payload = native.run_import(export_dir=export_dir, usr_root=usr_root, write=True)

    dataset_dir = usr_root / "usr_regulondb_native_promoters"
    assert payload["write"] is True
    assert (dataset_dir / "records.parquet").exists()
    assert (dataset_dir / "_views" / "sequence_views.parquet").exists()
    assert (dataset_dir / "_views" / "view_semantics.parquet").exists()
    assert (dataset_dir / "_relations" / "promoter_aliases.parquet").exists()
    assert (dataset_dir / "_relations" / "excluded_source_rows.parquet").exists()
    assert (dataset_dir / "_relations" / "skipped_source_rows.parquet").exists()
    assert (dataset_dir / ".events.log").exists()
    records = pq.read_table(dataset_dir / "records.parquet").to_pylist()
    views = pq.read_table(dataset_dir / "_views" / "sequence_views.parquet").to_pylist()
    semantics = pq.read_table(dataset_dir / "_views" / "view_semantics.parquet").to_pylist()
    skipped_rows = pq.read_table(dataset_dir / "_relations" / "skipped_source_rows.parquet").to_pylist()
    assert "sig35_variant" not in records[0]
    assert records[0]["regulondb__primary_promoter_id"] == "PM0001"
    assert payload["result"]["sequence_view_rows"] == 1
    assert payload["result"]["view_semantics_rows"] == 1
    assert views[0]["sequence_id"] == records[0]["id"]
    assert views[0]["product_kind"] == "source_record"
    assert views[0]["context_kind"] == "native_reference"
    assert views[0]["orientation"] == "unknown"
    assert views[0]["recommended_pooling"] == "seq_mean"
    assert semantics[0]["view_id"] == views[0]["view_id"]
    assert semantics[0]["source_family"] == "regulondb_native_promoter"
    assert semantics[0]["selection_basis"] == "regulondb_curated_promoter_sequence_with_sigma"
    assert "regulondb_native_promoter_panel" in semantics[0]["view_collections"]
    assert pq.read_table(dataset_dir / "_relations" / "excluded_source_rows.parquet").num_rows == 0
    assert skipped_rows[0]["promoter_id"] == "PM_MISSING_SEQUENCE"
    assert skipped_rows[0]["skip_reason"] == "missing_sequence"
    assert "usr_id" not in skipped_rows[0]
    assert not (dataset_dir / "._relations.tmp").exists()


def test_write_import_plan_rejects_orphan_relation_rows(tmp_path: Path) -> None:
    export_dir = _write_export(tmp_path, [_payload(promoter_id="PM0001")])
    usr_root = _tmp_usr_root(tmp_path)
    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)
    plan.relation_rows["promoter_aliases"][0]["usr_id"] = "missing_usr_id"

    with pytest.raises(SchemaError, match="invalid usr_id references"):
        native.write_import_plan(plan, usr_root=usr_root)

    assert not (usr_root / "usr_regulondb_native_promoters").exists()


def test_real_dnadesign_data_superset_dry_run_fidelity_contract(tmp_path: Path) -> None:
    data_root = Path("/Users/Shockwing/Dropbox/projects/phd/dnadesign-data")
    if not (data_root / "RegulonDB_13/promoters/PromoterSet.tsv").exists():
        pytest.skip("sibling dnadesign-data checkout is not available")
    sys.path.insert(0, str(data_root / "src"))
    from dnadesign_data.regulatory_parts import iter_promoter_source_files

    export_dir = tmp_path / "superset_export"
    export_dnadesign_data_promoter_superset(
        export_dir,
        data_root=data_root,
        provider=iter_promoter_source_files,
        fetched_at=FETCHED_AT,
    )
    usr_root = _tmp_usr_root(tmp_path)

    plan = native.build_import_plan(export_dir=export_dir, usr_root=usr_root)
    report = plan.validation_report

    assert report["base_row_count"] == 3182
    assert report["included_export_record_count"] == 6629
    assert report["base_duplicate_sequence_count"] == 0
    assert report["base_invalid_sequence_count"] == 0
    assert report["base_length_mismatch_count"] == 0
    assert report["base_row_required_metadata"] == ["sigma"]
    assert report["metadata_excluded_base_row_count"] == 645
    assert report["metadata_excluded_source_row_count"] == 1285
    assert report["metadata_excluded_source_rows_by_reason"] == {"missing_sigma": 1285}
    assert report["required_overlay_metadata_null_counts"] == {}
    assert report["orphan_relation_row_count"] == 0
    assert report["missing_relation_usr_id_count"] == 0
    assert report["duplicate_relation_row_counts"]["promoter_aliases"] == 0
    assert report["duplicate_relation_row_counts"]["sigma_affiliations"] == 0
    assert report["sigma_empty_label_row_count"] == 0
    assert report["sigma_factor_set_count_mismatch"] == 0
    assert report["sigma_missing_base_row_count"] == 0
    assert report["sigma_missing_alias_row_count"] == 0
    assert report["sigma_label_counts"] == {
        "sigma19": 1,
        "sigma24": 1042,
        "sigma28": 290,
        "sigma32": 624,
        "sigma38": 479,
        "sigma54": 198,
        "sigma70": 3991,
    }
    assert report["fuzzy_promoter_name_collision_count"] == 16
    assert report["skipped_source_row_count"] == 184
