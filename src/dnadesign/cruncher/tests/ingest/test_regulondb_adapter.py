"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/ingest/test_regulondb_adapter.py

Regression tests for RegulonDB adapter Cruncher ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.ingest.adapters import regulondb as regulondb_module
from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.ingest.models import DatasetQuery, MotifQuery, SiteQuery
from dnadesign.cruncher.ingest.promoters import PromoterQuery, PromoterSchemaError, summarize_promoter_collection
from dnadesign.cruncher.tests.fixtures.regulondb_payloads import (
    CPXR_ID,
    HT_DATASET_TYPES,
    HT_DATASETS,
    HT_PEAKS,
    HT_SOURCES,
    HT_TF_BINDING,
    LEXA_DATASET_ID,
    LEXA_ID,
    REGULON_DETAIL,
    REGULON_LIST,
    regulon_list_for_search,
)


def _fixture_transport(query: str, variables: dict) -> dict:
    if "listAllHTSources" in query:
        return HT_SOURCES
    if "listAllDatasetTypes" in query:
        return HT_DATASET_TYPES
    if "getDatasetsWithMetadata" in query:
        source = variables.get("source")
        return HT_DATASETS.get(source, {"getDatasetsWithMetadata": {"datasets": []}})
    if "getAllTFBindingOfDataset" in query:
        dataset_id = variables.get("datasetId")
        page = variables.get("page", 0)
        if page:
            return {"getAllTFBindingOfDataset": []}
        return HT_TF_BINDING.get(dataset_id, {"getAllTFBindingOfDataset": []})
    if "getAllPeaksOfDataset" in query:
        dataset_id = variables.get("datasetId")
        page = variables.get("page", 0)
        if page:
            return {"getAllPeaksOfDataset": []}
        return HT_PEAKS.get(dataset_id, {"getAllPeaksOfDataset": []})
    if "getAllRegulon" in query:
        return {"getAllRegulon": {"data": REGULON_LIST["getRegulonBy"]["data"]}}
    if "regulatoryInteractions" in query:
        search = (variables.get("search") or "").lower()
        if search in {LEXA_ID.lower(), "lexa"}:
            return REGULON_DETAIL[LEXA_ID]
        if search in {CPXR_ID.lower(), "cpxr"}:
            return REGULON_DETAIL[CPXR_ID]
        return {"getRegulonBy": {"data": []}}
    return regulon_list_for_search(variables.get("search"))


def _fixture_transport_no_ht(query: str, variables: dict) -> dict:
    if "listAllHTSources" in query:
        return {"listAllHTSources": []}
    return _fixture_transport(query, variables)


def _promoter_operon_item(
    *,
    operon_id: str,
    promoter_id: str,
    promoter_name: str,
    sequence: str,
    sigma: dict | None,
    tss: int,
) -> dict:
    return {
        "_id": operon_id,
        "operon": {"_id": operon_id, "name": f"{promoter_name}-operon", "strand": "forward"},
        "organism": {"_id": "RDBECOLIORC00001", "name": "Escherichia coli K-12"},
        "transcriptionUnits": [
            {
                "_id": f"{operon_id}-tu",
                "name": f"{promoter_name}-tu",
                "confidenceLevel": "S",
                "firstGene": {"_id": f"{operon_id}-gene", "name": promoter_name, "distanceToPromoter": 0},
                "promoter": {
                    "_id": promoter_id,
                    "name": promoter_name,
                    "sequence": sequence,
                    "score": None,
                    "confidenceLevel": "S",
                    "transcriptionStartSite": {
                        "leftEndPosition": tss,
                        "rightEndPosition": tss,
                        "range": 1,
                        "type": "TSS",
                    },
                    "bindsSigmaFactor": sigma or {"_id": None, "name": None, "abbreviatedName": None, "citations": []},
                    "boxes": [
                        {
                            "leftEndPosition": tss - 35,
                            "rightEndPosition": tss - 30,
                            "sequence": "TTGACA",
                            "type": "-35",
                        }
                    ],
                    "citations": [
                        {
                            "evidence": {"code": "EXP-IDA-TRANSCRIPTION-INIT-MAPPING", "name": "mapping", "type": "S"},
                            "publication": {"_id": "PUB1", "pmid": "32061118", "citation": "fixture"},
                        }
                    ],
                    "additiveEvidences": [{"category": "experimental", "code": "EXP", "type": "S"}],
                    "regulatorBindingSites": [],
                },
            }
        ],
    }


def test_regulondb_iter_promoters_paginates_operon_route_and_summarizes_sigma_counts() -> None:
    calls: list[tuple[str, int | None, int | None]] = []

    def transport(query: str, variables: dict) -> dict:
        if "getDatabaseInfo" in query:
            return {
                "getDatabaseInfo": [
                    {
                        "regulonDBVersion": "14.5.0",
                        "releaseDate": "2026-01-28",
                        "genomeVersion": "NC_000913.3",
                        "route": "/dump/rdb-mongo/regulondb_14.5.zip",
                    }
                ]
            }
        if "getAllOperon" in query:
            page = int(variables.get("page") or 0)
            limit = int(variables.get("limit") or 0)
            calls.append(("getAllOperon", limit, page))
            sigma70 = {
                "_id": "RDBECOLISGC00070",
                "name": "RNA polymerase sigma factor RpoD",
                "abbreviatedName": "RpoD",
                "citations": [],
            }
            sigma38 = {
                "_id": "RDBECOLISGC00038",
                "name": "RNA polymerase sigma factor RpoS",
                "abbreviatedName": "RpoS",
                "citations": [],
            }
            pages = {
                0: [
                    _promoter_operon_item(
                        operon_id="RDBECOLIOPC00001",
                        promoter_id="RDBECOLIPMC00001",
                        promoter_name="cpxPp",
                        sequence="aaTATAATggTTGACA",
                        sigma=sigma70,
                        tss=12345,
                    ),
                    _promoter_operon_item(
                        operon_id="RDBECOLIOPC00002",
                        promoter_id="RDBECOLIPMC00002",
                        promoter_name="unknownp",
                        sequence="ccggttaa",
                        sigma=None,
                        tss=22345,
                    ),
                ],
                1: [
                    _promoter_operon_item(
                        operon_id="RDBECOLIOPC00003",
                        promoter_id="RDBECOLIPMC00003",
                        promoter_name="osmYp",
                        sequence="ggttccaa",
                        sigma=sigma38,
                        tss=32345,
                    )
                ],
            }
            return {
                "getAllOperon": {
                    "pagination": {"currentPage": page, "hasNextPage": page == 0, "lastPage": 1},
                    "data": pages.get(page, []),
                }
            }
        raise AssertionError(f"unexpected RegulonDB query: {query}")

    adapter = RegulonDBAdapter(transport=transport)
    records = list(adapter.iter_promoters(PromoterQuery(limit=3, page_size=2)))

    assert calls == [("getAllOperon", 2, 0), ("getAllOperon", 2, 1)]
    assert [record.promoter_id for record in records] == [
        "RDBECOLIPMC00001",
        "RDBECOLIPMC00002",
        "RDBECOLIPMC00003",
    ]
    assert records[0].source_route == "operon_tu_promoter"
    assert records[0].source_release == "14.5.0"
    assert records[0].provenance.source_release_date == "2026-01-28"
    assert records[0].genome_accession == "NC_000913.3"
    assert records[0].tss_interval_0based == (12344, 12345)
    assert records[0].sigma_affiliations[0].abbrev == "RpoD"
    assert records[1].sigma_affiliations == ()
    assert records[1].provenance.raw_payload_ref == "getAllOperon[page=0].data[1].transcriptionUnits[0].promoter"

    summary = summarize_promoter_collection(records)
    assert summary.record_count == 3
    assert summary.unique_promoter_count == 3
    assert summary.missing_sigma_count == 1
    assert summary.sigma_factor_counts == {"RpoD": 1, "RpoS": 1}


def test_regulondb_list_promoters_reports_missing_sequence_without_normalizing() -> None:
    def transport(query: str, variables: dict) -> dict:
        if "getDatabaseInfo" in query:
            return {
                "getDatabaseInfo": [
                    {
                        "regulonDBVersion": "14.5.0",
                        "releaseDate": "2026-01-28",
                        "genomeVersion": "NC_000913.3",
                        "route": "/dump/rdb-mongo/regulondb_14.5.zip",
                    }
                ]
            }
        if "getAllOperon" in query:
            return {
                "getAllOperon": {
                    "pagination": {"currentPage": 0, "hasNextPage": False, "lastPage": 0},
                    "data": [
                        _promoter_operon_item(
                            operon_id="RDBECOLIOPC00004",
                            promoter_id="RDBECOLIPMC00004",
                            promoter_name="missing-sequence",
                            sequence=None,
                            sigma={
                                "_id": "RDBECOLISGC00070",
                                "name": "RNA polymerase sigma factor RpoD",
                                "abbreviatedName": "RpoD",
                                "citations": [],
                            },
                            tss=42345,
                        )
                    ],
                }
            }
        raise AssertionError(f"unexpected RegulonDB query: {query}")

    adapter = RegulonDBAdapter(transport=transport)
    descriptors = list(adapter.list_promoters(PromoterQuery(limit=1, page_size=1)))

    assert descriptors[0].promoter_id == "RDBECOLIPMC00004"
    assert descriptors[0].sequence_present is False
    assert descriptors[0].sigma_present is True
    assert descriptors[0].sigma_factor_labels == ("RpoD",)
    with pytest.raises(PromoterSchemaError, match="sequence"):
        list(adapter.iter_promoters(PromoterQuery(limit=1, page_size=1)))


def test_regulondb_promoter_enumeration_fails_on_partial_pagination() -> None:
    def transport(query: str, variables: dict) -> dict:
        if "getDatabaseInfo" in query:
            return {
                "getDatabaseInfo": [
                    {
                        "regulonDBVersion": "14.5.0",
                        "releaseDate": "2026-01-28",
                        "genomeVersion": "NC_000913.3",
                        "route": "/dump/rdb-mongo/regulondb_14.5.zip",
                    }
                ]
            }
        if "getAllOperon" in query:
            page = int(variables.get("page") or 0)
            if page == 0:
                data = [
                    _promoter_operon_item(
                        operon_id="RDBECOLIOPC00005",
                        promoter_id="RDBECOLIPMC00005",
                        promoter_name="first-page",
                        sequence="acgt",
                        sigma=None,
                        tss=52345,
                    )
                ]
            else:
                data = []
            return {
                "getAllOperon": {
                    "pagination": {"currentPage": page, "hasNextPage": True, "lastPage": 2},
                    "data": data,
                }
            }
        raise AssertionError(f"unexpected RegulonDB query: {query}")

    adapter = RegulonDBAdapter(transport=transport)

    with pytest.raises(RuntimeError, match="partial promoter pagination"):
        list(adapter.list_promoters(PromoterQuery(page_size=1)))


def test_list_motifs_returns_descriptors() -> None:
    adapter = RegulonDBAdapter(transport=_fixture_transport)
    results = adapter.list_motifs(MotifQuery(tf_name=None))
    assert results
    ids = {rec.motif_id for rec in results}
    assert LEXA_ID in ids
    assert CPXR_ID in ids


def test_get_motif_alignment_matrix() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(motif_matrix_source="alignment"),
        transport=_fixture_transport,
    )
    record = adapter.get_motif(LEXA_ID)
    assert record.descriptor.tf_name == "LexA"
    assert record.descriptor.length == 6
    assert record.descriptor.tags.get("matrix_source") == "alignment"


def test_get_motif_sites_matrix() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(motif_matrix_source="sites", min_sites_for_pwm=2),
        transport=_fixture_transport,
    )
    record = adapter.get_motif(LEXA_ID)
    assert record.descriptor.tags.get("matrix_source") == "sites"
    assert record.descriptor.tags.get("site_count") == "2"


def test_list_sites_curated() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=False),
        transport=_fixture_transport,
    )
    sites = list(adapter.list_sites(SiteQuery(tf_name="lexA")))
    assert len(sites) == 2
    assert sites[0].sequence == "ACGTAC"


def test_list_sites_curated_with_missing_ht_raises() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=True),
        transport=_fixture_transport_no_ht,
    )
    with pytest.raises(RuntimeError) as excinfo:
        list(adapter.list_sites(SiteQuery(tf_name="lexA")))
    assert "RegulonDB returned no HT sources" in str(excinfo.value)


def test_list_sites_ht() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=False, ht_sites=True),
        transport=_fixture_transport,
    )
    sites = list(adapter.list_sites(SiteQuery(tf_name="lexA", limit=10)))
    assert len(sites) == 2
    assert sites[0].motif_ref.startswith("regulondb:dataset:")


def test_list_datasets_for_tf() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=False, ht_sites=True),
        transport=_fixture_transport,
    )
    datasets = adapter.list_datasets(DatasetQuery(tf_name="lexA"))
    assert datasets
    assert any(ds.dataset_id == LEXA_DATASET_ID for ds in datasets)


def test_list_datasets_filters_row_level_dataset_source() -> None:
    def transport(query: str, variables: dict) -> dict:
        if "listAllHTSources" in query:
            return {"listAllHTSources": ["GALAGAN"]}
        if "listAllDatasetTypes" in query:
            return {"listAllDatasetTypes": ["TFBINDING"]}
        if "getDatasetsWithMetadata" in query:
            return {
                "getDatasetsWithMetadata": {
                    "datasets": [
                        {
                            "_id": "RHTECOLIBSD03022",
                            "collectionData": {"type": "TFBINDING", "source": "GALAGAN"},
                            "objectsTested": [
                                {
                                    "name": "DNA-binding transcriptional repressor LexA",
                                    "abbreviatedName": "LexA",
                                    "synonyms": ["LexA"],
                                }
                            ],
                            "referenceGenome": "U00096.3",
                            "assemblyGenomeId": None,
                        },
                        {
                            "_id": "RHTECOLIBSD02444",
                            "collectionData": {"type": "TFBINDING", "source": "BAUMGART"},
                            "objectsTested": [
                                {
                                    "name": "DNA-binding transcriptional repressor LexA",
                                    "abbreviatedName": "LexA",
                                    "synonyms": ["LexA"],
                                }
                            ],
                            "referenceGenome": "U00096.3",
                            "assemblyGenomeId": None,
                        },
                    ]
                }
            }
        return {"getRegulonBy": {"data": []}}

    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=False, ht_sites=True),
        transport=transport,
    )
    datasets = adapter.list_datasets(DatasetQuery(tf_name="lexA", dataset_source="GALAGAN"))
    assert [item.dataset_id for item in datasets] == ["RHTECOLIBSD03022"]
    assert all(item.dataset_source == "GALAGAN" for item in datasets)


def test_list_datasets_invalid_type_raises() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=False, ht_sites=True, ht_dataset_type="BADTYPE"),
        transport=_fixture_transport,
    )
    try:
        adapter.list_datasets(DatasetQuery())
    except ValueError as exc:
        assert "Unknown RegulonDB dataset type" in str(exc)
    else:
        raise AssertionError("Expected invalid dataset type to raise ValueError.")


def test_list_sites_ht_peaks() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=False, ht_sites=True, ht_binding_mode="peaks"),
        transport=_fixture_transport,
    )
    sites = list(adapter.list_sites(SiteQuery(tf_name="cpxR", limit=10)))
    assert len(sites) == 2
    assert sites[0].sequence is None
    assert sites[0].coordinate is not None
    assert sites[0].coordinate.contig == "U00096.3"
    assert sites[0].coordinate.assembly == "U00096.3"
    assert sites[0].provenance.tags.get("record_kind") == "ht_peak"


def test_list_sites_ht_without_records_raises_even_with_curated() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=True),
        transport=_fixture_transport,
    )
    with pytest.raises(ValueError) as excinfo:
        list(adapter.list_sites(SiteQuery(tf_name="cpxR")))
    assert "No HT binding-site records returned for TF CpxR" in str(excinfo.value)


def test_list_sites_curated_and_ht_with_limit_requires_explicit_mode() -> None:
    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=True),
        transport=_fixture_transport,
    )
    with pytest.raises(ValueError) as excinfo:
        list(adapter.list_sites(SiteQuery(tf_name="lexA", limit=1)))
    assert "explicit source mode" in str(excinfo.value)


def test_list_motifs_requires_inventory_shape() -> None:
    def transport(query: str, variables: dict) -> dict:
        if "getAllRegulon" in query:
            return {"getAllRegulon": {"data": None}}
        return {"getRegulonBy": {"data": []}}

    adapter = RegulonDBAdapter(transport=transport)
    with pytest.raises(RuntimeError) as excinfo:
        adapter.list_motifs(MotifQuery(tf_name=None))
    assert "getAllRegulon" in str(excinfo.value)


def test_graphql_length_error_includes_hint(monkeypatch) -> None:
    def fake_request_json(*_args, **_kwargs):
        return {"errors": [{"message": "Cannot read properties of undefined (reading 'length')"}]}

    monkeypatch.setattr(regulondb_module, "request_json", fake_request_json)
    adapter = RegulonDBAdapter(RegulonDBAdapterConfig())
    with pytest.raises(RuntimeError) as excinfo:
        adapter._post_graphql("query { getAllRegulon(limit: 1, page: 0) { data { _id } } }", {})
    assert "Remote inventory may be unavailable" in str(excinfo.value)
