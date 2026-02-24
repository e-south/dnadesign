"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/ingest/test_regulondb_adapter.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.ingest.adapters import regulondb as regulondb_module
from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.ingest.models import DatasetQuery, MotifQuery, SiteQuery
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
