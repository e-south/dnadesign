"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/fixtures/regulondb_payloads.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

LEXA_ID = "RDBECOLITFC00214"
CPXR_ID = "RDBECOLITFC00099"

LEXA_DATASET_ID = "RHTECOLIBSD02444"
CPXR_DATASET_ID = "RHTECOLIBSD02736"


def _matrix_payload(rows: list[list[float]]) -> str:
    return json.dumps(rows)


LEXA_MATRIX = _matrix_payload(
    [
        [0.3, 0.2, 0.3, 0.2],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.2, 0.3],
        [0.25, 0.2, 0.35, 0.2],
    ]
)

CPXR_MATRIX = _matrix_payload(
    [
        [0.2, 0.3, 0.3, 0.2],
        [0.3, 0.2, 0.2, 0.3],
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.2, 0.2, 0.2],
        [0.2, 0.3, 0.1, 0.4],
        [0.3, 0.2, 0.3, 0.2],
    ]
)

REGULON_LIST = {
    "getRegulonBy": {
        "data": [
            {
                "_id": LEXA_ID,
                "regulator": {
                    "name": "DNA-binding transcriptional repressor LexA",
                    "abbreviatedName": "LexA",
                    "synonyms": ["Tsl", "Spr", "UmuA", "ExrA", "LexA"],
                },
                "organism": {"name": "Escherichia coli K-12"},
            },
            {
                "_id": CPXR_ID,
                "regulator": {
                    "name": "DNA-binding transcriptional dual regulator CpxR",
                    "abbreviatedName": "CpxR",
                    "synonyms": ["YiiA", "CpxR", "CpxR response regulator"],
                },
                "organism": {"name": "Escherichia coli K-12"},
            },
        ]
    }
}

REGULON_DETAIL = {
    LEXA_ID: {
        "getRegulonBy": {
            "data": [
                {
                    "_id": LEXA_ID,
                    "regulator": {
                        "name": "DNA-binding transcriptional repressor LexA",
                        "abbreviatedName": "LexA",
                        "synonyms": ["Tsl", "Spr", "UmuA", "ExrA", "LexA"],
                    },
                    "organism": {"name": "Escherichia coli K-12"},
                    "regulatoryInteractions": [
                        {
                            "_id": "RI1",
                            "regulatoryBindingSites": {
                                "_id": "BS1",
                                "leftEndPosition": 100,
                                "rightEndPosition": 105,
                                "strand": "+",
                                "sequence": "aaACGTACtt",
                            },
                        },
                        {
                            "_id": "RI2",
                            "regulatoryBindingSites": {
                                "_id": "BS2",
                                "leftEndPosition": 200,
                                "rightEndPosition": 205,
                                "strand": "-",
                                "sequence": "ccACGTACgg",
                            },
                        },
                    ],
                    "aligmentMatrix": {"matrix": LEXA_MATRIX, "aligment": None, "consensus": "ACGTAC"},
                }
            ]
        }
    },
    CPXR_ID: {
        "getRegulonBy": {
            "data": [
                {
                    "_id": CPXR_ID,
                    "regulator": {
                        "name": "DNA-binding transcriptional dual regulator CpxR",
                        "abbreviatedName": "CpxR",
                        "synonyms": ["YiiA", "CpxR", "CpxR response regulator"],
                    },
                    "organism": {"name": "Escherichia coli K-12"},
                    "regulatoryInteractions": [
                        {
                            "_id": "RI3",
                            "regulatoryBindingSites": {
                                "_id": "BS3",
                                "leftEndPosition": 500,
                                "rightEndPosition": 505,
                                "strand": "+",
                                "sequence": "ttACGTTCaa",
                            },
                        },
                        {
                            "_id": "RI4",
                            "regulatoryBindingSites": {
                                "_id": "BS4",
                                "leftEndPosition": 600,
                                "rightEndPosition": 605,
                                "strand": "+",
                                "sequence": "ggACGTTCcc",
                            },
                        },
                    ],
                    "aligmentMatrix": {"matrix": CPXR_MATRIX, "aligment": None, "consensus": "ACGTTC"},
                }
            ]
        }
    },
}

HT_SOURCES = {"listAllHTSources": ["BAUMGART"]}
HT_DATASET_TYPES = {"listAllDatasetTypes": ["TFBINDING"]}

HT_DATASETS = {
    "BAUMGART": {
        "getDatasetsWithMetadata": {
            "datasets": [
                {
                    "_id": LEXA_DATASET_ID,
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
                {
                    "_id": CPXR_DATASET_ID,
                    "collectionData": {"type": "TFBINDING", "source": "BAUMGART"},
                    "objectsTested": [
                        {
                            "name": "DNA-binding transcriptional dual regulator CpxR",
                            "abbreviatedName": "CpxR",
                            "synonyms": ["CpxR"],
                        }
                    ],
                    "referenceGenome": "U00096.3",
                    "assemblyGenomeId": None,
                },
            ]
        }
    }
}

HT_TF_BINDING = {
    LEXA_DATASET_ID: {
        "getAllTFBindingOfDataset": [
            {
                "_id": "HT1",
                "chromosome": "chr",
                "chrLeftPosition": 1000,
                "chrRightPosition": 1005,
                "strand": "+",
                "sequence": "ACGTAC",
                "score": 12.3,
                "datasetIds": [LEXA_DATASET_ID],
                "peakId": "P1",
            },
            {
                "_id": "HT2",
                "chromosome": "chr",
                "chrLeftPosition": 1100,
                "chrRightPosition": 1105,
                "strand": "-",
                "sequence": "ACGTAC",
                "score": 10.1,
                "datasetIds": [LEXA_DATASET_ID],
                "peakId": "P2",
            },
        ]
    },
    CPXR_DATASET_ID: {"getAllTFBindingOfDataset": []},
}

HT_PEAKS = {
    CPXR_DATASET_ID: {
        "getAllPeaksOfDataset": [
            {
                "_id": "PK1",
                "name": "peak_1",
                "chromosome": "chr",
                "peakLeftPosition": 2000,
                "peakRightPosition": 2006,
                "score": 5.0,
                "siteIds": [],
                "datasetIds": [CPXR_DATASET_ID],
            },
            {
                "_id": "PK2",
                "name": "peak_2",
                "chromosome": "chr",
                "peakLeftPosition": 3000,
                "peakRightPosition": 3006,
                "score": 4.5,
                "siteIds": [],
                "datasetIds": [CPXR_DATASET_ID],
            },
        ]
    }
}


def regulon_list_for_search(search: str | None) -> dict:
    if not search:
        return REGULON_LIST
    search_norm = search.lower()
    filtered = []
    for item in REGULON_LIST["getRegulonBy"]["data"]:
        regulator = item.get("regulator") or {}
        candidates = [regulator.get("abbreviatedName"), regulator.get("name")] + list(regulator.get("synonyms") or [])
        if any(search_norm in str(cand).lower() for cand in candidates if cand):
            filtered.append(item)
    return {"getRegulonBy": {"data": filtered}}
