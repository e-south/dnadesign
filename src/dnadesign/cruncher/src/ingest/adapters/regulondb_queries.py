"""GraphQL documents used by the RegulonDB adapter."""

from __future__ import annotations

_REGULON_LIST_QUERY = """
query ($search: String, $limit: Int, $page: Int) {
  getRegulonBy(search: $search, limit: $limit, page: $page) {
    data {
      _id
      regulator { name abbreviatedName synonyms }
      organism { name }
    }
  }
}
"""

_REGULON_ALL_QUERY = """
query ($limit: Int, $page: Int) {
  getAllRegulon(limit: $limit, page: $page) {
    data {
      _id
      regulator { name abbreviatedName synonyms }
      organism { name }
    }
  }
}
"""

_REGULON_DETAIL_QUERY = """
query ($search: String, $limit: Int, $page: Int) {
  getRegulonBy(search: $search, limit: $limit, page: $page) {
    data {
      _id
      regulator { name abbreviatedName synonyms }
      regulatoryInteractions {
        _id
        regulatoryBindingSites {
          _id
          leftEndPosition
          rightEndPosition
          strand
          sequence
        }
      }
      aligmentMatrix {
        matrix
        aligment
        consensus
      }
    }
  }
}
"""

_HT_SOURCES_QUERY = """
query {
  listAllHTSources
}
"""

_HT_DATASET_TYPES_QUERY = """
query {
  listAllDatasetTypes
}
"""

_HT_DATASETS_QUERY = """
query($datasetType: String!, $source: String!) {
  getDatasetsWithMetadata(datasetType: $datasetType, source: $source) {
    datasets {
      _id
      collectionData { type source }
      objectsTested { name abbreviatedName synonyms }
      referenceGenome
      assemblyGenomeId
    }
  }
}
"""

_HT_TF_BINDING_QUERY = """
query($datasetId: String!, $limit: Int, $page: Int) {
  getAllTFBindingOfDataset(datasetId: $datasetId, limit: $limit, page: $page) {
    _id
    chromosome
    chrLeftPosition
    chrRightPosition
    strand
    sequence
    score
    datasetIds
    peakId
  }
}
"""

_HT_PEAKS_QUERY = """
query($datasetId: String!, $limit: Int, $page: Int) {
  getAllPeaksOfDataset(datasetId: $datasetId, limit: $limit, page: $page) {
    _id
    name
    chromosome
    peakLeftPosition
    peakRightPosition
    score
    siteIds
    datasetIds
  }
}
"""

_DATABASE_INFO_QUERY = """
query {
  getDatabaseInfo {
    regulonDBVersion
    releaseDate
    genomeVersion
    route
  }
}
"""

_OPERON_PROMOTER_QUERY = """
query($limit: Int, $page: Int) {
  getAllOperon(limit: $limit, page: $page) {
    pagination {
      currentPage
      lastPage
      totalResults
      hasNextPage
      limit
    }
    data {
      _id
      operon {
        _id
        name
        strand
      }
      organism {
        _id
        name
      }
      transcriptionUnits {
        _id
        name
        confidenceLevel
        firstGene {
          _id
          name
          distanceToPromoter
        }
        promoter {
          _id
          name
          sequence
          score
          confidenceLevel
          transcriptionStartSite {
            leftEndPosition
            rightEndPosition
            range
            type
          }
          bindsSigmaFactor {
            _id
            name
            abbreviatedName
            citations {
              evidence {
                code
                name
                type
              }
              publication {
                _id
                pmid
                citation
              }
            }
          }
          boxes {
            leftEndPosition
            rightEndPosition
            sequence
            type
          }
          citations {
            evidence {
              code
              name
              type
            }
            publication {
              _id
              pmid
              citation
            }
          }
          additiveEvidences {
            category
            code
            type
          }
          regulatorBindingSites {
            regulator {
              _id
              name
              abbreviatedName
            }
            function
            mechanism
            regulatoryInteractions {
              _id
              confidenceLevel
              function
              mechanism
              regulatorySite {
                _id
                leftEndPosition
                rightEndPosition
                sequence
              }
              citations {
                evidence {
                  code
                  name
                  type
                }
                publication {
                  _id
                  pmid
                  citation
                }
              }
              additiveEvidences {
                category
                code
                type
              }
            }
          }
        }
      }
    }
  }
}
"""
