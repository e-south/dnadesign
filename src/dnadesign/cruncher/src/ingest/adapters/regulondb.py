"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/adapters/regulondb.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import re
import ssl
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set
from urllib.error import HTTPError, URLError

import certifi

from dnadesign.cruncher.ingest.http_client import HttpRetryPolicy, request_json
from dnadesign.cruncher.ingest.models import (
    DatasetDescriptor,
    DatasetQuery,
    GenomicInterval,
    MotifDescriptor,
    MotifQuery,
    MotifRecord,
    OrganismRef,
    Provenance,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import (
    build_motif_record,
    compute_pwm_from_sites,
    normalize_site_sequence,
)

_REGULONDB_URL = "https://regulondb.ccg.unam.mx/graphql"
_REGULONDB_INTERMEDIATE_PEM = "globalsign_rsa_ov_ssl_ca_2018.pem"
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
logger = logging.getLogger(__name__)
_GRAPHQL_LENGTH_ERROR = "Cannot read properties of undefined (reading 'length')"


def _format_graphql_errors(errors: object) -> str:
    if isinstance(errors, list):
        messages = []
        for err in errors:
            if isinstance(err, dict) and "message" in err:
                messages.append(str(err["message"]))
            else:
                messages.append(str(err))
        return "; ".join(messages)
    return str(errors)


def _raise_graphql_error(errors: object) -> None:
    message = _format_graphql_errors(errors)
    hint = ""
    if _GRAPHQL_LENGTH_ERROR in message:
        hint = (
            "RegulonDB GraphQL inventory appears to be failing server-side. "
            "Remote inventory may be unavailable; try again later or scope to cache "
            "(`cruncher sources summary --scope cache`). If you only need a quick check, "
            "use `--remote-limit` to reduce load."
        )
    if hint:
        raise RuntimeError(f"RegulonDB GraphQL error: {message}. {hint}")
    raise RuntimeError(f"RegulonDB GraphQL error: {message}")


def _expect_mapping(container: dict, key: str, context: str) -> dict:
    value = container.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"RegulonDB {context} response missing '{key}'.")
    return value


def _expect_list_field(container: dict, key: str, context: str) -> list:
    value = container.get(key)
    if value is None:
        raise RuntimeError(f"RegulonDB {context} response missing '{key}'.")
    if not isinstance(value, list):
        raise RuntimeError(f"RegulonDB {context} response '{key}' must be a list.")
    return value


def _expect_list_nested(container: dict, key: str, subkey: str, context: str) -> list:
    root = _expect_mapping(container, key, context)
    value = root.get(subkey)
    if value is None:
        raise RuntimeError(f"RegulonDB {context} response missing '{key}.{subkey}'.")
    if not isinstance(value, list):
        raise RuntimeError(f"RegulonDB {context} response '{key}.{subkey}' must be a list.")
    return value


def _extract_regulon_items(data: dict, *, root_key: str, context: str) -> list[dict]:
    items = _expect_list_nested(data, root_key, "data", context)
    return items


def _load_regulondb_intermediate() -> str:
    try:
        pem_path = resources.files("dnadesign.cruncher.ingest.certs").joinpath(_REGULONDB_INTERMEDIATE_PEM)
        return pem_path.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "RegulonDB intermediate CA bundle is missing from package data; reinstall cruncher."
        ) from exc


def _build_ssl_context(config: "RegulonDBAdapterConfig") -> ssl.SSLContext:
    if not config.verify_ssl:
        return ssl._create_unverified_context()
    context = ssl.create_default_context(cafile=certifi.where())
    if config.ca_bundle:
        context.load_verify_locations(cafile=str(config.ca_bundle))
        return context
    context.load_verify_locations(cadata=_load_regulondb_intermediate())
    return context


@dataclass(frozen=True, slots=True)
class RegulonDBAdapterConfig:
    base_url: str = _REGULONDB_URL
    verify_ssl: bool = True
    ca_bundle: Optional[str] = None
    timeout_seconds: int = 30
    retry_policy: HttpRetryPolicy = field(default_factory=HttpRetryPolicy)
    motif_matrix_source: str = "alignment"  # alignment | sites
    alignment_matrix_semantics: str = "probabilities"  # probabilities | counts
    min_sites_for_pwm: int = 2
    pseudocounts: float = 0.5
    allow_low_sites: bool = False
    curated_sites: bool = True
    ht_sites: bool = False
    ht_dataset_sources: Optional[List[str]] = None
    ht_dataset_type: str = "TFBINDING"
    ht_binding_mode: str = "tfbinding"  # tfbinding | peaks
    uppercase_binding_site_only: bool = True


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


def _tokenize_line(line: str) -> list[str]:
    return [tok for tok in re.split(r"[,\s]+", line.strip()) if tok]


def _parse_float(token: str) -> float:
    if not _FLOAT_RE.match(token):
        raise ValueError(f"invalid numeric token: {token}")
    return float(token)


def _parse_alignment_matrix(text: str) -> List[List[float]]:
    if not text or not text.strip():
        raise ValueError("alignment matrix payload is empty")
    raw = text.strip()
    # JSON payload
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        if all(isinstance(row, list) for row in parsed):
            if all(len(row) == 4 for row in parsed):
                return [[float(v) for v in row] for row in parsed]
            if len(parsed) == 4 and all(isinstance(row, list) for row in parsed):
                lengths = {len(row) for row in parsed}
                if len(lengths) != 1:
                    raise ValueError("alignment matrix JSON rows must have equal length")
                length = lengths.pop()
                return [
                    [
                        float(parsed[0][i]),
                        float(parsed[1][i]),
                        float(parsed[2][i]),
                        float(parsed[3][i]),
                    ]
                    for i in range(length)
                ]
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("alignment matrix has no data rows")
    tokens = [_tokenize_line(line) for line in lines]
    # header row A C G T
    if len(tokens) > 1 and len(tokens[0]) == 4 and all(tok.upper() in "ACGT" for tok in tokens[0]):
        rows = []
        for row in tokens[1:]:
            if len(row) != 4:
                raise ValueError("alignment matrix rows must have 4 numeric columns")
            rows.append([_parse_float(v) for v in row])
        return rows
    # base-labeled rows
    if all(row and row[0].upper() in "ACGT" for row in tokens):
        base_rows: Dict[str, list[float]] = {}
        for row in tokens:
            base = row[0].upper()
            nums = [_parse_float(v) for v in row[1:]]
            base_rows[base] = nums
        if set(base_rows.keys()) != {"A", "C", "G", "T"}:
            raise ValueError("alignment matrix must include A/C/G/T rows")
        lengths = {len(vals) for vals in base_rows.values()}
        if len(lengths) != 1:
            raise ValueError("alignment matrix base rows must have equal length")
        length = lengths.pop()
        return [[base_rows["A"][i], base_rows["C"][i], base_rows["G"][i], base_rows["T"][i]] for i in range(length)]
    # position rows with 4 numeric columns
    if all(len(row) == 4 for row in tokens):
        return [[_parse_float(v) for v in row] for row in tokens]
    raise ValueError("unrecognized alignment matrix format")


def _parse_alignment_sequences(text: str) -> List[str]:
    if not text or not text.strip():
        raise ValueError("alignment payload is empty")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("alignment payload has no sequences")
    sequences: List[str] = []
    if any(line.startswith(">") for line in lines):
        current: list[str] = []
        for line in lines:
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            sequences.append("".join(current))
    else:
        sequences = lines
    cleaned: List[str] = []
    for seq in sequences:
        seq = seq.strip().upper()
        if not seq:
            continue
        if any(ch not in "ACGT-" for ch in seq):
            raise ValueError("alignment sequences must contain only A/C/G/T/- characters")
        cleaned.append(seq)
    if not cleaned:
        raise ValueError("alignment sequences are empty after cleaning")
    return cleaned


def _compute_pwm_from_alignment(sequences: List[str]) -> List[List[float]]:
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError("alignment sequences must be the same length")
    length = lengths.pop()
    matrix: List[List[float]] = []
    for i in range(length):
        counts = Counter()
        for seq in sequences:
            base = seq[i]
            if base in "ACGT":
                counts[base] += 1
        total = sum(counts.get(b, 0) for b in "ACGT")
        if total == 0:
            raise ValueError("alignment column has no A/C/G/T bases")
        matrix.append(
            [
                counts.get("A", 0) / total,
                counts.get("C", 0) / total,
                counts.get("G", 0) / total,
                counts.get("T", 0) / total,
            ]
        )
    return matrix


class RegulonDBAdapter:
    source_id = "regulondb"

    def __init__(
        self,
        config: Optional[RegulonDBAdapterConfig] = None,
        transport: Optional[Callable[[str, Dict[str, object]], dict]] = None,
    ) -> None:
        self._config = config or RegulonDBAdapterConfig()
        if self._config.verify_ssl and self._config.ca_bundle:
            ca_path = Path(self._config.ca_bundle)
            if not ca_path.exists():
                raise FileNotFoundError(f"RegulonDB CA bundle not found: {ca_path}")
        self._transport = transport or self._post_graphql
        if self._config.ht_binding_mode not in {"tfbinding", "peaks"}:
            raise ValueError("ht_binding_mode must be one of: tfbinding, peaks")
        self._ht_sources_cache: Optional[List[str]] = None
        self._ht_dataset_types_cache: Optional[List[str]] = None
        self._ht_datasets_cache: Dict[str, List[dict]] = {}
        self._ht_dataset_meta: Dict[str, dict] = {}

    def capabilities(self) -> Set[str]:
        caps = {"motifs:list", "motifs:get", "motifs:iter"}
        if self._config.curated_sites or self._config.ht_sites:
            caps.add("sites:list")
        if self._config.ht_sites:
            caps.add("sites:ht")
            if self._config.ht_binding_mode == "tfbinding":
                caps.add("sites:ht-tfbinding")
            if self._config.ht_binding_mode == "peaks":
                caps.add("sites:ht-peaks")
        caps.add("datasets:list")
        return caps

    def _post_graphql(self, query: str, variables: Dict[str, object]) -> dict:
        data = json.dumps({"query": query, "variables": variables}).encode()
        context = _build_ssl_context(self._config)
        try:
            payload = request_json(
                self._config.base_url,
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=self._config.timeout_seconds,
                context=context,
                retry=self._config.retry_policy,
            )
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise RuntimeError(f"RegulonDB HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            hint = (
                "RegulonDB TLS chain is incomplete; cruncher ships the current intermediate CA. "
                "If this persists, set ingest.regulondb.ca_bundle to an updated CA bundle "
                "(or verify_ssl=false as a last resort)."
            )
            raise RuntimeError(f"RegulonDB connection error: {exc}. {hint}") from exc
        if "errors" in payload:
            _raise_graphql_error(payload["errors"])
        data_obj = payload.get("data")
        if not isinstance(data_obj, dict):
            raise RuntimeError("RegulonDB GraphQL response missing data object.")
        return data_obj

    def _get_regulon(self, search: str, limit: int = 20, *, exact: bool = False) -> Optional[dict]:
        data = self._transport(_REGULON_DETAIL_QUERY, {"search": search, "limit": limit, "page": 0})
        items = _extract_regulon_items(data, root_key="getRegulonBy", context="regulon detail")
        if not items:
            return None
        if not exact:
            return items[0]
        if len(items) == 1:
            return items[0]
        # Try to disambiguate by exact id or exact TF name matches.
        matches = [
            item
            for item in items
            if item.get("_id") == search or self._match_tf_name(search, item.get("regulator") or {})
        ]
        if len(matches) == 1:
            return matches[0]
        candidates = ", ".join(str(item.get("_id")) for item in items)
        raise ValueError(
            f"Ambiguous regulon search '{search}'. Candidates: {candidates}. Use --motif-id or lockfile resolution."
        )

    def _resolve_motif_search(self, query: MotifQuery) -> Optional[str]:
        if query.tf_name is not None:
            search = query.tf_name.strip()
            if not search:
                raise ValueError("tf_name must be non-empty for motif search")
            return search
        if query.regex is not None:
            search = query.regex.strip()
            if not search:
                raise ValueError("regex must be non-empty for motif search")
            return search
        return None

    def _collect_tf_candidates(self, regulator: dict) -> list[str]:
        candidates: list[str] = []
        for key in ("abbreviatedName", "name", "synonyms"):
            value = regulator.get(key)
            if not value:
                continue
            if isinstance(value, str):
                candidates.extend([v.strip() for v in value.replace(";", ",").split(",") if v.strip()])
            elif isinstance(value, list):
                candidates.extend([str(v).strip() for v in value if str(v).strip()])
        seen: set[str] = set()
        deduped = []
        for cand in candidates:
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            deduped.append(cand)
        return deduped

    def _extract_tf_name(self, item: dict) -> str:
        regulator = item.get("regulator") or {}
        return regulator.get("abbreviatedName") or regulator.get("name") or item.get("_id")

    def _extract_synonyms(self, item: dict) -> list[str]:
        regulator = item.get("regulator") or {}
        return self._collect_tf_candidates(regulator)

    def _organism_from_item(self, item: dict) -> Optional[OrganismRef]:
        org = item.get("organism") or {}
        name = org.get("name")
        if name is None and org.get("taxon") is None and org.get("strain") is None and org.get("assembly") is None:
            return None
        return OrganismRef(
            taxon=org.get("taxon"),
            name=name,
            strain=org.get("strain"),
            assembly=org.get("assembly"),
        )

    def _alignment_matrix_from_regulon(self, item: dict) -> List[List[float]]:
        matrix_payload = (item.get("aligmentMatrix") or {}).get("matrix")
        alignment_payload = (item.get("aligmentMatrix") or {}).get("aligment")
        if matrix_payload:
            matrix = _parse_alignment_matrix(matrix_payload)
        elif alignment_payload:
            sequences = _parse_alignment_sequences(alignment_payload)
            matrix = _compute_pwm_from_alignment(sequences)
        else:
            raise ValueError(
                "alignment matrix is missing for regulon. "
                "Hint: set ingest.regulondb.motif_matrix_source=sites and "
                "use `cruncher fetch sites` to build PWMs from binding sites."
            )
        if self._config.alignment_matrix_semantics == "counts":
            normalized = []
            for row in matrix:
                total = sum(row)
                if total <= 0:
                    raise ValueError("alignment matrix row has zero total")
                normalized.append([v / total for v in row])
            return normalized
        if self._config.alignment_matrix_semantics != "probabilities":
            raise ValueError("alignment_matrix_semantics must be 'probabilities' or 'counts'")
        return matrix

    def _binding_sites_from_regulon(self, item: dict) -> list[dict]:
        sites = []
        for inter in item.get("regulatoryInteractions") or []:
            site = inter.get("regulatoryBindingSites")
            if site:
                site = dict(site)
                site["_interaction_id"] = inter.get("_id")
                sites.append(site)
        return sites

    def _normalize_strand(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        raw = value.strip().lower()
        if raw in {"+", "forward", "plus"}:
            return "+"
        if raw in {"-", "reverse", "minus"}:
            return "-"
        raise ValueError(f"unrecognized strand value: {value}")

    def _site_instance_from_curated(
        self,
        site: dict,
        motif_ref: str,
        organism: Optional[OrganismRef],
    ) -> Optional[SiteInstance]:
        raw_seq = site.get("sequence")
        seq = None
        if raw_seq is not None and str(raw_seq).strip():
            seq = normalize_site_sequence(raw_seq, self._config.uppercase_binding_site_only)
        left = site.get("leftEndPosition")
        right = site.get("rightEndPosition")
        coord = None
        if left is not None and right is not None:
            start = int(left) - 1
            end = int(right)
            if start < 0 or end <= start:
                raise ValueError("invalid curated binding-site coordinates")
            coord = GenomicInterval(contig="unknown", start=start, end=end, assembly=None)
        if seq is None and coord is None:
            return None
        provenance = Provenance(
            retrieved_at=datetime.now(timezone.utc),
            source_url=self._config.base_url,
            tags={
                "coord_system": "regulondb_1based_inclusive",
                "record_kind": "curated",
            },
        )
        return SiteInstance(
            source=self.source_id,
            site_id=site.get("_id") or "",
            motif_ref=motif_ref,
            organism=organism,
            coordinate=coord,
            sequence=seq,
            strand=self._normalize_strand(site.get("strand")),
            score=None,
            evidence={"interaction_id": site.get("_interaction_id")},
            provenance=provenance,
        )

    def _list_ht_sources(self) -> List[str]:
        if self._config.ht_dataset_sources is not None:
            return self._config.ht_dataset_sources
        if self._ht_sources_cache is not None:
            return self._ht_sources_cache
        data = self._transport(_HT_SOURCES_QUERY, {})
        sources = _expect_list_field(data, "listAllHTSources", "HT sources")
        if not sources:
            raise RuntimeError("RegulonDB returned no HT sources.")
        self._ht_sources_cache = sources
        return sources

    def _list_ht_dataset_types(self) -> List[str]:
        if self._ht_dataset_types_cache is not None:
            return self._ht_dataset_types_cache
        data = self._transport(_HT_DATASET_TYPES_QUERY, {})
        types = _expect_list_field(data, "listAllDatasetTypes", "HT dataset types")
        if not types:
            raise RuntimeError("RegulonDB returned no dataset types for HT datasets.")
        self._ht_dataset_types_cache = types
        return types

    def _validate_ht_dataset_type(self) -> None:
        types = self._list_ht_dataset_types()
        if self._config.ht_dataset_type not in types:
            available = ", ".join(sorted(types))
            raise ValueError(
                f"Unknown RegulonDB dataset type '{self._config.ht_dataset_type}'. Available types: {available}"
            )

    def _list_ht_datasets(self, source: str) -> List[dict]:
        if source in self._ht_datasets_cache:
            return self._ht_datasets_cache[source]
        self._validate_ht_dataset_type()
        data = self._transport(
            _HT_DATASETS_QUERY,
            {"datasetType": self._config.ht_dataset_type, "source": source},
        )
        datasets = _expect_list_nested(data, "getDatasetsWithMetadata", "datasets", "HT dataset metadata")
        for dataset in datasets:
            dataset_id = dataset.get("_id")
            if dataset_id:
                self._ht_dataset_meta[dataset_id] = dataset
        self._ht_datasets_cache[source] = datasets
        return datasets

    def list_datasets(self, query: DatasetQuery) -> list[DatasetDescriptor]:
        datasets: list[DatasetDescriptor] = []
        seen: set[str] = set()
        for source in self._list_ht_sources():
            if query.dataset_source and query.dataset_source != source:
                continue
            raw = self._list_ht_datasets(source)
            for dataset in raw:
                dataset_id = dataset.get("_id")
                if not dataset_id:
                    continue
                if dataset_id in seen:
                    continue
                collection = dataset.get("collectionData") or {}
                dataset_type = collection.get("type") or dataset.get("datasetType")
                dataset_source = collection.get("source") or source
                if query.dataset_source and dataset_source != query.dataset_source:
                    continue
                objects = dataset.get("objectsTested") or []
                tf_names = tuple(sorted({name for obj in objects for name in self._collect_tf_candidates(obj)}))
                if query.dataset_type and dataset_type != query.dataset_type:
                    continue
                if query.tf_name:
                    tf_norm = query.tf_name.strip().lower()
                    if not any(tf_norm == name.lower() for name in tf_names):
                        continue
                descriptor = DatasetDescriptor(
                    source=self.source_id,
                    dataset_id=dataset_id,
                    dataset_type=dataset_type,
                    dataset_source=dataset_source,
                    method=collection.get("type"),
                    tf_names=tf_names,
                    reference_genome=dataset.get("referenceGenome") or dataset.get("assemblyGenomeId"),
                )
                datasets.append(descriptor)
                seen.add(dataset_id)
        return datasets

    def _match_tf_name(self, tf_name: str, obj: dict) -> bool:
        tf_norm = tf_name.strip().lower()
        candidates = self._collect_tf_candidates(obj)
        return any(tf_norm == cand.lower() for cand in candidates)

    def _resolve_ht_contig(self, contig: Optional[str], assembly: Optional[str]) -> str:
        if contig:
            norm = contig.strip().lower()
            if norm not in {"chr", "chromosome", "chrom"}:
                return contig
        if assembly:
            return assembly
        return contig or "unknown"

    def _find_ht_datasets_for_tf(self, tf_name: str) -> List[str]:
        dataset_ids: List[str] = []
        seen: set[str] = set()
        for source in self._list_ht_sources():
            datasets = self._list_ht_datasets(source)
            for dataset in datasets:
                objects_tested = dataset.get("objectsTested") or []
                if any(self._match_tf_name(tf_name, obj) for obj in objects_tested):
                    dataset_id = dataset["_id"]
                    if dataset_id in seen:
                        continue
                    seen.add(dataset_id)
                    dataset_ids.append(dataset_id)
        return dataset_ids

    def _iter_ht_sites(self, dataset_id: str, limit: Optional[int]) -> Iterable[SiteInstance]:
        page = 0
        page_size = 500
        yielded = 0
        dataset_meta = self._ht_dataset_meta.get(dataset_id, {})
        assembly = dataset_meta.get("referenceGenome") or dataset_meta.get("assemblyGenomeId")
        collection = dataset_meta.get("collectionData") or {}
        dataset_source = collection.get("source")
        dataset_method = collection.get("type")
        while True:
            data = self._transport(
                _HT_TF_BINDING_QUERY,
                {"datasetId": dataset_id, "limit": page_size, "page": page},
            )
            items = _expect_list_field(data, "getAllTFBindingOfDataset", "HT TF binding")
            if not items:
                break
            for item in items:
                left = item.get("chrLeftPosition")
                right = item.get("chrRightPosition")
                coord = None
                if left is not None and right is not None:
                    start = int(left) - 1
                    end = int(right)
                    if start < 0 or end <= start:
                        raise ValueError("invalid HT binding-site coordinates")
                    contig = self._resolve_ht_contig(item.get("chromosome"), assembly)
                    coord = GenomicInterval(
                        contig=contig,
                        start=start,
                        end=end,
                        assembly=assembly,
                    )
                seq = normalize_site_sequence(item.get("sequence"), False) if item.get("sequence") else None
                tags = {
                    "coord_system": "regulondb_1based_inclusive",
                    "record_kind": "ht_tfbinding",
                    "dataset_id": dataset_id,
                }
                if assembly:
                    tags["reference_genome"] = assembly
                if dataset_source:
                    tags["dataset_source"] = dataset_source
                if dataset_method:
                    tags["dataset_method"] = dataset_method
                provenance = Provenance(
                    retrieved_at=datetime.now(timezone.utc),
                    source_url=self._config.base_url,
                    tags=tags,
                )
                site = SiteInstance(
                    source=self.source_id,
                    site_id=item.get("_id") or "",
                    motif_ref=f"{self.source_id}:dataset:{dataset_id}",
                    organism=None,
                    coordinate=coord,
                    sequence=seq,
                    strand=self._normalize_strand(item.get("strand")),
                    score=item.get("score"),
                    evidence={
                        "peak_id": item.get("peakId"),
                        "dataset_ids": item.get("datasetIds"),
                    },
                    provenance=provenance,
                )
                yield site
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            page += 1

    def _iter_ht_peaks(self, dataset_id: str, limit: Optional[int]) -> Iterable[SiteInstance]:
        page = 0
        page_size = 500
        yielded = 0
        dataset_meta = self._ht_dataset_meta.get(dataset_id, {})
        assembly = dataset_meta.get("referenceGenome") or dataset_meta.get("assemblyGenomeId")
        collection = dataset_meta.get("collectionData") or {}
        dataset_source = collection.get("source")
        dataset_method = collection.get("type")
        while True:
            data = self._transport(
                _HT_PEAKS_QUERY,
                {"datasetId": dataset_id, "limit": page_size, "page": page},
            )
            items = _expect_list_field(data, "getAllPeaksOfDataset", "HT peaks")
            if not items:
                break
            for item in items:
                left = item.get("peakLeftPosition")
                right = item.get("peakRightPosition")
                coord = None
                if left is not None and right is not None:
                    start = int(left) - 1
                    end = int(right)
                    if start < 0 or end <= start:
                        raise ValueError("invalid HT peak coordinates")
                    contig = self._resolve_ht_contig(item.get("chromosome"), assembly)
                    coord = GenomicInterval(
                        contig=contig,
                        start=start,
                        end=end,
                        assembly=assembly,
                    )
                tags = {
                    "coord_system": "regulondb_1based_inclusive",
                    "record_kind": "ht_peak",
                    "dataset_id": dataset_id,
                }
                if assembly:
                    tags["reference_genome"] = assembly
                if dataset_source:
                    tags["dataset_source"] = dataset_source
                if dataset_method:
                    tags["dataset_method"] = dataset_method
                provenance = Provenance(
                    retrieved_at=datetime.now(timezone.utc),
                    source_url=self._config.base_url,
                    tags=tags,
                )
                site = SiteInstance(
                    source=self.source_id,
                    site_id=item.get("_id") or "",
                    motif_ref=f"{self.source_id}:dataset:{dataset_id}",
                    organism=None,
                    coordinate=coord,
                    sequence=None,
                    strand=None,
                    score=item.get("score"),
                    evidence={
                        "site_ids": item.get("siteIds"),
                        "dataset_ids": item.get("datasetIds"),
                    },
                    provenance=provenance,
                )
                yield site
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            page += 1

    def list_motifs(self, query: MotifQuery) -> list[MotifDescriptor]:
        search = self._resolve_motif_search(query)
        if search:
            data = self._transport(
                _REGULON_LIST_QUERY,
                {"search": search, "limit": query.limit or 50, "page": 0},
            )
            items = _extract_regulon_items(data, root_key="getRegulonBy", context="regulon search")
        else:
            data = self._transport(
                _REGULON_ALL_QUERY,
                {"limit": query.limit or 50, "page": 0},
            )
            items = _extract_regulon_items(data, root_key="getAllRegulon", context="regulon inventory")
        descriptors: list[MotifDescriptor] = []
        for item in items:
            name = self._extract_tf_name(item)
            org = self._organism_from_item(item)
            desc = MotifDescriptor(
                source=self.source_id,
                motif_id=item.get("_id"),
                tf_name=name,
                organism=org,
                length=0,
                kind="PFM",
            )
            descriptors.append(desc)
        return descriptors

    def iter_motifs(self, query: MotifQuery, *, page_size: int = 200) -> Iterable[MotifDescriptor]:
        search = self._resolve_motif_search(query)
        page = 0
        yielded = 0
        while True:
            if search:
                data = self._transport(
                    _REGULON_LIST_QUERY,
                    {"search": search, "limit": page_size, "page": page},
                )
                items = _extract_regulon_items(data, root_key="getRegulonBy", context="regulon search")
            else:
                data = self._transport(
                    _REGULON_ALL_QUERY,
                    {"limit": page_size, "page": page},
                )
                items = _extract_regulon_items(data, root_key="getAllRegulon", context="regulon inventory")
            if not items:
                break
            for item in items:
                name = self._extract_tf_name(item)
                org = self._organism_from_item(item)
                yield MotifDescriptor(
                    source=self.source_id,
                    motif_id=item.get("_id"),
                    tf_name=name,
                    organism=org,
                    length=0,
                    kind="PFM",
                )
                yielded += 1
                if query.limit is not None and yielded >= query.limit:
                    return
            page += 1

    def get_motif(self, motif_id: str) -> MotifRecord:
        item = self._get_regulon(motif_id, exact=True)
        if not item:
            raise ValueError(f"No regulon found for {motif_id}")
        name = self._extract_tf_name(item)
        organism = self._organism_from_item(item)
        tags = {}
        synonyms = self._extract_synonyms(item)
        if synonyms:
            tags["synonyms"] = ";".join(synonyms)
        if self._config.motif_matrix_source == "alignment":
            matrix = self._alignment_matrix_from_regulon(item)
            matrix_source = "alignment"
            tags["matrix_source"] = matrix_source
        elif self._config.motif_matrix_source == "sites":
            sites = self._binding_sites_from_regulon(item)
            sequences = []
            for site in sites:
                raw_seq = site.get("sequence")
                if raw_seq is None or not str(raw_seq).strip():
                    continue
                sequences.append(normalize_site_sequence(raw_seq, self._config.uppercase_binding_site_only))
            if not sequences:
                raise ValueError(f"No curated binding sites available for {name}")
            try:
                matrix, site_count = compute_pwm_from_sites(
                    sequences,
                    min_sites=self._config.min_sites_for_pwm,
                    strict_min_sites=not self._config.allow_low_sites,
                    return_count=True,
                    pseudocounts=self._config.pseudocounts,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unable to build PWM from curated sites for {name}: {exc}. "
                    "Use `cruncher fetch sites` and let cruncher.catalog.site_window_lengths "
                    "window variable-length sites during parse/sample."
                ) from exc
            matrix_source = "sites"
            tags["matrix_source"] = matrix_source
            tags["site_count"] = str(site_count)
            tags["pwm_backend"] = "biopython"
            tags["pseudocounts"] = str(self._config.pseudocounts)
        else:
            raise ValueError("motif_matrix_source must be 'alignment' or 'sites'")
        raw_payload = json.dumps(item, sort_keys=True)
        return build_motif_record(
            source=self.source_id,
            motif_id=item.get("_id"),
            tf_name=name,
            matrix=matrix,
            matrix_semantics="probabilities",
            organism=organism,
            raw_payload=raw_payload,
            retrieved_at=datetime.now(timezone.utc),
            source_url=self._config.base_url,
            tags=tags,
        )

    def list_sites(self, query: SiteQuery) -> Iterable[SiteInstance]:
        if not query.tf_name and not query.motif_id:
            raise ValueError("site queries require tf_name or motif_id")
        if self._config.curated_sites and self._config.ht_sites and query.limit is not None and not query.dataset_id:
            raise ValueError(
                "site limit with curated+HT retrieval requires explicit source mode: "
                "pin --dataset-id, disable curated_sites, or omit --limit."
            )
        remaining = query.limit
        if self._config.curated_sites and not query.dataset_id:
            search = query.motif_id or query.tf_name
            item = self._get_regulon(search, exact=True)
            if not item:
                raise ValueError(f"No regulon found for {search}")
            motif_ref = f"{self.source_id}:{item.get('_id')}"
            organism = self._organism_from_item(item)
            for site in self._binding_sites_from_regulon(item):
                try:
                    site_obj = self._site_instance_from_curated(site, motif_ref, organism)
                except ValueError as exc:
                    logger.warning("Skipping curated site %s: %s", site.get("_id"), exc)
                    continue
                if site_obj is None:
                    continue
                yield site_obj
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return
            tf_name = self._extract_tf_name(item)
        else:
            tf_name = query.tf_name
        if self._config.ht_sites:
            if not tf_name:
                raise ValueError("HT site queries require tf_name")
            if query.dataset_id:
                dataset_ids = [query.dataset_id]
            else:
                dataset_ids = self._find_ht_datasets_for_tf(tf_name)
            if not dataset_ids:
                raise ValueError(f"No HT datasets found for TF {tf_name}")
            yielded = 0
            for dataset_id in dataset_ids:
                if self._config.ht_binding_mode == "tfbinding":
                    for site in self._iter_ht_sites(dataset_id, remaining):
                        yield site
                        yielded += 1
                        if remaining is not None:
                            remaining -= 1
                            if remaining <= 0:
                                return
                if self._config.ht_binding_mode == "peaks":
                    for site in self._iter_ht_peaks(dataset_id, remaining):
                        yield site
                        yielded += 1
                        if remaining is not None:
                            remaining -= 1
                            if remaining <= 0:
                                return
            if self._config.ht_binding_mode == "tfbinding":
                if yielded == 0:
                    if query.dataset_id:
                        raise ValueError(
                            f"No HT TFBinding records returned for dataset {query.dataset_id} (TF {tf_name})"
                        )
                    raise ValueError(
                        f"No HT binding-site records returned for TF {tf_name}. "
                        "If datasets expose peaks only, set ingest.regulondb.ht_binding_mode=peaks."
                    )
            if self._config.ht_binding_mode == "peaks":
                if yielded == 0:
                    if query.dataset_id:
                        raise ValueError(f"No HT peak records returned for dataset {query.dataset_id} (TF {tf_name})")
                    raise ValueError(f"No HT peak records returned for TF {tf_name}")
        elif query.dataset_id:
            raise ValueError("HT dataset_id requires ht_sites=true in config.")

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery) -> Iterable[SiteInstance]:
        site_query = SiteQuery(
            organism=query.organism,
            motif_id=motif_id,
            tf_name=query.tf_name,
            dataset_id=query.dataset_id,
            region=query.region,
            limit=query.limit,
        )
        return self.list_sites(site_query)
