"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/registry.py

Register and construct ingest source adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from dnadesign.cruncher.config.schema_v3 import (
    IngestConfig,
    LocalMotifSourceConfig,
    LocalSiteSourceConfig,
)
from dnadesign.cruncher.ingest.adapters.base import SourceAdapter
from dnadesign.cruncher.ingest.adapters.local import (
    LocalMotifAdapter,
    LocalMotifAdapterConfig,
)
from dnadesign.cruncher.ingest.adapters.local_sites import (
    LocalSiteAdapter,
    LocalSiteAdapterConfig,
)
from dnadesign.cruncher.ingest.adapters.regulondb import (
    RegulonDBAdapter,
    RegulonDBAdapterConfig,
)
from dnadesign.cruncher.ingest.http_client import HttpRetryPolicy
from dnadesign.cruncher.ingest.models import OrganismRef
from dnadesign.cruncher.utils.paths import resolve_workspace_root

AdapterFactory = Callable[[IngestConfig], SourceAdapter]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    description: str


class SourceRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, AdapterFactory] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, source_id: str, factory: AdapterFactory, description: str) -> None:
        if source_id in self._factories:
            raise ValueError(f"Source '{source_id}' is already registered")
        self._factories[source_id] = factory
        self._descriptions[source_id] = description

    def create(self, source_id: str, ingest_config: IngestConfig) -> SourceAdapter:
        if source_id not in self._factories:
            raise ValueError(f"Unknown source '{source_id}'. Run `cruncher sources list`.")
        return self._factories[source_id](ingest_config)

    def list_sources(self) -> List[SourceSpec]:
        return [SourceSpec(source_id=k, description=self._descriptions[k]) for k in sorted(self._factories)]


def _build_regulondb(config: IngestConfig) -> SourceAdapter:
    reg_cfg = config.regulondb
    retry_policy = HttpRetryPolicy(
        retries=config.http.retries,
        backoff_seconds=config.http.backoff_seconds,
        max_backoff_seconds=config.http.max_backoff_seconds,
        retry_statuses=tuple(config.http.retry_statuses),
        respect_retry_after=config.http.respect_retry_after,
    )
    return RegulonDBAdapter(
        RegulonDBAdapterConfig(
            base_url=reg_cfg.base_url,
            verify_ssl=reg_cfg.verify_ssl,
            ca_bundle=str(reg_cfg.ca_bundle) if reg_cfg.ca_bundle else None,
            timeout_seconds=reg_cfg.timeout_seconds,
            retry_policy=retry_policy,
            motif_matrix_source=reg_cfg.motif_matrix_source,
            alignment_matrix_semantics=reg_cfg.alignment_matrix_semantics,
            min_sites_for_pwm=reg_cfg.min_sites_for_pwm,
            pseudocounts=reg_cfg.pseudocounts,
            allow_low_sites=reg_cfg.allow_low_sites,
            curated_sites=reg_cfg.curated_sites,
            ht_sites=reg_cfg.ht_sites,
            ht_dataset_sources=reg_cfg.ht_dataset_sources,
            ht_dataset_type=reg_cfg.ht_dataset_type,
            ht_binding_mode=reg_cfg.ht_binding_mode,
            uppercase_binding_site_only=reg_cfg.uppercase_binding_site_only,
        )
    )


def _resolve_root(root: Path, *, config_path: Path | None) -> Path:
    if root.is_absolute() or config_path is None:
        return root
    return (resolve_workspace_root(config_path) / root).resolve()


def _resolve_path(path: Path, *, config_path: Path | None) -> Path:
    if path.is_absolute() or config_path is None:
        return path
    return (resolve_workspace_root(config_path) / path).resolve()


def _local_source_factory(
    src: LocalMotifSourceConfig,
    *,
    config_path: Path | None,
    extra_parser_modules: Sequence[str] | None,
) -> AdapterFactory:
    root = _resolve_root(src.root, config_path=config_path)
    format_map: dict[str, str] = {}
    for ext, fmt in src.format_map.items():
        ext_key = ext.lower().strip()
        if not ext_key:
            raise ValueError(f"Local source '{src.source_id}' format_map contains an empty extension.")
        if not ext_key.startswith("."):
            ext_key = f".{ext_key}"
        fmt_name = str(fmt).strip().upper()
        if not fmt_name:
            raise ValueError(f"Local source '{src.source_id}' format_map contains an empty parser format.")
        format_map[ext_key] = fmt_name
    default_format = src.default_format.strip().upper() if src.default_format else None
    organism = None
    if src.organism:
        organism = OrganismRef(
            taxon=src.organism.taxon,
            name=src.organism.name,
            strain=src.organism.strain,
            assembly=src.organism.assembly,
        )
    adapter_config = LocalMotifAdapterConfig(
        source_id=src.source_id,
        root=root,
        patterns=tuple(src.patterns),
        recursive=src.recursive,
        format_map=format_map,
        default_format=default_format,
        tf_name_strategy=src.tf_name_strategy,
        matrix_semantics=src.matrix_semantics,
        organism=organism,
        citation=src.citation,
        license=src.license,
        source_url=src.source_url,
        source_version=src.source_version,
        tags=dict(src.tags),
        extra_parser_modules=tuple(extra_parser_modules or ()),
        extract_sites=src.extract_sites,
        meme_motif_selector=src.meme_motif_selector,
    )

    def _factory(_config: IngestConfig) -> SourceAdapter:
        return LocalMotifAdapter(adapter_config)

    return _factory


def _site_source_factory(
    src: LocalSiteSourceConfig,
    *,
    config_path: Path | None,
) -> AdapterFactory:
    path = _resolve_path(src.path, config_path=config_path)
    organism = None
    if src.organism:
        organism = OrganismRef(
            taxon=src.organism.taxon,
            name=src.organism.name,
            strain=src.organism.strain,
            assembly=src.organism.assembly,
        )
    adapter_config = LocalSiteAdapterConfig(
        source_id=src.source_id,
        path=path,
        tf_name=src.tf_name,
        organism=organism,
        citation=src.citation,
        license=src.license,
        source_url=src.source_url,
        source_version=src.source_version,
        record_kind=src.record_kind,
        tags=dict(src.tags),
    )

    def _factory(_config: IngestConfig) -> SourceAdapter:
        return LocalSiteAdapter(adapter_config)

    return _factory


def default_registry(
    ingest_config: IngestConfig | None = None,
    *,
    config_path: Path | None = None,
    extra_parser_modules: Sequence[str] | None = None,
) -> SourceRegistry:
    registry = SourceRegistry()
    registry.register("regulondb", _build_regulondb, "RegulonDB datamarts GraphQL (curated + HT)")
    if ingest_config:
        for src in ingest_config.local_sources:
            description = src.description or f"Local motifs at {src.root}"
            registry.register(
                src.source_id,
                _local_source_factory(
                    src,
                    config_path=config_path,
                    extra_parser_modules=extra_parser_modules,
                ),
                description,
            )
        for src in ingest_config.site_sources:
            description = src.description or f"Local sites at {src.path}"
            registry.register(
                src.source_id,
                _site_source_factory(
                    src,
                    config_path=config_path,
                ),
                description,
            )
    return registry
