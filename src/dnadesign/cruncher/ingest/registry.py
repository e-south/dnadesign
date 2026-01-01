"""Source adapter registry for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from dnadesign.cruncher.config.schema_v2 import IngestConfig
from dnadesign.cruncher.ingest.adapters.base import SourceAdapter
from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.ingest.http_client import HttpRetryPolicy

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
            allow_low_sites=reg_cfg.allow_low_sites,
            curated_sites=reg_cfg.curated_sites,
            ht_sites=reg_cfg.ht_sites,
            ht_dataset_sources=reg_cfg.ht_dataset_sources,
            ht_dataset_type=reg_cfg.ht_dataset_type,
            ht_binding_mode=reg_cfg.ht_binding_mode,
            uppercase_binding_site_only=reg_cfg.uppercase_binding_site_only,
        )
    )


_DEFAULT_REGISTRY: SourceRegistry | None = None


def default_registry() -> SourceRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = SourceRegistry()
        registry.register("regulondb", _build_regulondb, "RegulonDB datamarts GraphQL (curated + HT)")
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
